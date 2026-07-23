#include "QuadrotorDynamicsPlugin.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo {

namespace {

constexpr double kIntegralClamp = 40.0;

double Clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

}  // namespace

void QuadrotorDynamicsPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf) {
  model_ = model;
  if (!model_ || !sdf) {
    gzerr << "[QuadrotorDynamicsPlugin] model or SDF is null\n";
    return;
  }

  link_name_ = sdf->Get<std::string>("link_name", "base_link").first;

  target_.X(sdf->Get<double>("target_x", -280.0).first);
  target_.Y(sdf->Get<double>("target_y", -400.0).first);
  target_.Z(sdf->Get<double>("target_z", 50.0).first);

  kp_ = sdf->Get<double>("kp", 6.0).first;
  ki_ = sdf->Get<double>("ki", 0.08).first;
  kd_ = sdf->Get<double>("kd", 3.5).first;
  kp_z_ = sdf->Get<double>("kp_z", kp_).first;
  ki_z_ = sdf->Get<double>("ki_z", ki_).first;
  kd_z_ = sdf->Get<double>("kd_z", kd_).first;
  enable_xy_ = sdf->Get<bool>("enable_xy", true).first;

  att_kp_ = sdf->Get<double>("att_kp", 8.0).first;
  att_kd_ = sdf->Get<double>("att_kd", 1.2).first;
  yaw_kp_ = sdf->Get<double>("yaw_kp", 2.0).first;
  yaw_kd_ = sdf->Get<double>("yaw_kd", 0.4).first;
  max_tilt_rad_ = sdf->Get<double>("max_tilt_rad", 0.55).first;

  k_f_ = sdf->Get<double>("k_f", 5.5e-4).first;
  k_m_ = sdf->Get<double>("k_m", 8.8e-6).first;
  motor_tau_ = sdf->Get<double>("motor_tau", 0.08).first;
  max_omega_ = sdf->Get<double>("max_omega", 520.0).first;
  min_omega_ = sdf->Get<double>("min_omega", 0.0).first;
  max_thrust_per_motor_ = sdf->Get<double>("max_thrust_per_motor", 12.0).first;
  joint_motor_fmax_ = sdf->Get<double>("joint_motor_fmax", 0.5).first;

  log_every_n_ = sdf->Get<int>("log_every_n", 250).first;
  disable_topic_ = sdf->Get<std::string>("disable_topic", std::string()).first;
  crash_zero_thrust_ = sdf->Get<bool>("crash_zero_thrust", true).first;

  auto link = model_->GetLink(link_name_);
  auto world = model_->GetWorld();
  if (link && link->GetInertial()) {
    mass_ = link->GetInertial()->Mass();
  }
  if (world) {
    gravity_z_ = world->Gravity().Length();
  }

  if (sdf->HasElement("rotor")) {
    sdf::ElementPtr el = sdf->GetElement("rotor");
    while (el) {
      Rotor r;
      r.joint_name = el->Get<std::string>("joint", std::string()).first;
      r.arm_body.X(el->Get<double>("x", 0.0).first);
      r.arm_body.Y(el->Get<double>("y", 0.0).first);
      r.arm_body.Z(el->Get<double>("z", 0.023).first);
      r.spin_dir = el->Get<int>("spin_dir", 1).first;
      if (!r.joint_name.empty()) {
        r.joint = model_->GetJoint(r.joint_name);
        if (!r.joint) {
          gzerr << "[QuadrotorDynamicsPlugin] joint not found: " << r.joint_name << "\n";
        } else {
          rotors_.push_back(r);
        }
      }
      el = el->GetNextElement("rotor");
    }
  }

  if (rotors_.size() != 4) {
    gzwarn << "[QuadrotorDynamicsPlugin] expected 4 rotors, got " << rotors_.size()
           << "; control allocation may be ill-conditioned\n";
  }

  last_time_ = world ? world->SimTime() : common::Time::Zero;

  std::ostringstream rotor_summary;
  for (size_t i = 0; i < rotors_.size(); ++i) {
    rotor_summary << rotors_[i].joint_name << "@(" << rotors_[i].arm_body.X() << ","
                  << rotors_[i].arm_body.Y() << "," << rotors_[i].arm_body.Z()
                  << ") dir=" << rotors_[i].spin_dir;
    if (i + 1 < rotors_.size()) rotor_summary << "; ";
  }

  gzmsg << "[QuadrotorDynamicsPlugin] target=(" << target_.X() << "," << target_.Y() << ","
        << target_.Z() << ") mass=" << mass_ << " kg k_f=" << k_f_ << " k_m=" << k_m_
        << " motor_tau=" << motor_tau_ << " s max_omega=" << max_omega_
        << " rotors: " << rotor_summary.str() << "\n";

  if (!disable_topic_.empty()) {
    node_.reset(new transport::Node());
    node_->Init(world ? world->Name() : "");
    disable_sub_ = node_->Subscribe(disable_topic_, &QuadrotorDynamicsPlugin::OnDisableMsg, this);
    gzmsg << "[QuadrotorDynamicsPlugin] subscribed disable_topic=" << disable_topic_ << "\n";
  }

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&QuadrotorDynamicsPlugin::OnUpdate, this));
}

void QuadrotorDynamicsPlugin::OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg) {
  if (disabled_.exchange(true)) return;
  integral_.Set(0.0, 0.0, 0.0);
  prev_err_.Set(0.0, 0.0, 0.0);
  first_step_ = true;
  for (auto& r : rotors_) {
    r.omega_cmd = 0.0;
  }
  gzmsg << "[QuadrotorDynamicsPlugin] disabled by '" << (msg ? msg->data() : std::string("?"))
        << "' on " << disable_topic_ << " (crash_zero_thrust=" << (crash_zero_thrust_ ? 1 : 0) << ")\n";
}

bool QuadrotorDynamicsPlugin::AllocateMotors(double thrust_total, double tau_roll, double tau_pitch,
                                             double tau_yaw, std::array<double, 4>& thrust_out) const {
  const size_t n = rotors_.size();
  if (n != 4) return false;

  // Mixing matrix M: [T_total; tau_roll; tau_pitch; tau_yaw] = M * [T0..T3]
  // tau_roll  = sum(y_i * T_i), tau_pitch = sum(-x_i * T_i)  (body frame, force along +Z)
  // tau_yaw   = (k_m/k_f) * sum(spin_dir_i * T_i)
  const double yaw_scale = (k_f_ > 0.0) ? (k_m_ / k_f_) : 0.0;

  std::array<std::array<double, 4>, 4> M{};
  M[0] = {1.0, 1.0, 1.0, 1.0};
  for (size_t i = 0; i < 4; ++i) {
    M[1][i] = rotors_[i].arm_body.Y();
    M[2][i] = -rotors_[i].arm_body.X();
    M[3][i] = yaw_scale * static_cast<double>(rotors_[i].spin_dir);
  }

  const std::array<double, 4> b{thrust_total, tau_roll, tau_pitch, tau_yaw};

  // Solve 4x4 via Gaussian elimination (well-conditioned for iris arm geometry).
  std::array<std::array<double, 5>, 4> aug{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) aug[i][j] = M[i][j];
    aug[i][4] = b[i];
  }

  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    double max_abs = std::fabs(aug[col][col]);
    for (int row = col + 1; row < 4; ++row) {
      const double v = std::fabs(aug[row][col]);
      if (v > max_abs) {
        max_abs = v;
        pivot = row;
      }
    }
    if (max_abs < 1.0e-9) return false;
    if (pivot != col) std::swap(aug[pivot], aug[col]);

    const double div = aug[col][col];
    for (int j = col; j < 5; ++j) aug[col][j] /= div;

    for (int row = 0; row < 4; ++row) {
      if (row == col) continue;
      const double factor = aug[row][col];
      for (int j = col; j < 5; ++j) aug[row][j] -= factor * aug[col][j];
    }
  }

  for (int i = 0; i < 4; ++i) {
    thrust_out[i] = Clamp(aug[i][4], 0.0, max_thrust_per_motor_);
  }
  return true;
}

void QuadrotorDynamicsPlugin::ApplyRotorForces(physics::LinkPtr link) {
  double tau_yaw_body = 0.0;
  for (auto& r : rotors_) {
    const double T = k_f_ * r.omega * r.omega;
    const ignition::math::Vector3d force_body(0.0, 0.0, T);
    link->AddForceAtRelativePosition(force_body, r.arm_body);
    tau_yaw_body += static_cast<double>(r.spin_dir) * k_m_ * r.omega * r.omega;

    if (r.joint) {
      const double spin_sign = static_cast<double>(r.spin_dir);
      r.joint->SetParam("fmax", 0, joint_motor_fmax_);
      r.joint->SetParam("vel", 0, spin_sign * r.omega);
    }
  }
  link->AddRelativeTorque(ignition::math::Vector3d(0.0, 0.0, tau_yaw_body));
}

void QuadrotorDynamicsPlugin::OnUpdate() {
  if (!model_) return;

  auto link = model_->GetLink(link_name_);
  if (!link) return;

  auto world = model_->GetWorld();
  if (!world) return;

  if (disabled_.load() && crash_zero_thrust_) {
    for (auto& r : rotors_) {
      r.omega = 0.0;
      r.omega_cmd = 0.0;
      if (r.joint) {
        r.joint->SetParam("vel", 0, 0.0);
      }
    }
    return;
  }

  const common::Time now = world->SimTime();
  double dt = (now - last_time_).Double();
  if (dt <= 0.0 || dt > 1.0) {
    auto pe = world->Physics();
    dt = pe ? pe->GetMaxStepSize() : 0.004;
  }
  last_time_ = now;

  const auto pose = link->WorldPose();
  const ignition::math::Vector3d pos = pose.Pos();
  const ignition::math::Vector3d vel = link->WorldLinearVel();
  const ignition::math::Vector3d omega_body = link->RelativeAngularVel();

  const ignition::math::Vector3d err = target_ - pos;
  if (first_step_) {
    prev_err_ = err;
    first_step_ = false;
  }
  const ignition::math::Vector3d derr = (err - prev_err_) / dt;
  prev_err_ = err;

  integral_ += err * dt;
  integral_.X(Clamp(integral_.X(), -kIntegralClamp, kIntegralClamp));
  integral_.Y(Clamp(integral_.Y(), -kIntegralClamp, kIntegralClamp));
  integral_.Z(Clamp(integral_.Z(), -kIntegralClamp, kIntegralClamp));

  double ax = kp_ * err.X() + ki_ * integral_.X() + kd_ * derr.X();
  double ay = kp_ * err.Y() + ki_ * integral_.Y() + kd_ * derr.Y();
  double az = kp_z_ * err.Z() + ki_z_ * integral_.Z() + kd_z_ * derr.Z();

  if (!enable_xy_) {
    ax = 0.0;
    ay = 0.0;
    integral_.X(0.0);
    integral_.Y(0.0);
  }

  // Desired world acceleration including gravity compensation (thrust fights g).
  const ignition::math::Vector3d a_cmd(ax, ay, az + gravity_z_);

  // Thrust vector must align with commanded acceleration direction (body +Z when level).
  ignition::math::Vector3d thrust_w = mass_ * a_cmd;
  const double thrust_mag = thrust_w.Length();
  if (thrust_mag < 1.0e-3) {
    thrust_w = ignition::math::Vector3d(0.0, 0.0, mass_ * gravity_z_);
  } else {
    // Clamp tilt by limiting horizontal component relative to vertical.
    const double tz = thrust_w.Z();
    const double t_horiz = std::hypot(thrust_w.X(), thrust_w.Y());
    const double max_horiz = std::tan(max_tilt_rad_) * std::max(tz, mass_ * gravity_z_ * 0.2);
    if (t_horiz > max_horiz && t_horiz > 1.0e-6) {
      const double scale = max_horiz / t_horiz;
      thrust_w.X(thrust_w.X() * scale);
      thrust_w.Y(thrust_w.Y() * scale);
    }
  }

  const double thrust_total = thrust_w.Length();

  // Desired roll/pitch from thrust direction (yaw held at 0).
  const double roll_des =
      std::atan2(thrust_w.Y(), std::hypot(thrust_w.X(), thrust_w.Z()));
  const double pitch_des = std::atan2(-thrust_w.X(), thrust_w.Z());

  const auto euler = pose.Rot().Euler();
  const double roll = euler.X();
  const double pitch = euler.Y();
  const double yaw = euler.Z();

  const double p = omega_body.X();
  const double q = omega_body.Y();
  const double r_rate = omega_body.Z();

  const double tau_roll = att_kp_ * (roll_des - roll) - att_kd_ * p;
  const double tau_pitch = att_kp_ * (pitch_des - pitch) - att_kd_ * q;
  const double tau_yaw = -yaw_kp_ * yaw - yaw_kd_ * r_rate;

  std::array<double, 4> thrust_motors{};
  if (!AllocateMotors(thrust_total, tau_roll, tau_pitch, tau_yaw, thrust_motors)) {
    const double hover_each = mass_ * gravity_z_ / static_cast<double>(std::max<size_t>(rotors_.size(), 1));
    for (size_t i = 0; i < rotors_.size(); ++i) thrust_motors[i] = hover_each;
  }

  for (size_t i = 0; i < rotors_.size(); ++i) {
    const double T_cmd = thrust_motors[i];
    double omega_target = (k_f_ > 0.0) ? std::sqrt(std::max(0.0, T_cmd / k_f_)) : 0.0;
    omega_target = Clamp(omega_target, min_omega_, max_omega_);
    rotors_[i].omega_cmd = omega_target;
  }

  const double tau_motor = std::max(motor_tau_, 1.0e-3);
  for (auto& rotor : rotors_) {
    const double alpha = std::min(1.0, dt / tau_motor);
    rotor.omega += alpha * (rotor.omega_cmd - rotor.omega);
    rotor.omega = Clamp(rotor.omega, min_omega_, max_omega_);
  }

  ApplyRotorForces(link);

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      const double roll_deg = roll * 180.0 / M_PI;
      const double pitch_deg = pitch * 180.0 / M_PI;
      gzmsg << "[QuadrotorDynamicsPlugin] err=" << err.X() << "," << err.Y() << "," << err.Z()
            << " thrust_total=" << thrust_total << " N roll=" << roll_deg << "deg pitch=" << pitch_deg
            << "deg omega0=" << rotors_[0].omega << " rad/s\n";
    }
  }
}

GZ_REGISTER_MODEL_PLUGIN(QuadrotorDynamicsPlugin)

}  // namespace gazebo

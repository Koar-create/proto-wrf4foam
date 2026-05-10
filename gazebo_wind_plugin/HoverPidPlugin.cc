#include "HoverPidPlugin.hh"

#include <cmath>

#include <ignition/math/Vector3.hh>

namespace gazebo {

void HoverPidPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf) {
  model_ = model;
  if (!model_ || !sdf) {
    gzerr << "[HoverPidPlugin] model or SDF is null\n";
    return;
  }

  link_name_ = sdf->Get<std::string>("link_name", "base_link").first;

  target_.X(sdf->Get<double>("target_x", 1420.0).first);
  target_.Y(sdf->Get<double>("target_y", -880.0).first);
  target_.Z(sdf->Get<double>("target_z", 50.0).first);

  kp_ = sdf->Get<double>("kp", 8.0).first;
  ki_ = sdf->Get<double>("ki", 0.1).first;
  kd_ = sdf->Get<double>("kd", 4.0).first;

  kp_z_ = sdf->Get<double>("kp_z", kp_).first;
  ki_z_ = sdf->Get<double>("ki_z", ki_).first;
  kd_z_ = sdf->Get<double>("kd_z", kd_).first;

  enable_xy_ = sdf->Get<bool>("enable_xy", true).first;

  enable_attitude_recovery_ = sdf->Get<bool>("enable_attitude_recovery", false).first;
  attitude_kp_ = sdf->Get<double>("attitude_kp", 15.0).first;

  gravity_compensation_ = sdf->Get<bool>("gravity_compensation", false).first;
  if (gravity_compensation_) {
    auto link = model_->GetLink(link_name_);
    auto world_for_g = model_->GetWorld();
    if (link && world_for_g) {
      const double mass = link->GetInertial() ? link->GetInertial()->Mass() : 0.0;
      const double g_mag = world_for_g->Gravity().Length();
      gravity_ff_force_z_ = mass * g_mag;
      gzmsg << "[HoverPidPlugin] gravity_compensation: mass=" << mass << " kg, |g|=" << g_mag
            << " m/s^2, ff_thrust=" << gravity_ff_force_z_ << " N\n";
    } else {
      gzwarn << "[HoverPidPlugin] gravity_compensation requested but link/world unavailable; skipping FF.\n";
    }
  }

  drift_after_seconds_ = sdf->Get<double>("drift_after_seconds", -1.0).first;

  disable_topic_ = sdf->Get<std::string>("disable_topic", std::string()).first;
  crash_zero_thrust_ = sdf->Get<bool>("crash_zero_thrust", false).first;

  log_every_n_ = sdf->Get<int>("log_every_n", 250).first;

  last_time_ = model_->GetWorld() ? model_->GetWorld()->SimTime() : common::Time::Zero;

  gzmsg << "[HoverPidPlugin] target=(" << target_.X() << "," << target_.Y() << "," << target_.Z() << ") Kp_xy=" << kp_
        << " Ki_xy=" << ki_ << " Kd_xy=" << kd_ << " Kp_z=" << kp_z_ << " Ki_z=" << ki_z_ << " Kd_z=" << kd_z_
        << " link=" << link_name_ << " enable_xy=" << (enable_xy_ ? 1 : 0)
        << " drift_after_seconds=" << drift_after_seconds_ << " enable_attitude_recovery=" << (enable_attitude_recovery_ ? 1 : 0)
        << " disable_topic=" << disable_topic_ << " crash_zero_thrust=" << (crash_zero_thrust_ ? 1 : 0)
        << "\n";

  if (!disable_topic_.empty()) {
    auto world = model_->GetWorld();
    node_.reset(new transport::Node());
    node_->Init(world ? world->Name() : "");
    disable_sub_ = node_->Subscribe(disable_topic_, &HoverPidPlugin::OnDisableMsg, this);
    gzmsg << "[HoverPidPlugin] subscribed disable_topic=" << disable_topic_ << "\n";
  }

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&HoverPidPlugin::OnUpdate, this));
}

void HoverPidPlugin::OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg) {
  if (disabled_.exchange(true)) return;
  integral_.Set(0.0, 0.0, 0.0);
  prev_err_.Set(0.0, 0.0, 0.0);
  first_step_ = true;
  gzmsg << "[HoverPidPlugin] disabled by '" << (msg ? msg->data() : std::string("?"))
        << "' on " << disable_topic_ << " (crash_zero_thrust="
        << (crash_zero_thrust_ ? 1 : 0) << ")\n";
}

void HoverPidPlugin::OnUpdate() {
  if (!model_) return;

  auto link = model_->GetLink(link_name_);
  if (!link) return;

  auto world = model_->GetWorld();
  if (!world) return;

  if (disabled_.load() && crash_zero_thrust_) {
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
  const ignition::math::Vector3d err = target_ - pos;

  if (first_step_) {
    prev_err_ = err;
    first_step_ = false;
  }

  const ignition::math::Vector3d derr = (err - prev_err_) / dt;
  prev_err_ = err;

  integral_ += err * dt;
  // Simple anti-windup: clamp integral magnitude per axis (m·s)
  const double i_max = 50.0;
  integral_.X(std::max(-i_max, std::min(i_max, integral_.X())));
  integral_.Y(std::max(-i_max, std::min(i_max, integral_.Y())));
  integral_.Z(std::max(-i_max, std::min(i_max, integral_.Z())));

  double fx = kp_ * err.X() + ki_ * integral_.X() + kd_ * derr.X();
  double fy = kp_ * err.Y() + ki_ * integral_.Y() + kd_ * derr.Y();
  double fz = kp_z_ * err.Z() + ki_z_ * integral_.Z() + kd_z_ * derr.Z();
  if (gravity_compensation_) {
    fz += gravity_ff_force_z_;
  }

  bool xy_on = enable_xy_;
  if (drift_after_seconds_ >= 0.0) {
    const double sim_t = world->SimTime().Double();
    xy_on = sim_t < drift_after_seconds_;
  }

  if (!xy_on) {
    fx = 0.0;
    fy = 0.0;
    integral_.X(0.0);
    integral_.Y(0.0);
  }

  const ignition::math::Vector3d force(fx, fy, fz);

  link->AddForce(force);

  if (enable_attitude_recovery_) {
    const auto euler = pose.Rot().Euler();
    const double roll = euler.X();
    const double pitch = euler.Y();
    link->AddTorque(ignition::math::Vector3d(-attitude_kp_ * roll, -attitude_kp_ * pitch, 0.0));
  }

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      const auto euler = pose.Rot().Euler();
      const double roll_deg = euler.X() * 180.0 / M_PI;
      const double pitch_deg = euler.Y() * 180.0 / M_PI;
      gzmsg << "[HoverPidPlugin] hover_error=" << err.X() << "," << err.Y() << "," << err.Z() << " hover_force="
            << force.X() << "," << force.Y() << "," << force.Z() << " roll=" << roll_deg << "deg pitch=" << pitch_deg
            << "deg\n";
    }
  }
}

GZ_REGISTER_MODEL_PLUGIN(HoverPidPlugin)

}  // namespace gazebo

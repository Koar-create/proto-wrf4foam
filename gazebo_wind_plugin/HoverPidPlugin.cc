#include "HoverPidPlugin.hh"

#include <cmath>

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

  enable_xy_ = sdf->Get<bool>("enable_xy", true).first;

  log_every_n_ = sdf->Get<int>("log_every_n", 250).first;

  last_time_ = model_->GetWorld() ? model_->GetWorld()->SimTime() : common::Time::Zero;

  gzmsg << "[HoverPidPlugin] target=(" << target_.X() << "," << target_.Y() << "," << target_.Z() << ") Kp=" << kp_
        << " Ki=" << ki_ << " Kd=" << kd_ << " link=" << link_name_ << " enable_xy=" << (enable_xy_ ? 1 : 0) << "\n";

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&HoverPidPlugin::OnUpdate, this));
}

void HoverPidPlugin::OnUpdate() {
  if (!model_) return;

  auto link = model_->GetLink(link_name_);
  if (!link) return;

  auto world = model_->GetWorld();
  if (!world) return;

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
  const double fz = kp_ * err.Z() + ki_ * integral_.Z() + kd_ * derr.Z();

  if (!enable_xy_) {
    fx = 0.0;
    fy = 0.0;
    integral_.X(0.0);
    integral_.Y(0.0);
  }

  const ignition::math::Vector3d force(fx, fy, fz);

  link->AddForce(force);

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      gzmsg << "[HoverPidPlugin] hover_error=" << err.X() << "," << err.Y() << "," << err.Z() << " hover_force="
            << force.X() << "," << force.Y() << "," << force.Z() << "\n";
    }
  }
}

GZ_REGISTER_MODEL_PLUGIN(HoverPidPlugin)

}  // namespace gazebo

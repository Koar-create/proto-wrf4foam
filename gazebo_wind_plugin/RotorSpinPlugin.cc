#include "RotorSpinPlugin.hh"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace gazebo {

void RotorSpinPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf) {
  model_ = model;
  if (!model_ || !sdf) {
    gzerr << "[RotorSpinPlugin] model or SDF is null\n";
    return;
  }

  max_torque_ = sdf->Get<double>("max_torque", 0.5).first;
  spin_up_tau_ = sdf->Get<double>("spin_up_tau", 1.5).first;
  spin_down_tau_ = sdf->Get<double>("spin_down_tau", 1.0).first;
  log_every_n_ = sdf->Get<int>("log_every_n", 0).first;

  if (sdf->HasElement("rotor")) {
    sdf::ElementPtr el = sdf->GetElement("rotor");
    while (el) {
      Rotor r;
      r.name = el->Get<std::string>("joint", std::string()).first;
      r.rate = el->Get<double>("rate", 0.0).first;
      if (!r.name.empty()) {
        r.joint = model_->GetJoint(r.name);
        if (!r.joint) {
          gzerr << "[RotorSpinPlugin] joint not found: " << r.name << "\n";
        } else {
          rotors_.push_back(r);
        }
      }
      el = el->GetNextElement("rotor");
    }
  }

  if (rotors_.empty()) {
    gzwarn << "[RotorSpinPlugin] no <rotor> entries; plugin is a no-op\n";
    return;
  }

  disable_topic_ = sdf->Get<std::string>("disable_topic", std::string()).first;
  if (!disable_topic_.empty()) {
    auto world = model_->GetWorld();
    node_.reset(new transport::Node());
    node_->Init(world ? world->Name() : "");
    disable_sub_ = node_->Subscribe(disable_topic_, &RotorSpinPlugin::OnDisableMsg, this);
  }

  load_time_ = model_->GetWorld() ? model_->GetWorld()->SimTime() : common::Time::Zero;

  std::ostringstream summary;
  for (size_t i = 0; i < rotors_.size(); ++i) {
    summary << rotors_[i].name << "@" << rotors_[i].rate;
    if (i + 1 < rotors_.size()) summary << ", ";
  }
  gzmsg << "[RotorSpinPlugin] " << rotors_.size() << " rotor(s): " << summary.str()
        << " | max_torque=" << max_torque_ << " N*m, spin_up_tau=" << spin_up_tau_
        << " s, spin_down_tau=" << spin_down_tau_ << " s, disable_topic=" << disable_topic_ << "\n";

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&RotorSpinPlugin::OnUpdate, this));
}

void RotorSpinPlugin::OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg) {
  if (disabled_.exchange(true)) return;
  auto world = model_ ? model_->GetWorld() : nullptr;
  disable_time_ = world ? world->SimTime() : common::Time::Zero;
  gzmsg << "[RotorSpinPlugin] disabled by '" << (msg ? msg->data() : std::string("?")) << "' on "
        << disable_topic_ << " (spin_down_tau=" << spin_down_tau_ << " s)\n";
}

void RotorSpinPlugin::OnUpdate() {
  if (!model_) return;
  auto world = model_->GetWorld();
  if (!world) return;

  const common::Time now = world->SimTime();
  const double t_alive = (now - load_time_).Double();
  const double ramp_up = (spin_up_tau_ > 0.0) ? std::min(1.0, std::max(0.0, t_alive / spin_up_tau_)) : 1.0;

  double decay = 1.0;
  if (disabled_.load() && spin_down_tau_ > 0.0) {
    const double t_disable = (now - disable_time_).Double();
    decay = std::exp(-t_disable / spin_down_tau_);
    if (decay < 1.0e-3) decay = 0.0;
  }

  for (auto& r : rotors_) {
    if (!r.joint) continue;
    const double target = r.rate * ramp_up * decay;
    r.joint->SetParam("fmax", 0, max_torque_);
    r.joint->SetParam("vel", 0, target);
  }

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0 && !rotors_.empty()) {
      const auto& r0 = rotors_.front();
      gzmsg << "[RotorSpinPlugin] ramp_up=" << ramp_up << " decay=" << decay
            << " target_rate(" << r0.name << ")=" << (r0.rate * ramp_up * decay) << " rad/s\n";
    }
  }
}

GZ_REGISTER_MODEL_PLUGIN(RotorSpinPlugin)

}  // namespace gazebo

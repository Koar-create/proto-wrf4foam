#pragma once

#include <atomic>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo {

/// Visual / kinematic rotor spin driver. Each `<rotor>` block names a revolute
/// joint and the steady-state angular velocity (rad/s, signed: + = CCW, - = CW)
/// the plugin will drive it at via the ODE joint motor (SetParam vel/fmax).
///
/// The plugin smoothly ramps target velocity from 0 to `rate` over
/// `<spin_up_tau>` seconds at load time, and after the optional `<disable_topic>`
/// latches it exponentially decays to 0 with time constant `<spin_down_tau>` so
/// the rotors visibly spool down on crash. No collision is required on rotor
/// links; `max_torque` (default small) is the joint motor's f-max so the
/// reaction torque on the parent body stays bounded.
class RotorSpinPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

private:
  struct Rotor {
    physics::JointPtr joint;
    std::string name;
    double rate{0.0};  // signed steady-state angular velocity (rad/s)
  };

  void OnUpdate();
  void OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg);

  physics::ModelPtr model_;
  event::ConnectionPtr update_conn_;

  std::vector<Rotor> rotors_;
  double max_torque_{0.5};
  double spin_up_tau_{1.5};
  double spin_down_tau_{1.0};

  common::Time load_time_;
  common::Time disable_time_;
  std::atomic<bool> disabled_{false};

  std::string disable_topic_;
  transport::NodePtr node_;
  transport::SubscriberPtr disable_sub_;

  int log_every_n_{0};
  std::uint64_t step_i_{0};
};

}  // namespace gazebo

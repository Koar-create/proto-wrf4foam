#pragma once

#include <cstdint>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

#include <boost/shared_ptr.hpp>

#include <atomic>
#include <string>

namespace gazebo {

/// World-frame translational PID on base_link position; applies force each step.
class HoverPidPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

private:
  void OnUpdate();

  physics::ModelPtr model_;
  event::ConnectionPtr update_conn_;

  std::string link_name_{"base_link"};

  ignition::math::Vector3d target_{1420.0, -880.0, 50.0};
  double kp_{8.0};
  double ki_{0.1};
  double kd_{4.0};
  /// Z-axis PID gains; if omitted in SDF, default to kp_/ki_/kd_ at load time.
  double kp_z_{8.0};
  double ki_z_{0.1};
  double kd_z_{4.0};
  bool enable_xy_{true};

  ignition::math::Vector3d integral_{0, 0, 0};
  ignition::math::Vector3d prev_err_{0, 0, 0};
  bool first_step_{true};

  int log_every_n_{250};
  std::uint64_t step_i_{0};
  common::Time last_time_;

  bool enable_attitude_recovery_{false};
  double attitude_kp_{15.0};

  /// If >= 0: for sim_time < value force XY PID on, then force XY off. If < 0: use enable_xy from SDF only.
  double drift_after_seconds_{-1.0};

  /// Optional Gazebo transport topic. If non-empty, subscribe to GzString
  /// messages; first message latches `disabled_` so all force/torque outputs
  /// become zero. Used by ContactWatcherPlugin on crash.
  std::string disable_topic_;
  bool crash_zero_thrust_{false};
  std::atomic<bool> disabled_{false};

  transport::NodePtr node_;
  transport::SubscriberPtr disable_sub_;

  void OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg);
};

}  // namespace gazebo

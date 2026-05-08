#pragma once

#include <cstdint>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

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
};

}  // namespace gazebo

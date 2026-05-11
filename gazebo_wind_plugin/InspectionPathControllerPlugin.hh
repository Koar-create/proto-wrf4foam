#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

#include <ignition/math/Vector3.hh>

namespace gazebo {

/// World-frame waypoint follower with optional 2D building barrier (demo CBF-style).
/// Reuses the HoverPidPlugin-style translational PID + optional attitude damping.
class InspectionPathControllerPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

private:
  void OnUpdate();
  void OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg);

  static void closestPointOnBox2D(double px, double py, double min_x, double max_x, double min_y, double max_y,
                                  double* cx, double* cy);
  static void surfaceDistanceAndNormal2D(double px, double py, double min_x, double max_x, double min_y,
                                           double max_y, double* dist, double* nx, double* ny, bool* inside);

  physics::ModelPtr model_;
  event::ConnectionPtr update_conn_;

  std::string link_name_{"base_link"};

  std::vector<ignition::math::Vector3d> waypoints_;
  std::size_t wp_index_{0};
  double arrival_radius_{3.0};
  bool loop_{true};

  double kp_{8.0};
  double ki_{0.1};
  double kd_{4.0};
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

  bool gravity_compensation_{false};
  double gravity_ff_force_z_{0.0};

  double drift_after_seconds_{-1.0};

  /// If >=0 and current waypoint index >= this value, multiply XY PID gains by soften_xy_scale_.
  int soften_xy_after_wp_{-1};
  double soften_xy_scale_{1.0};

  bool enable_barrier_{false};
  double building_min_x_{0.0};
  double building_max_x_{0.0};
  double building_min_y_{0.0};
  double building_max_y_{0.0};
  double safety_margin_{6.0};
  double barrier_gain_{100.0};
  double barrier_vel_gain_{25.0};

  std::string disable_topic_;
  bool crash_zero_thrust_{false};
  std::atomic<bool> disabled_{false};

  transport::NodePtr node_;
  transport::SubscriberPtr disable_sub_;

  void OnDisableMsgImpl(const std::string& reason);
};

}  // namespace gazebo

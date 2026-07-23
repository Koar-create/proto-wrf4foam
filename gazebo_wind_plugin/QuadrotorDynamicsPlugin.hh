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

namespace gazebo {

/// Demo quadrotor dynamics: position PID -> desired thrust vector in world frame,
/// attitude PD -> roll/pitch/yaw torques, X-configuration motor mixing, first-order
/// motor speed response, per-rotor thrust at body +Z and reaction yaw torque.
/// Replaces world-frame HoverPidPlugin for physically consistent wind interaction.
class QuadrotorDynamicsPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

private:
  struct Rotor {
    physics::JointPtr joint;
    std::string joint_name;
    ignition::math::Vector3d arm_body{0, 0, 0};  // rotor position in base_link frame (m)
    int spin_dir{1};                               // +1 CCW, -1 CW (yaw reaction sign)
    double omega{0.0};                             // current motor speed (rad/s)
    double omega_cmd{0.0};
  };

  void OnUpdate();
  void OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg);
  bool AllocateMotors(double thrust_total, double tau_roll, double tau_pitch, double tau_yaw,
                      std::array<double, 4>& thrust_out) const;
  void ApplyRotorForces(physics::LinkPtr link);

  physics::ModelPtr model_;
  event::ConnectionPtr update_conn_;

  std::string link_name_{"base_link"};
  std::vector<Rotor> rotors_;

  ignition::math::Vector3d target_{-280.0, -400.0, 50.0};

  double mass_{1.5};
  double gravity_z_{9.81};

  // Translational PID (world-frame error -> desired acceleration)
  double kp_{6.0};
  double ki_{0.08};
  double kd_{3.5};
  double kp_z_{7.0};
  double ki_z_{0.08};
  double kd_z_{4.0};
  bool enable_xy_{true};

  ignition::math::Vector3d integral_{0, 0, 0};
  ignition::math::Vector3d prev_err_{0, 0, 0};
  bool first_step_{true};

  // Attitude PD (desired roll/pitch from thrust tilt; yaw held near zero)
  double att_kp_{8.0};
  double att_kd_{1.2};
  double yaw_kp_{2.0};
  double yaw_kd_{0.4};
  double max_tilt_rad_{0.55};  // ~31 deg max commanded tilt

  // Motor / thrust model: T = k_f * omega^2, tau_z_i = spin_dir * k_m * omega^2
  double k_f_{5.5e-4};
  double k_m_{8.8e-6};
  double motor_tau_{0.08};
  double max_omega_{520.0};
  double min_omega_{0.0};
  double max_thrust_per_motor_{12.0};
  double joint_motor_fmax_{0.5};

  int log_every_n_{250};
  std::uint64_t step_i_{0};
  common::Time last_time_;

  std::string disable_topic_;
  bool crash_zero_thrust_{true};
  std::atomic<bool> disabled_{false};

  transport::NodePtr node_;
  transport::SubscriberPtr disable_sub_;
};

}  // namespace gazebo

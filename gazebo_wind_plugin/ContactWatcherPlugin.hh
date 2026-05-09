#pragma once

#include <cstdint>
#include <string>

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo {

/// SensorPlugin attached to a contact sensor on the drone collision body.
/// Logs contact summaries and, when the peak normal force exceeds
/// crash_threshold_n once, publishes a one-shot disable message on a Gazebo
/// transport topic so HoverPidPlugin can zero its outputs (crash → free fall).
class ContactWatcherPlugin : public SensorPlugin {
public:
  ContactWatcherPlugin() = default;
  ~ContactWatcherPlugin() override = default;

  void Load(sensors::SensorPtr sensor, sdf::ElementPtr sdf) override;

private:
  void OnContact();

  sensors::ContactSensorPtr contact_sensor_;
  event::ConnectionPtr update_conn_;

  transport::NodePtr node_;
  transport::PublisherPtr disable_pub_;

  std::string disable_topic_{"~/hover_pid/disable"};
  double crash_threshold_n_{5.0};
  int log_every_n_{50};
  std::uint64_t step_i_{0};
  bool latched_crash_{false};
};

}  // namespace gazebo

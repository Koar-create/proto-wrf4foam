#include "ContactWatcherPlugin.hh"

#include <algorithm>
#include <cmath>
#include <string>

#include <gazebo/msgs/msgs.hh>
#include <gazebo/sensors/ContactSensor.hh>

namespace gazebo {

void ContactWatcherPlugin::Load(sensors::SensorPtr sensor, sdf::ElementPtr sdf) {
  contact_sensor_ = std::dynamic_pointer_cast<sensors::ContactSensor>(sensor);
  if (!contact_sensor_) {
    gzerr << "[ContactWatcherPlugin] sensor is not a ContactSensor\n";
    return;
  }

  if (sdf) {
    crash_threshold_n_ = sdf->Get<double>("crash_threshold_n", crash_threshold_n_).first;
    log_every_n_ = sdf->Get<int>("log_every_n", log_every_n_).first;
    disable_topic_ = sdf->Get<std::string>("publish_disable_topic", disable_topic_).first;
  }

  contact_sensor_->SetActive(true);

  node_.reset(new transport::Node());
  node_->Init();
  if (!disable_topic_.empty()) {
    disable_pub_ = node_->Advertise<msgs::GzString>(disable_topic_);
  }

  gzmsg << "[ContactWatcherPlugin] sensor=" << contact_sensor_->Name()
        << " parent=" << contact_sensor_->ParentName()
        << " crash_threshold_n=" << crash_threshold_n_
        << " disable_topic=" << disable_topic_ << "\n";

  update_conn_ = contact_sensor_->ConnectUpdated(std::bind(&ContactWatcherPlugin::OnContact, this));
}

void ContactWatcherPlugin::OnContact() {
  if (!contact_sensor_) return;

  msgs::Contacts contacts = contact_sensor_->Contacts();
  const int n = contacts.contact_size();
  if (n <= 0) return;

  double peak = 0.0;
  std::string a, b;
  for (int i = 0; i < n; ++i) {
    const auto& c = contacts.contact(i);
    for (int j = 0; j < c.wrench_size(); ++j) {
      const auto& w = c.wrench(j).body_1_wrench();
      const double mag = std::sqrt(
          w.force().x() * w.force().x() + w.force().y() * w.force().y() + w.force().z() * w.force().z());
      if (mag > peak) {
        peak = mag;
        a = c.collision1();
        b = c.collision2();
      }
    }
  }

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      gzmsg << "[ContactWatcher] count=" << n << " peak_force=" << peak
            << "N a=" << a << " b=" << b << "\n";
    }
  }

  if (!latched_crash_ && peak >= crash_threshold_n_) {
    latched_crash_ = true;
    gzmsg << "[ContactWatcher] CRASH peak_force=" << peak
          << "N (>=" << crash_threshold_n_ << ") a=" << a << " b=" << b << "\n";
    if (disable_pub_) {
      msgs::GzString s;
      s.set_data("crash");
      disable_pub_->Publish(s);
      gzmsg << "[ContactWatcher] disable msg published on " << disable_topic_ << "\n";
    }
  }
}

GZ_REGISTER_SENSOR_PLUGIN(ContactWatcherPlugin)

}  // namespace gazebo

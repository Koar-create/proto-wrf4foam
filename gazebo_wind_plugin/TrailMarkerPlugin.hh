#pragma once

#include <cstdint>
#include <deque>
#include <string>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo {

/// Drop visual-only spherical markers along a link trajectory.
///
/// Implementation uses Gazebo Classic transport:
/// - publishes to ~/factory to spawn a marker model
/// - publishes to ~/request with request=entity_delete to delete old markers
class TrailMarkerPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

private:
  void OnUpdate();
  std::string MakeMarkerSdf(const std::string& model_name, const ignition::math::Pose3d& pose) const;
  void SpawnMarker(const ignition::math::Pose3d& pose);
  void DeleteMarker(const std::string& name);

  physics::ModelPtr model_;
  event::ConnectionPtr update_conn_;

  std::string link_name_{"base_link"};
  double sample_period_s_{0.75};
  double marker_radius_{0.8};
  std::size_t max_points_{60};
  ignition::math::Color color_{1.0f, 0.3f, 0.1f, 0.65f};

  std::string name_prefix_{"trail_marker_"};
  std::uint64_t seq_{0};
  common::Time last_sample_time_;

  std::deque<std::string> alive_markers_;

  transport::NodePtr node_;
  transport::PublisherPtr factory_pub_;
  transport::PublisherPtr request_pub_;
};

}  // namespace gazebo


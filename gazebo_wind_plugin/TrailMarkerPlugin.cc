#include "TrailMarkerPlugin.hh"

#include <algorithm>
#include <sstream>

#include <gazebo/msgs/msgs.hh>

namespace gazebo {

void TrailMarkerPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf) {
  model_ = model;
  if (!model_ || !sdf) {
    gzerr << "[TrailMarkerPlugin] model or SDF is null\n";
    return;
  }

  link_name_ = sdf->Get<std::string>("link_name", link_name_).first;
  sample_period_s_ = sdf->Get<double>("sample_period", sample_period_s_).first;
  marker_radius_ = sdf->Get<double>("marker_radius", marker_radius_).first;
  max_points_ = static_cast<std::size_t>(std::max(0, sdf->Get<int>("max_points", static_cast<int>(max_points_)).first));
  name_prefix_ = sdf->Get<std::string>("name_prefix", name_prefix_).first;

  const double r = sdf->Get<double>("color_r", color_.R()).first;
  const double g = sdf->Get<double>("color_g", color_.G()).first;
  const double b = sdf->Get<double>("color_b", color_.B()).first;
  const double a = sdf->Get<double>("color_a", color_.A()).first;
  color_ = ignition::math::Color(static_cast<float>(r), static_cast<float>(g), static_cast<float>(b),
                                 static_cast<float>(a));

  auto world = model_->GetWorld();
  node_.reset(new transport::Node());
  node_->Init(world ? world->Name() : "");
  factory_pub_ = node_->Advertise<msgs::Factory>("~/factory");
  request_pub_ = node_->Advertise<msgs::Request>("~/request");

  last_sample_time_ = world ? world->SimTime() : common::Time::Zero;

  gzmsg << "[TrailMarkerPlugin] link=" << link_name_ << " sample_period=" << sample_period_s_
        << " marker_radius=" << marker_radius_ << " max_points=" << max_points_ << "\n";

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&TrailMarkerPlugin::OnUpdate, this));
}

std::string TrailMarkerPlugin::MakeMarkerSdf(const std::string& model_name, const ignition::math::Pose3d& pose) const {
  std::ostringstream ss;
  ss << "<?xml version='1.0'?>\n";
  ss << "<sdf version='1.6'>\n";
  ss << "  <model name='" << model_name << "'>\n";
  ss << "    <static>true</static>\n";
  ss << "    <pose>" << pose.Pos().X() << " " << pose.Pos().Y() << " " << pose.Pos().Z() << " 0 0 0</pose>\n";
  ss << "    <link name='link'>\n";
  ss << "      <gravity>false</gravity>\n";
  ss << "      <visual name='v'>\n";
  ss << "        <geometry><sphere><radius>" << marker_radius_ << "</radius></sphere></geometry>\n";
  ss << "        <material>\n";
  ss << "          <ambient>" << color_.R() << " " << color_.G() << " " << color_.B() << " " << color_.A()
     << "</ambient>\n";
  ss << "          <diffuse>" << color_.R() << " " << color_.G() << " " << color_.B() << " " << color_.A()
     << "</diffuse>\n";
  ss << "        </material>\n";
  ss << "        <cast_shadows>false</cast_shadows>\n";
  ss << "      </visual>\n";
  ss << "    </link>\n";
  ss << "  </model>\n";
  ss << "</sdf>\n";
  return ss.str();
}

void TrailMarkerPlugin::SpawnMarker(const ignition::math::Pose3d& pose) {
  if (!factory_pub_) return;

  const std::string name = name_prefix_ + std::to_string(model_ ? model_->GetId() : 0) + "_" + std::to_string(seq_++);
  msgs::Factory fac;
  fac.set_sdf(MakeMarkerSdf(name, pose));
  factory_pub_->Publish(fac);

  alive_markers_.push_back(name);
  while (alive_markers_.size() > max_points_ && !alive_markers_.empty()) {
    DeleteMarker(alive_markers_.front());
    alive_markers_.pop_front();
  }
}

void TrailMarkerPlugin::DeleteMarker(const std::string& name) {
  if (!request_pub_) return;
  msgs::Request req;
  req.set_id(static_cast<int>(seq_));  // best-effort unique-ish
  req.set_request("entity_delete");
  req.set_data(name);
  request_pub_->Publish(req);
}

void TrailMarkerPlugin::OnUpdate() {
  if (!model_) return;
  auto world = model_->GetWorld();
  if (!world) return;

  auto link = model_->GetLink(link_name_);
  if (!link) return;

  const common::Time now = world->SimTime();
  const double dt = (now - last_sample_time_).Double();
  if (dt < sample_period_s_) return;

  last_sample_time_ = now;
  SpawnMarker(link->WorldPose());
}

GZ_REGISTER_MODEL_PLUGIN(TrailMarkerPlugin)

}  // namespace gazebo


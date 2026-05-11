#include "InspectionPathControllerPlugin.hh"

#include <algorithm>
#include <cmath>

namespace gazebo {

void InspectionPathControllerPlugin::closestPointOnBox2D(double px, double py, double min_x, double max_x,
                                                       double min_y, double max_y, double* cx, double* cy) {
  const double x_clamped = std::max(min_x, std::min(max_x, px));
  const double y_clamped = std::max(min_y, std::min(max_y, py));
  *cx = x_clamped;
  *cy = y_clamped;
}

void InspectionPathControllerPlugin::surfaceDistanceAndNormal2D(double px, double py, double min_x, double max_x,
                                                                double min_y, double max_y, double* dist, double* nx,
                                                                double* ny, bool* inside) {
  const bool in_x = (px >= min_x && px <= max_x);
  const bool in_y = (py >= min_y && py <= max_y);
  *inside = in_x && in_y;

  double cx = 0.0;
  double cy = 0.0;
  closestPointOnBox2D(px, py, min_x, max_x, min_y, max_y, &cx, &cy);

  const double ex = px - cx;
  const double ey = py - cy;
  const double len = std::hypot(ex, ey);

  if (len > 1.0e-9) {
    *nx = ex / len;
    *ny = ey / len;
    *dist = len;
    return;
  }

  // On an edge or corner of the box (or deep inside with degenerate closest point): pick smallest exit axis.
  const double d_left = px - min_x;
  const double d_right = max_x - px;
  const double d_down = py - min_y;
  const double d_up = max_y - py;
  const double m = std::min(std::min(d_left, d_right), std::min(d_down, d_up));
  if (m == d_left) {
    *nx = -1.0;
    *ny = 0.0;
  } else if (m == d_right) {
    *nx = 1.0;
    *ny = 0.0;
  } else if (m == d_down) {
    *nx = 0.0;
    *ny = -1.0;
  } else {
    *nx = 0.0;
    *ny = 1.0;
  }
  *dist = 0.0;
}

void InspectionPathControllerPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf) {
  model_ = model;
  if (!model_ || !sdf) {
    gzerr << "[InspectionPathControllerPlugin] model or SDF is null\n";
    return;
  }

  link_name_ = sdf->Get<std::string>("link_name", "base_link").first;

  if (sdf->HasElement("waypoint")) {
    sdf::ElementPtr el = sdf->GetElement("waypoint");
    while (el) {
      const double wx = el->Get<double>("x", 0.0).first;
      const double wy = el->Get<double>("y", 0.0).first;
      const double wz = el->Get<double>("z", 80.0).first;
      waypoints_.emplace_back(wx, wy, wz);
      el = el->GetNextElement("waypoint");
    }
  }

  if (waypoints_.empty()) {
    gzerr << "[InspectionPathControllerPlugin] no <waypoint> elements; plugin disabled\n";
    return;
  }

  arrival_radius_ = sdf->Get<double>("arrival_radius", 3.0).first;
  loop_ = sdf->Get<bool>("loop", true).first;

  kp_ = sdf->Get<double>("kp", 8.0).first;
  ki_ = sdf->Get<double>("ki", 0.1).first;
  kd_ = sdf->Get<double>("kd", 4.0).first;

  kp_z_ = sdf->Get<double>("kp_z", kp_).first;
  ki_z_ = sdf->Get<double>("ki_z", ki_).first;
  kd_z_ = sdf->Get<double>("kd_z", kd_).first;

  enable_xy_ = sdf->Get<bool>("enable_xy", true).first;

  enable_attitude_recovery_ = sdf->Get<bool>("enable_attitude_recovery", false).first;
  attitude_kp_ = sdf->Get<double>("attitude_kp", 15.0).first;

  gravity_compensation_ = sdf->Get<bool>("gravity_compensation", false).first;
  if (gravity_compensation_) {
    auto link = model_->GetLink(link_name_);
    auto world_for_g = model_->GetWorld();
    if (link && world_for_g) {
      const double mass = link->GetInertial() ? link->GetInertial()->Mass() : 0.0;
      const double g_mag = world_for_g->Gravity().Length();
      gravity_ff_force_z_ = mass * g_mag;
      gzmsg << "[InspectionPathControllerPlugin] gravity_compensation: mass=" << mass << " kg, |g|=" << g_mag
            << " m/s^2, ff_thrust=" << gravity_ff_force_z_ << " N\n";
    } else {
      gzwarn << "[InspectionPathControllerPlugin] gravity_compensation requested but link/world unavailable\n";
    }
  }

  drift_after_seconds_ = sdf->Get<double>("drift_after_seconds", -1.0).first;

  soften_xy_after_wp_ = sdf->Get<int>("soften_xy_after_waypoint_index", -1).first;
  soften_xy_scale_ = sdf->Get<double>("soften_xy_gain_scale", 1.0).first;

  enable_barrier_ = sdf->Get<bool>("enable_barrier_avoidance", false).first;
  building_min_x_ = sdf->Get<double>("building_min_x", 1436.0).first;
  building_max_x_ = sdf->Get<double>("building_max_x", 1506.0).first;
  building_min_y_ = sdf->Get<double>("building_min_y", 1314.0).first;
  building_max_y_ = sdf->Get<double>("building_max_y", 1366.0).first;
  safety_margin_ = sdf->Get<double>("safety_margin", 6.0).first;
  barrier_gain_ = sdf->Get<double>("barrier_gain", 100.0).first;
  barrier_vel_gain_ = sdf->Get<double>("barrier_vel_gain", 25.0).first;

  disable_topic_ = sdf->Get<std::string>("disable_topic", std::string()).first;
  crash_zero_thrust_ = sdf->Get<bool>("crash_zero_thrust", false).first;

  log_every_n_ = sdf->Get<int>("log_every_n", 250).first;

  last_time_ = model_->GetWorld() ? model_->GetWorld()->SimTime() : common::Time::Zero;

  gzmsg << "[InspectionPathControllerPlugin] waypoints=" << waypoints_.size() << " arrival_r=" << arrival_radius_
        << " loop=" << (loop_ ? 1 : 0) << " enable_xy=" << (enable_xy_ ? 1 : 0)
        << " barrier=" << (enable_barrier_ ? 1 : 0) << " soften_after_wp=" << soften_xy_after_wp_
        << " soften_scale=" << soften_xy_scale_ << " disable_topic=" << disable_topic_
        << " crash_zero_thrust=" << (crash_zero_thrust_ ? 1 : 0) << "\n";

  if (!disable_topic_.empty()) {
    auto world = model_->GetWorld();
    node_.reset(new transport::Node());
    node_->Init(world ? world->Name() : "");
    disable_sub_ = node_->Subscribe(disable_topic_, &InspectionPathControllerPlugin::OnDisableMsg, this);
    gzmsg << "[InspectionPathControllerPlugin] subscribed disable_topic=" << disable_topic_ << "\n";
  }

  update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&InspectionPathControllerPlugin::OnUpdate, this));
}

void InspectionPathControllerPlugin::OnDisableMsg(const boost::shared_ptr<const msgs::GzString>& msg) {
  OnDisableMsgImpl(msg ? msg->data() : std::string("?"));
}

void InspectionPathControllerPlugin::OnDisableMsgImpl(const std::string& reason) {
  if (disabled_.exchange(true)) return;
  integral_.Set(0.0, 0.0, 0.0);
  prev_err_.Set(0.0, 0.0, 0.0);
  first_step_ = true;
  gzmsg << "[InspectionPathControllerPlugin] disabled by '" << reason << "' on " << disable_topic_
        << " (crash_zero_thrust=" << (crash_zero_thrust_ ? 1 : 0) << ")\n";
}

void InspectionPathControllerPlugin::OnUpdate() {
  if (!model_ || waypoints_.empty()) return;

  auto link = model_->GetLink(link_name_);
  if (!link) return;

  auto world = model_->GetWorld();
  if (!world) return;

  if (disabled_.load() && crash_zero_thrust_) {
    return;
  }

  const common::Time now = world->SimTime();
  double dt = (now - last_time_).Double();
  if (dt <= 0.0 || dt > 1.0) {
    auto pe = world->Physics();
    dt = pe ? pe->GetMaxStepSize() : 0.004;
  }
  last_time_ = now;

  const auto pose = link->WorldPose();
  const ignition::math::Vector3d pos = pose.Pos();

  ignition::math::Vector3d target = waypoints_[wp_index_];
  ignition::math::Vector3d err = target - pos;

  if (err.Length() <= arrival_radius_) {
    ++wp_index_;
    if (wp_index_ >= waypoints_.size()) {
      if (loop_) {
        wp_index_ = 0;
        gzmsg << "[InspectionPathControllerPlugin] loop: restart waypoints\n";
      } else {
        wp_index_ = waypoints_.size() - 1;
      }
    }
    target = waypoints_[wp_index_];
    err = target - pos;
    first_step_ = true;
    prev_err_ = err;
  }

  if (first_step_) {
    prev_err_ = err;
    first_step_ = false;
  }

  const ignition::math::Vector3d derr = (err - prev_err_) / dt;
  prev_err_ = err;

  integral_ += err * dt;
  const double i_max = 50.0;
  integral_.X(std::max(-i_max, std::min(i_max, integral_.X())));
  integral_.Y(std::max(-i_max, std::min(i_max, integral_.Y())));
  integral_.Z(std::max(-i_max, std::min(i_max, integral_.Z())));

  double kp_xy = kp_;
  double ki_xy = ki_;
  double kd_xy = kd_;
  if (soften_xy_after_wp_ >= 0 && static_cast<int>(wp_index_) >= soften_xy_after_wp_) {
    kp_xy *= soften_xy_scale_;
    ki_xy *= soften_xy_scale_;
    kd_xy *= soften_xy_scale_;
  }

  double fx = kp_xy * err.X() + ki_xy * integral_.X() + kd_xy * derr.X();
  double fy = kp_xy * err.Y() + ki_xy * integral_.Y() + kd_xy * derr.Y();
  double fz = kp_z_ * err.Z() + ki_z_ * integral_.Z() + kd_z_ * derr.Z();
  if (gravity_compensation_) {
    fz += gravity_ff_force_z_;
  }

  bool xy_on = enable_xy_;
  if (drift_after_seconds_ >= 0.0) {
    const double sim_t = world->SimTime().Double();
    xy_on = sim_t < drift_after_seconds_;
  }

  if (!xy_on) {
    fx = 0.0;
    fy = 0.0;
    integral_.X(0.0);
    integral_.Y(0.0);
  }

  if (enable_barrier_ && xy_on) {
    double dist = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    bool inside = false;
    surfaceDistanceAndNormal2D(pos.X(), pos.Y(), building_min_x_, building_max_x_, building_min_y_, building_max_y_,
                               &dist, &nx, &ny, &inside);

    const auto vel = link->WorldLinearVel();
    const double vxy_x = vel.X();
    const double vxy_y = vel.Y();

    auto push = [&](double deficit) {
      if (deficit <= 0.0) return;
      fx += barrier_gain_ * deficit * nx;
      fy += barrier_gain_ * deficit * ny;
      const double vn = vxy_x * nx + vxy_y * ny;
      if (vn < 0.0) {
        fx += barrier_vel_gain_ * (-vn) * nx;
        fy += barrier_vel_gain_ * (-vn) * ny;
      }
    };

    if (inside) {
      push(safety_margin_ + 1.0);
    } else if (dist < safety_margin_) {
      push(safety_margin_ - dist);
    }
  }

  const ignition::math::Vector3d force(fx, fy, fz);
  link->AddForce(force);

  if (enable_attitude_recovery_) {
    const auto euler = pose.Rot().Euler();
    const double roll = euler.X();
    const double pitch = euler.Y();
    link->AddTorque(ignition::math::Vector3d(-attitude_kp_ * roll, -attitude_kp_ * pitch, 0.0));
  }

  if (log_every_n_ > 0) {
    ++step_i_;
    if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
      const auto euler = pose.Rot().Euler();
      const double roll_deg = euler.X() * 180.0 / M_PI;
      const double pitch_deg = euler.Y() * 180.0 / M_PI;
      gzmsg << "[InspectionPathControllerPlugin] wp=" << wp_index_ << "/" << waypoints_.size()
            << " target=(" << target.X() << "," << target.Y() << "," << target.Z() << ") err=(" << err.X() << ","
            << err.Y() << "," << err.Z() << ") F=(" << force.X() << "," << force.Y() << "," << force.Z() << ") roll="
            << roll_deg << "deg pitch=" << pitch_deg << "deg\n";
    }
  }
}

GZ_REGISTER_MODEL_PLUGIN(InspectionPathControllerPlugin)

}  // namespace gazebo

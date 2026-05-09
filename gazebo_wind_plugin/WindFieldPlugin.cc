#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include <ignition/math/Vector3.hh>

#include <cmath>
#include <string>

#include "lut_reader/WindLUT.hh"

namespace gazebo {

class WindFieldPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model->GetWorld();

    if (!sdf) {
      gzerr << "[WindFieldPlugin] SDF is null\n";
      return;
    }

    const std::string json_path = sdf->Get<std::string>("lut_json", "").first;
    const std::string vti_path = sdf->Get<std::string>("lut_vti", "").first;
    link_name_ = sdf->Get<std::string>("link_name", "base_link").first;

    rho_ = sdf->Get<double>("rho", 1.225).first;
    C_D_ = sdf->Get<double>("C_D", 1.0).first;
    area_ = sdf->Get<double>("area", 0.04).first;

    offset_x_ = sdf->Get<double>("world_to_lut_offset_x", 0.0).first;
    offset_y_ = sdf->Get<double>("world_to_lut_offset_y", 0.0).first;
    offset_z_ = sdf->Get<double>("world_to_lut_offset_z", 0.0).first;

    log_every_n_ = sdf->Get<int>("log_every_n", 0).first;

    hotspot_x_ = sdf->Get<double>("hotspot_x", 1420.0).first;
    hotspot_y_ = sdf->Get<double>("hotspot_y", -880.0).first;
    hotspot_z_ = sdf->Get<double>("hotspot_z", 145.0).first;

    enable_wind_torque_ = sdf->Get<bool>("enable_wind_torque", false).first;
    wind_torque_arm_z_ = sdf->Get<double>("wind_torque_arm_z", 0.15).first;

    hotspot_snap_outdoor_ = sdf->Get<bool>("hotspot_snap_outdoor", true).first;
    hotspot_snap_max_radius_m_ = sdf->Get<double>("hotspot_snap_max_radius_m", 120.0).first;
    hotspot_snap_min_wind_ = sdf->Get<double>("hotspot_snap_min_wind", 0.05).first;

    if (json_path.empty() || vti_path.empty()) {
      gzerr << "[WindFieldPlugin] Missing <lut_json> or <lut_vti> in SDF\n";
      return;
    }

    std::string err;
    if (!lut_.loadFromJsonAndVti(json_path, vti_path, &err)) {
      gzerr << "[WindFieldPlugin] LUT load failed: " << err << "\n";
      return;
    }

    gzmsg << "[WindFieldPlugin] LUT loaded: dims=(" << lut_.dims[0] << "," << lut_.dims[1] << "," << lut_.dims[2]
          << ") origin=(" << lut_.origin[0] << "," << lut_.origin[1] << "," << lut_.origin[2] << ") spacing=("
          << lut_.spacing[0] << "," << lut_.spacing[1] << "," << lut_.spacing[2] << ")\n";

    {
      double hx = hotspot_x_;
      double hy = hotspot_y_;
      double hz = hotspot_z_;
      if (hotspot_snap_outdoor_) {
        double sx = hx;
        double sy = hy;
        double sz = hz;
        if (lut_.snapHotspotNearestOutdoor(hx, hy, hz, hotspot_snap_max_radius_m_, hotspot_snap_min_wind_, &sx, &sy,
                                           &sz)) {
          const double moved = std::hypot(sx - hx, sy - hy) + std::fabs(sz - hz);
          if (moved > 1.0e-3) {
            gzmsg << "[WindFieldPlugin] hotspot_snap_outdoor: (" << hx << "," << hy << "," << hz << ") -> (" << sx
                  << "," << sy << "," << sz << ") within " << hotspot_snap_max_radius_m_ << " m (LUT mask / low-|U|)\n";
          }
          hx = sx;
          hy = sy;
          hz = sz;
        } else {
          gzwarn << "[WindFieldPlugin] hotspot_snap_outdoor failed within " << hotspot_snap_max_radius_m_
                 << " m; using raw hotspot\n";
        }
      }
      const auto hv = lut_.query(hx, hy, hz);
      const double u_mag = std::sqrt(static_cast<double>(hv[0]) * hv[0] + static_cast<double>(hv[1]) * hv[1] +
                                     static_cast<double>(hv[2]) * hv[2]);
      gzmsg << "[WindFieldPlugin] hotspot_check LUT(" << hx << "," << hy << "," << hz << ") wind=(" << hv[0] << ","
            << hv[1] << "," << hv[2] << ") |U|=" << u_mag << " m/s\n";
    }

    update_conn_ = event::Events::ConnectWorldUpdateBegin(std::bind(&WindFieldPlugin::OnUpdate, this));
  }

  void OnUpdate() {
    if (!model_) return;

    auto link = model_->GetLink(link_name_);
    if (!link) return;

    const auto pose = link->WorldPose();
    const double x = pose.Pos().X() + offset_x_;
    const double y = pose.Pos().Y() + offset_y_;
    const double z = pose.Pos().Z() + offset_z_;

    const auto uvw = lut_.query(x, y, z);
    const double u_wind = uvw[0];
    const double v_wind = uvw[1];
    const double w_wind = uvw[2];

    const auto vel = link->WorldLinearVel();
    const double u_rel = u_wind - vel.X();
    const double v_rel = v_wind - vel.Y();
    const double w_rel = w_wind - vel.Z();
    const double speed_rel = std::sqrt(u_rel * u_rel + v_rel * v_rel + w_rel * w_rel);

    if (!(speed_rel > 0.0)) return;

    const double F_scale = 0.5 * rho_ * C_D_ * area_ * speed_rel;
    ignition::math::Vector3d force(F_scale * u_rel, F_scale * v_rel, F_scale * w_rel);
    link->AddForce(force);

    if (enable_wind_torque_) {
      ignition::math::Vector3d moment_arm(0.0, 0.0, wind_torque_arm_z_);
      ignition::math::Vector3d torque = moment_arm.Cross(force);
      link->AddTorque(torque);
    }

    if (log_every_n_ > 0) {
      ++step_i_;
      if (step_i_ % static_cast<std::uint64_t>(log_every_n_) == 0) {
        gzmsg << "[WindFieldPlugin] pos=(" << pose.Pos().X() << "," << pose.Pos().Y() << "," << pose.Pos().Z()
              << ") wind=" << u_wind << "," << v_wind << "," << w_wind << " force=(" << force.X() << "," << force.Y()
              << "," << force.Z() << ")\n";
      }
    }
  }

private:
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  event::ConnectionPtr update_conn_;

  WindLUT lut_;
  std::string link_name_;

  double rho_{1.225};
  double C_D_{1.0};
  double area_{0.04};
  double offset_x_{0.0};
  double offset_y_{0.0};
  double offset_z_{0.0};

  int log_every_n_{0};
  std::uint64_t step_i_{0};

  double hotspot_x_{1420.0};
  double hotspot_y_{-880.0};
  double hotspot_z_{145.0};

  bool enable_wind_torque_{false};
  double wind_torque_arm_z_{0.15};

  bool hotspot_snap_outdoor_{true};
  double hotspot_snap_max_radius_m_{120.0};
  double hotspot_snap_min_wind_{0.05};
};

GZ_REGISTER_MODEL_PLUGIN(WindFieldPlugin)

}  // namespace gazebo


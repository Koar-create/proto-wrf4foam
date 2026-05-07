#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct WindLUT {
  std::array<double, 3> origin{0.0, 0.0, 0.0};
  std::array<double, 3> spacing{1.0, 1.0, 1.0};
  std::array<int, 3> dims{0, 0, 0};  // {Nx, Ny, Nz}

  // Flattened arrays in x-fastest order (VTK ImageData point order):
  // linear_index = (ix * Ny + iy) * Nz + iz
  std::vector<float> U;                 // size = Nx*Ny*Nz*3 (u,v,w)
  std::vector<std::uint8_t> valid_mask; // size = Nx*Ny*Nz (0/1). Optional.
  std::vector<std::uint8_t> inside_building; // size = Nx*Ny*Nz (0/1). Optional.

  bool loadFromJsonAndVti(const std::string& json_path, const std::string& vti_path, std::string* err);

  // Query trilinear-interpolated wind at (x,y,z) in LUT coordinates (meters).
  // Out-of-range returns {0,0,0}.
  // If a corner is invalid (valid_mask==0) it contributes as zero.
  std::array<float, 3> query(double x, double y, double z) const;

private:
  inline int idx(int ix, int iy, int iz) const {
    return (ix * dims[1] + iy) * dims[2] + iz;
  }
  inline int idx4(int ix, int iy, int iz, int c) const {
    return idx(ix, iy, iz) * 3 + c;
  }
};


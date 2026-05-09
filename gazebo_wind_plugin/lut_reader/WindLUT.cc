#include "lut_reader/WindLUT.hh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <sstream>

#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkXMLImageDataReader.h>

namespace {

std::string readAllText(const std::string& path, std::string* err) {
  std::ifstream f(path);
  if (!f) {
    if (err) {
      *err = "failed to open file: " + path + " (errno=" + std::to_string(errno) + ")";
    }
    return {};
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool parseJsonNumberArray3(const std::string& json, const std::string& key, std::array<double, 3>* out,
                           std::string* err) {
  // Minimal parser for `"key": [a, b, c]` (numbers can be ints or floats, whitespace allowed).
  // This avoids adding a JSON dependency for just 3 arrays.
  const std::string needle = "\"" + key + "\"";
  const std::size_t kpos = json.find(needle);
  if (kpos == std::string::npos) {
    if (err) *err = "json key not found: " + key;
    return false;
  }
  const std::size_t lbr = json.find('[', kpos);
  const std::size_t rbr = json.find(']', lbr);
  if (lbr == std::string::npos || rbr == std::string::npos || rbr <= lbr) {
    if (err) *err = "json array brackets not found for key: " + key;
    return false;
  }
  const std::string body = json.substr(lbr + 1, rbr - lbr - 1);
  std::istringstream iss(body);
  double a, b, c;
  char comma1, comma2;
  if (!(iss >> a)) {
    if (err) *err = "failed to parse first number for key: " + key;
    return false;
  }
  iss >> std::ws;
  if (!(iss >> comma1) || comma1 != ',') {
    if (err) *err = "failed to parse first comma for key: " + key;
    return false;
  }
  if (!(iss >> b)) {
    if (err) *err = "failed to parse second number for key: " + key;
    return false;
  }
  iss >> std::ws;
  if (!(iss >> comma2) || comma2 != ',') {
    if (err) *err = "failed to parse second comma for key: " + key;
    return false;
  }
  if (!(iss >> c)) {
    if (err) *err = "failed to parse third number for key: " + key;
    return false;
  }
  *out = {a, b, c};
  return true;
}

bool parseJsonIntArray3(const std::string& json, const std::string& key, std::array<int, 3>* out,
                        std::string* err) {
  std::array<double, 3> tmp{};
  if (!parseJsonNumberArray3(json, key, &tmp, err)) return false;
  *out = {static_cast<int>(std::lround(tmp[0])), static_cast<int>(std::lround(tmp[1])), static_cast<int>(std::lround(tmp[2]))};
  return true;
}

template <typename T>
void copyNumericArrayToU8(vtkDataArray* arr, std::vector<std::uint8_t>* out) {
  const vtkIdType n = arr->GetNumberOfTuples();
  out->resize(static_cast<std::size_t>(n));
  for (vtkIdType i = 0; i < n; ++i) {
    const double v = arr->GetTuple1(i);
    (*out)[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v != 0.0 ? 1 : 0);
  }
}

}  // namespace

bool WindLUT::loadFromJsonAndVti(const std::string& json_path, const std::string& vti_path, std::string* err) {
  std::string jerr;
  const std::string json = readAllText(json_path, &jerr);
  if (json.empty()) {
    if (err) *err = jerr;
    return false;
  }

  std::array<double, 3> json_origin{}, json_spacing{};
  std::array<int, 3> json_dims{};
  if (!parseJsonNumberArray3(json, "origin", &json_origin, err)) return false;
  if (!parseJsonNumberArray3(json, "spacing", &json_spacing, err)) return false;
  if (!parseJsonIntArray3(json, "dimensions", &json_dims, err)) return false;

  vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
  reader->SetFileName(vti_path.c_str());
  reader->Update();

  vtkImageData* img = reader->GetOutput();
  if (!img) {
    if (err) *err = "VTK reader returned null output";
    return false;
  }

  int vti_dims[3] = {0, 0, 0};
  img->GetDimensions(vti_dims);

  if (vti_dims[0] != json_dims[0] || vti_dims[1] != json_dims[1] || vti_dims[2] != json_dims[2]) {
    if (err) {
      *err = "dimension mismatch: json dims=(" + std::to_string(json_dims[0]) + "," + std::to_string(json_dims[1]) + "," +
             std::to_string(json_dims[2]) + ") vti dims=(" + std::to_string(vti_dims[0]) + "," + std::to_string(vti_dims[1]) +
             "," + std::to_string(vti_dims[2]) + ")";
    }
    return false;
  }

  origin = json_origin;
  spacing = json_spacing;
  dims = json_dims;

  vtkPointData* pd = img->GetPointData();
  if (!pd) {
    if (err) *err = "VTK image has no point data";
    return false;
  }

  vtkDataArray* U_arr = pd->GetArray("U");
  if (!U_arr) U_arr = pd->GetVectors("U");
  if (!U_arr) {
    if (err) *err = "VTK point data missing array 'U'";
    return false;
  }

  if (U_arr->GetNumberOfComponents() != 3) {
    if (err) *err = "array 'U' must have 3 components, got " + std::to_string(U_arr->GetNumberOfComponents());
    return false;
  }

  const vtkIdType npts = img->GetNumberOfPoints();
  if (vtkFloatArray* fa = vtkFloatArray::SafeDownCast(U_arr)) {
    const float* ptr = fa->GetPointer(0);
    if (!ptr) {
      if (err) *err = "array 'U' has null data pointer";
      return false;
    }
    U.assign(ptr, ptr + static_cast<std::size_t>(npts) * 3);
  } else {
    // Fallback (slower): copy via GetTuple.
    U.resize(static_cast<std::size_t>(npts) * 3);
    for (vtkIdType i = 0; i < npts; ++i) {
      double tuple[3] = {0.0, 0.0, 0.0};
      U_arr->GetTuple(i, tuple);
      const std::size_t base = static_cast<std::size_t>(i) * 3;
      U[base + 0] = static_cast<float>(tuple[0]);
      U[base + 1] = static_cast<float>(tuple[1]);
      U[base + 2] = static_cast<float>(tuple[2]);
    }
  }

  if (vtkDataArray* vm = pd->GetArray("valid_mask")) {
    if (vtkUnsignedCharArray* u8 = vtkUnsignedCharArray::SafeDownCast(vm)) {
      const unsigned char* ptr = u8->GetPointer(0);
      valid_mask.assign(ptr, ptr + static_cast<std::size_t>(npts));
    } else {
      copyNumericArrayToU8<std::uint8_t>(vm, &valid_mask);
    }
  } else if (vtkDataArray* vm2 = pd->GetArray("vtkValidPointMask")) {
    if (vtkUnsignedCharArray* u8 = vtkUnsignedCharArray::SafeDownCast(vm2)) {
      const unsigned char* ptr = u8->GetPointer(0);
      valid_mask.assign(ptr, ptr + static_cast<std::size_t>(npts));
    } else {
      copyNumericArrayToU8<std::uint8_t>(vm2, &valid_mask);
    }
  } else {
    valid_mask.clear();
  }

  if (vtkDataArray* ib = pd->GetArray("inside_building")) {
    if (vtkUnsignedCharArray* u8 = vtkUnsignedCharArray::SafeDownCast(ib)) {
      const unsigned char* ptr = u8->GetPointer(0);
      inside_building.assign(ptr, ptr + static_cast<std::size_t>(npts));
    } else {
      copyNumericArrayToU8<std::uint8_t>(ib, &inside_building);
    }
  } else {
    inside_building.clear();
  }

  return true;
}

bool WindLUT::cellIsOutdoor(int ix, int iy, int iz) const {
  if (ix < 0 || iy < 0 || iz < 0 || ix >= dims[0] || iy >= dims[1] || iz >= dims[2]) return false;
  const std::size_t id = static_cast<std::size_t>(idx(ix, iy, iz));
  if (!valid_mask.empty() && valid_mask[id] == 0) return false;
  if (!inside_building.empty() && inside_building[id] != 0) return false;
  return true;
}

bool WindLUT::snapHotspotNearestOutdoor(double x_in, double y_in, double z_in, double max_radius_m,
                                        double min_wind_fallback, double* out_x, double* out_y,
                                        double* out_z) const {
  if (out_x == nullptr || out_y == nullptr || out_z == nullptr) return false;
  if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || U.empty()) return false;

  auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); };

  const int iz =
      clampi(static_cast<int>(std::lround((z_in - origin[2]) / spacing[2])), 0, dims[2] - 1);
  const int ixc =
      clampi(static_cast<int>(std::lround((x_in - origin[0]) / spacing[0])), 0, dims[0] - 1);
  const int iyc =
      clampi(static_cast<int>(std::lround((y_in - origin[1]) / spacing[1])), 0, dims[1] - 1);

  const double z_grid = origin[2] + static_cast<double>(iz) * spacing[2];
  const double max_r2 = max_radius_m * max_radius_m;
  const int max_step =
      std::max(1, static_cast<int>(std::ceil(max_radius_m / std::min(spacing[0], spacing[1]))));

  auto umag_at_grid_xy = [&](int ix, int iy) -> double {
    const double px = origin[0] + static_cast<double>(ix) * spacing[0];
    const double py = origin[1] + static_cast<double>(iy) * spacing[1];
    const auto w = query(px, py, z_grid);
    return std::sqrt(static_cast<double>(w[0]) * w[0] + static_cast<double>(w[1]) * w[1] +
                       static_cast<double>(w[2]) * w[2]);
  };

  auto search_xy = [&](bool use_mask) -> bool {
    bool found = false;
    double best_d2 = 0.0;
    int best_ix = 0;
    int best_iy = 0;
    for (int dx = -max_step; dx <= max_step; ++dx) {
      for (int dy = -max_step; dy <= max_step; ++dy) {
        const int ix = ixc + dx;
        const int iy = iyc + dy;
        if (ix < 0 || ix >= dims[0] || iy < 0 || iy >= dims[1]) continue;
        const double px = origin[0] + static_cast<double>(ix) * spacing[0];
        const double py = origin[1] + static_cast<double>(iy) * spacing[1];
        const double tdx = px - x_in;
        const double tdy = py - y_in;
        if (tdx * tdx + tdy * tdy > max_r2) continue;

        bool ok = false;
        if (use_mask) {
          ok = cellIsOutdoor(ix, iy, iz);
        } else {
          ok = umag_at_grid_xy(ix, iy) >= min_wind_fallback;
        }
        if (!ok) continue;
        const double d2 = tdx * tdx + tdy * tdy;
        if (!found || d2 < best_d2) {
          found = true;
          best_d2 = d2;
          best_ix = ix;
          best_iy = iy;
        }
      }
    }
    if (found) {
      *out_x = origin[0] + static_cast<double>(best_ix) * spacing[0];
      *out_y = origin[1] + static_cast<double>(best_iy) * spacing[1];
      *out_z = z_grid;
      return true;
    }
    return false;
  };

  const bool have_building_mask = !inside_building.empty();

  if (have_building_mask) {
    if (cellIsOutdoor(ixc, iyc, iz)) {
      *out_x = x_in;
      *out_y = y_in;
      *out_z = z_in;
      return true;
    }
    if (search_xy(true)) return true;
    return search_xy(false);
  }

  const auto w0 = query(x_in, y_in, z_in);
  const double u0 = std::sqrt(static_cast<double>(w0[0]) * w0[0] + static_cast<double>(w0[1]) * w0[1] +
                              static_cast<double>(w0[2]) * w0[2]);
  if (u0 >= min_wind_fallback) {
    *out_x = x_in;
    *out_y = y_in;
    *out_z = z_in;
    return true;
  }
  return search_xy(false);
}

std::array<float, 3> WindLUT::query(double x, double y, double z) const {
  if (dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) return {0.f, 0.f, 0.f};
  if (U.empty()) return {0.f, 0.f, 0.f};

  const double fx = (x - origin[0]) / spacing[0];
  const double fy = (y - origin[1]) / spacing[1];
  const double fz = (z - origin[2]) / spacing[2];

  if (fx < 0.0 || fy < 0.0 || fz < 0.0) return {0.f, 0.f, 0.f};
  if (fx > static_cast<double>(dims[0] - 1) || fy > static_cast<double>(dims[1] - 1) ||
      fz > static_cast<double>(dims[2] - 1))
    return {0.f, 0.f, 0.f};

  const int ix0 = std::min(static_cast<int>(fx), dims[0] - 2);
  const int iy0 = std::min(static_cast<int>(fy), dims[1] - 2);
  const int iz0 = std::min(static_cast<int>(fz), dims[2] - 2);

  const double tx = fx - ix0;
  const double ty = fy - iy0;
  const double tz = fz - iz0;

  std::array<float, 3> out{0.f, 0.f, 0.f};
  for (int dx = 0; dx <= 1; ++dx) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dz = 0; dz <= 1; ++dz) {
        const double w = (dx ? tx : 1.0 - tx) * (dy ? ty : 1.0 - ty) * (dz ? tz : 1.0 - tz);
        const int ix = ix0 + dx;
        const int iy = iy0 + dy;
        const int iz = iz0 + dz;
        const int i = idx(ix, iy, iz);

        if (!valid_mask.empty() && valid_mask[static_cast<std::size_t>(i)] == 0) continue;
        if (!inside_building.empty() && inside_building[static_cast<std::size_t>(i)] != 0) continue;

        const std::size_t base = static_cast<std::size_t>(i) * 3;
        out[0] += static_cast<float>(w) * U[base + 0];
        out[1] += static_cast<float>(w) * U[base + 1];
        out[2] += static_cast<float>(w) * U[base + 2];
      }
    }
  }
  return out;
}


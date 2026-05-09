#!/usr/bin/env bash
# RBM-4 pt1：无水平 PID，风将机体推向 demo_building_collision（默认 box 代理）
# 须在仓库根目录执行：cd /path/to/WRF-OpenFOAM-Coupling && ./scripts/run_gazebo_guangzhou_demo_pt1_crash.sh <子命令>
set -euo pipefail

WORLD_REL="gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world"

require_repo_root() {
  if [[ ! -f "${PWD}/${WORLD_REL}" ]]; then
    echo "错误：请在 WRF-OpenFOAM-Coupling 仓库根目录下执行（未找到 ${WORLD_REL}，当前目录: ${PWD}）" >&2
    exit 1
  fi
}

export_gazebo_env() {
  export GAZEBO_PLUGIN_PATH="${PWD}/gazebo_wind_plugin/build:${GAZEBO_PLUGIN_PATH:-}"
  export GAZEBO_MODEL_PATH="${PWD}/gazebo_wind_plugin/models:/usr/share/gazebo-11/models:${GAZEBO_MODEL_PATH:-}"
  export GAZEBO_MODEL_DATABASE_URI=""
}

cmd_help() {
  cat <<'EOF'
用法（在 WRF-OpenFOAM-Coupling 根目录）:
  ./scripts/run_gazebo_guangzhou_demo_pt1_crash.sh <子命令>

子命令:
  help              显示本说明
  env               打印当前 shell 下需 export 的变量（便于复制到其它终端）
  build             cmake 配置并编译插件（生成 libWindFieldPlugin.so 等）
  cache             将高精度 LUT 从仓库 data/ 同步到 WSL 原生缓存（若本机已有可跳过）
  arrows            从缓存 NPZ 重新生成 wind_arrows_hotspot_hires 模型（需已安装 python3 + numpy）
  collision-build   一键跑碰撞流水线：extract（CoACD 凸分解）→ build（生成 demo_building_collision/model.sdf）→ verify（LUT 对齐 PNG），再可选 probe（推荐 pt1 spawn）
                    环境变量：BUILDINGS_STL=<path>（默认尝试 LUT 兜底）；HOTSPOT='1470,1350'；RADIUS_M=50；LUT_VTI=<path>
  server            仅启动 gzserver（无 GUI），--verbose
  gui               启动 gazebo（gzserver + gzclient），--verbose
  smoke             约 22s 无头运行 gzserver 做快速日志自检（timeout）

World: gazebo_wind_plugin/worlds/guangzhou_demo_pt1_crash.world
模型: iris_wind_quad_hires_pt1_crash（与 run_gazebo_guangzhou_wind_hires_demo.sh 共用 hires LUT 与箭头 bbox 逻辑）

注意: 若本机已有 gzserver 占用默认 master 端口 11345，需先结束旧进程再启动第二套仿真。
EOF
}

cmd_env() {
  require_repo_root
  cat <<EOF
export GAZEBO_PLUGIN_PATH="${PWD}/gazebo_wind_plugin/build:\${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_MODEL_PATH="${PWD}/gazebo_wind_plugin/models:/usr/share/gazebo-11/models:\${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI=""
EOF
}

cmd_build() {
  require_repo_root
  cmake -S gazebo_wind_plugin -B gazebo_wind_plugin/build
  cmake --build gazebo_wind_plugin/build -j
}

cmd_cache() {
  require_repo_root
  local src="data/wind_lut/20250903_1400_hires"
  local dst="${HOME}/wrf_openfoam_coupling_cache/wind_lut/20250903_1400_hires"
  mkdir -p "${dst}"
  local copied=0
  for f in wind_lut.json wind_lut.vti wind_lut.npz; do
    if [[ -f "${src}/${f}" ]]; then
      cp -v "${src}/${f}" "${dst}/"
      copied=1
    fi
  done
  if [[ "${copied}" -eq 0 ]]; then
    echo "提示：仓库 ${src}/ 下暂无可复制文件（常见于 LUT 仅在 Windows 盘）；若 ${dst} 已有文件，可直接 server/gui。" >&2
  fi
}

cmd_arrows() {
  require_repo_root
  local npz="${HOME}/wrf_openfoam_coupling_cache/wind_lut/20250903_1400_hires/wind_lut.npz"
  if [[ ! -f "${npz}" ]]; then
    echo "错误：未找到 ${npz}，请先 cache 或手动放置 hires LUT。" >&2
    exit 1
  fi
  python3 scripts/generate_gazebo_wind_arrows.py \
    --npz "${npz}" \
    --model-name wind_arrows_hotspot_hires \
    --out-model-dir gazebo_wind_plugin/models/wind_arrows_hotspot_hires \
    --color-mode hsv \
    --bbox-mode \
    --x-min 1050 --x-max 1850 --y-min 950 --y-max 1750 --step 80 \
    --z-levels 10
}

cmd_collision_build() {
  require_repo_root
  local hotspot="${HOTSPOT:-1470,1350}"
  local radius_m="${RADIUS_M:-50}"
  local buildings_stl="${BUILDINGS_STL:-}"
  local lut_vti="${LUT_VTI:-${HOME}/wrf_openfoam_coupling_cache/wind_lut/20250903_1400_hires/wind_lut.vti}"
  local manifest="data/demo_assets/collision_manifest.json"
  local model_dir="gazebo_wind_plugin/models/demo_building_collision"

  echo "[collision-build] step 1/3 extract"
  if [[ -n "${buildings_stl}" && -f "${buildings_stl}" ]]; then
    python3 scripts/extract_demo_building_collision.py \
      --input "${buildings_stl}" \
      --hotspot "${hotspot}" \
      --radius-m "${radius_m}"
  elif [[ -f "${lut_vti}" ]]; then
    echo "[collision-build] BUILDINGS_STL not set or missing; using LUT fallback ${lut_vti}"
    python3 scripts/extract_demo_building_collision.py \
      --from-lut "${lut_vti}" \
      --hotspot "${hotspot}" \
      --radius-m "${radius_m}"
  else
    echo "错误：既无 BUILDINGS_STL 又无 LUT_VTI 可用（${lut_vti}）。" >&2
    exit 1
  fi

  echo "[collision-build] step 2/3 build model"
  python3 scripts/build_demo_collision_model.py \
    --manifest "${manifest}" \
    --model-dir "${model_dir}"

  if [[ -f "${lut_vti}" ]]; then
    echo "[collision-build] step 3/3 verify alignment"
    set +e
    python3 scripts/verify_collision_alignment.py \
      --manifest "${manifest}" \
      --lut "${lut_vti}" \
      --z 80 \
      --out data/demo_assets/qc_alignment.png
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 && "${rc}" -ne 2 ]]; then
      echo "[collision-build] verify failed unexpectedly rc=${rc}" >&2
      exit "${rc}"
    fi
    echo "[collision-build] (optional) recommended pt1 spawn:"
    python3 scripts/probe_wind_at_hotspot.py \
      --lut "${lut_vti}" \
      --manifest "${manifest}" \
      --hotspot 1470,1350,80 || true
  else
    echo "[collision-build] LUT_VTI 不存在，跳过 verify/probe；可手动运行 verify_collision_alignment.py / probe_wind_at_hotspot.py"
  fi
}

cmd_server() {
  require_repo_root
  export_gazebo_env
  gzserver "${WORLD_REL}" --verbose
}

cmd_gui() {
  require_repo_root
  export_gazebo_env
  gazebo "${WORLD_REL}" --verbose
}

cmd_smoke() {
  require_repo_root
  export_gazebo_env
  timeout 22s gzserver "${WORLD_REL}" --verbose
}

main() {
  local sub="${1:-help}"
  case "${sub}" in
    help|-h|--help) cmd_help ;;
    env) cmd_env ;;
    build) cmd_build ;;
    cache) cmd_cache ;;
    arrows) cmd_arrows ;;
    collision-build|collision|build-collision) cmd_collision_build ;;
    server) cmd_server ;;
    gui) cmd_gui ;;
    smoke) cmd_smoke ;;
    *)
      echo "未知子命令: ${sub}" >&2
      cmd_help
      exit 1
      ;;
  esac
}

main "$@"

#!/usr/bin/env bash
# RBM-4 pt2：与 pt1 同巡检航线 + 演示型 barrier（绕开建筑）；stub 控制器可换 HOCBF
# 须在仓库根目录执行：cd /path/to/WRF-OpenFOAM-Coupling && ./scripts/run_gazebo_guangzhou_demo_pt2_hover.sh <子命令>
set -euo pipefail

WORLD_REL="gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world"

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
  ./scripts/run_gazebo_guangzhou_demo_pt2_hover.sh <子命令>

子命令:
  help     显示本说明
  env      打印当前 shell 下需 export 的变量（便于复制到其它终端）
  build    cmake 配置并编译插件（生成 libWindFieldPlugin.so 等）
  cache    将高精度 LUT 从仓库 data/ 同步到 WSL 原生缓存（若本机已有可跳过）
  arrows   从缓存 NPZ 重新生成 wind_arrows_hotspot_hires 模型（需已安装 python3 + numpy）
  server   仅启动 gzserver（无 GUI），--verbose
  gui      启动 gazebo（gzserver + gzclient），--verbose
  smoke    约 75s 无头 gzserver（覆盖部分航线与 barrier 日志）

World: gazebo_wind_plugin/worlds/guangzhou_demo_pt2_hover.world
模型: iris_wind_quad_hires_pt2_hover（日志中应出现 roll=/pitch= 等）

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
  timeout 75s gzserver "${WORLD_REL}" --verbose
}

main() {
  local sub="${1:-help}"
  case "${sub}" in
    help|-h|--help) cmd_help ;;
    env) cmd_env ;;
    build) cmd_build ;;
    cache) cmd_cache ;;
    arrows) cmd_arrows ;;
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

#!/usr/bin/env bash
# 标准风场 demo：gazebo_wind_plugin/worlds/guangzhou_wind.world
# 须在仓库根目录执行：cd /path/to/WRF-OpenFOAM-Coupling && ./scripts/run_gazebo_guangzhou_wind_demo.sh <子命令>
set -euo pipefail

WORLD_REL="gazebo_wind_plugin/worlds/guangzhou_wind.world"

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
  ./scripts/run_gazebo_guangzhou_wind_demo.sh <子命令>

子命令:
  help     显示本说明
  env      打印当前 shell 下需 export 的变量（便于复制到其它终端）
  build    cmake 配置并编译插件（生成 libWindFieldPlugin.so 等）
  cache    将 LUT 从仓库 data/ 同步到 WSL 原生缓存（降低 VTI 读取延迟）
  server   仅启动 gzserver（无 GUI），--verbose
  gui      启动 gazebo（gzserver + gzclient），--verbose
  smoke    约 22s 无头运行 gzserver 做快速日志自检（timeout）

等价手工命令参考:
  export GAZEBO_PLUGIN_PATH="$PWD/gazebo_wind_plugin/build:${GAZEBO_PLUGIN_PATH:-}"
  export GAZEBO_MODEL_PATH="$PWD/gazebo_wind_plugin/models:/usr/share/gazebo-11/models:${GAZEBO_MODEL_PATH:-}"
  export GAZEBO_MODEL_DATABASE_URI=""
  gzserver gazebo_wind_plugin/worlds/guangzhou_wind.world --verbose
  gazebo gazebo_wind_plugin/worlds/guangzhou_wind.world --verbose
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
  local dst="${HOME}/wrf_openfoam_coupling_cache/wind_lut/20250903_1400"
  mkdir -p "${dst}"
  cp -v data/wind_lut/20250903_1400/wind_lut.json "${dst}/"
  cp -v data/wind_lut/20250903_1400/wind_lut.vti "${dst}/"
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

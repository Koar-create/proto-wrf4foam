#!/bin/bash
# NOTE: some tools (notably `gazebo --version`) may return non-zero even on success.
# This script is for diagnostics only; it should keep going and print as much as possible.
set -u

echo "=== Gazebo version ==="
gazebo --version || true

echo
echo "=== pkg-config gazebo ==="
pkg-config --modversion gazebo 2>/dev/null || echo "pkg-config gazebo not found"

echo
echo "=== libvtk ==="
(dpkg -l | grep -E "libvtk[0-9]" | head -20) 2>/dev/null || true

echo
echo "=== cmake ==="
cmake --version 2>/dev/null || true

echo
echo "=== GCC ==="
gcc --version 2>/dev/null | head -1 || true

echo
echo "=== WSL2 distro ==="
cat /etc/os-release 2>/dev/null | grep PRETTY_NAME || true


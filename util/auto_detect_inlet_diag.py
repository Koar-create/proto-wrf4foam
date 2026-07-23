#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

def read_foam_vector_field(filepath):
    """鲁棒地解析 OpenFOAM 格式的边界 U 场"""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data = []
    in_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("/*"): 
            continue
        if not in_data:
            if line == "(": 
                in_data = True
        else:
            if line == ")": 
                break
            if line.startswith("("):
                vals = line.strip("()").split()
                data.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(data)

PATCHES = ['west', 'east', 'south', 'north']
DEFAULT_Z_MAX = 300.0  # 关注高度上限 (m)：人行/建筑冠层及附近边界层


def classify_wind_direction(avg_u: float, avg_v: float) -> str:
    if avg_u > 0 and avg_v > 0:
        return "西南风 (Southwesterly)"
    if avg_u < 0 and avg_v > 0:
        return "东南风 (Southeasterly)"
    if avg_u > 0 and avg_v < 0:
        return "西北风 (Northwesterly)"
    return "东北风 (Northeasterly)"


_OUTWARD_NORMAL = {
    'west': (-1.0, 0.0),
    'east': (1.0, 0.0),
    'south': (0.0, -1.0),
    'north': (0.0, 1.0),
}


def outward_normal_velocity(patch: str, u: float, v: float) -> float:
    """沿该边界外法向的速度分量：正值=流出域外，负值=流入域内。"""
    nx, ny = _OUTWARD_NORMAL[patch]
    return u * nx + v * ny


def is_inlet_patch(patch: str, u: float, v: float) -> bool:
    return outward_normal_velocity(patch, u, v) < 0


def read_boundary_wind_data(
    bd_dir: str,
    *,
    level: str = "surface",
    z_min: float = 0.0,
    z_max: float | None = None,
) -> dict[str, dict[str, float]]:
    """从 OpenFOAM boundaryData 读取各 patch 平均 U/V。

    level:
      - "surface": 仅取最靠近地面且 z>0 的一层（通常 z≈10 m），
        与近地面 CFD 关注层及 Region B 图一致（默认主判据）。
      - "band": 对 z ∈ (z_min, z_max] 的所有点做平均（默认 z_max=300 m），
        用作“关注高度段”对照，避免被域顶以上强平流扭曲。
      - "full": 整根边界柱（0 到域顶）逐点平均，可能被高空风向反转扭曲。
    """
    if level not in ("surface", "band", "full"):
        raise ValueError(f"未知 level: {level}（应为 surface/band/full）")
    if level == "band" and z_max is None:
        z_max = DEFAULT_Z_MAX

    wind_data: dict[str, dict[str, float]] = {}
    for patch in PATCHES:
        u_file = os.path.join(bd_dir, patch, "0", "U")
        u_array = read_foam_vector_field(u_file)
        if u_array is None or len(u_array) == 0:
            continue

        mask = np.ones(len(u_array), dtype=bool)
        if level in ("surface", "band"):
            pts_file = os.path.join(bd_dir, patch, "points")
            pts = read_foam_vector_field(pts_file)
            if pts is not None and len(pts) == len(u_array):
                z = pts[:, 2]
                if level == "surface":
                    z_pos = z[z > 1e-6]
                    z_surf = float(z_pos.min()) if len(z_pos) else float(z.min())
                    mask = np.isclose(z, z_surf, atol=0.5)
                else:
                    mask = (z > z_min) & (z <= float(z_max))
                    if not np.any(mask):
                        continue

        wind_data[patch] = {
            'U': float(np.mean(u_array[mask, 0])),
            'V': float(np.mean(u_array[mask, 1])),
        }
    return wind_data


def compute_mass_balance(
    wind_data: dict[str, dict[str, float]],
    flux_wind_data: dict[str, dict[str, float]],
) -> dict:
    """检验"强迫入流边界 + 自由(inletOutlet)边界"组合是否会导致质量不平衡。

    分类（谁是入流/谁是出流）完全由 wind_data（通常近地面层）动态决定，
    不假设固定是哪个方位——west/east/south/north 中任意子集都可能是入流或出流。

    flux_wind_data: 用于估算净法向流量的对照数据（推荐用关注高度带 average，
    默认 0–300 m；也可用全柱平均）。求解器在强迫入流边界上没有自由度去调流量；
    全域质量守恒主要靠剩余自由(inletOutlet)边界找平。本函数比较"自由边界为
    满足平衡所需承担的净出流"与"自由边界自身原始数据隐含的净出流"。
    """
    vout = {
        patch: outward_normal_velocity(patch, vel['U'], vel['V'])
        for patch, vel in flux_wind_data.items()
    }

    inlets = [
        p for p, vel in wind_data.items()
        if is_inlet_patch(p, vel['U'], vel['V'])
    ]
    outlets = [p for p in wind_data if p not in inlets]

    fixed_inflow_sum = sum(vout[p] for p in inlets if p in vout)
    outlets_raw_sum = sum(vout[p] for p in outlets if p in vout)
    required_outlet_total = -fixed_inflow_sum
    imbalance = required_outlet_total - outlets_raw_sum

    magnitudes = [abs(v) for v in vout.values()]
    ref_scale = float(np.mean(magnitudes)) if magnitudes else 0.0

    severity: str | None = None
    if not inlets:
        severity = (
            "❗ 严重: 没有任何边界被判定为入流，全部为自由边界——"
            "域内没有明确的驱动风场来源，请检查风向判定或数据是否异常。"
        )
    elif not outlets:
        severity = (
            "❗ 严重: 没有任何边界被判定为自由出流边界——"
            "全部强迫入流会导致压力方程缺少泄压出口（不可压求解器通常需要"
            "至少一个 fixedValue p 出口），请人工复核。"
        )
    elif ref_scale > 1e-6 and abs(imbalance) > 0.3 * ref_scale:
        outlets_str = ', '.join(p.upper() for p in outlets)
        severity = (
            f"⚠️  警告: 不平衡量 {imbalance:+.3f} m/s 达到典型流速幅值的 "
            f"{abs(imbalance) / ref_scale * 100:.0f}%，自由边界({outlets_str}) "
            f"出口速度可能被迫明显偏离真实风速，建议复核，或让更多边界"
            f"共同分担自由出流。"
        )

    return {
        'inlets': inlets,
        'outlets': outlets,
        'vout_full': vout,  # 兼容旧字段名
        'vout': vout,
        'fixed_inflow_sum': fixed_inflow_sum,
        'outlets_raw_sum': outlets_raw_sum,
        'required_outlet_total': required_outlet_total,
        'imbalance': imbalance,
        'ref_scale': ref_scale,
        'severity': severity,
    }


def format_mass_balance_section(
    balance: dict,
    *,
    flux_label: str = "关注高度带",
) -> str:
    """把 compute_mass_balance() 的结果渲染成可读文本段落。"""
    vout = balance.get('vout', balance.get('vout_full', {}))
    lines: list[str] = []
    lines.append("")
    lines.append("⚖️  [质量通量平衡校验 (Mass Flux Balance Check)]")
    lines.append(f"   （各边界{flux_label}净法向流量；正值=出流，负值=入流）")
    for patch in PATCHES:
        if patch not in vout:
            continue
        tag = 'IN ' if patch in balance['inlets'] else 'OUT'
        lines.append(f"   [{tag}] {patch.upper():<5}: {vout[patch]:+.3f} m/s")

    inlets_str = ', '.join(p.upper() for p in balance['inlets']) or '(无)'
    outlets_str = ', '.join(p.upper() for p in balance['outlets']) or '(无)'
    lines.append(
        f"   强迫入流边界({inlets_str}) {flux_label}净出流合计: "
        f"{balance['fixed_inflow_sum']:+.3f} m/s"
    )
    lines.append(
        f"   自由边界({outlets_str}) 需承担净出流(维持全域质量守恒): "
        f"{balance['required_outlet_total']:+.3f} m/s"
    )
    lines.append(
        f"   自由边界自身原始数据净出流合计: {balance['outlets_raw_sum']:+.3f} m/s"
    )
    lines.append(f"   不平衡量(需求 − 原始): {balance['imbalance']:+.3f} m/s")

    if balance['severity']:
        lines.append(f"   {balance['severity']}")
    else:
        lines.append("   ✅ 不平衡量在合理范围内，自由边界出口速度预期接近其原始真实风速。")
    return "\n".join(lines)


def recommend_boundary_conditions(
    wind_data: dict[str, dict[str, float]],
    flux_wind_data: dict[str, dict[str, float]] | None = None,
) -> dict:
    """根据近地面风向判定，返回各侧向边界的推荐 U/p 类型。"""
    if flux_wind_data is None:
        flux_wind_data = wind_data
    balance = compute_mass_balance(wind_data, flux_wind_data)
    recommended: dict[str, dict[str, str]] = {}
    for patch in PATCHES:
        if patch not in wind_data:
            continue
        if patch in balance['inlets']:
            recommended[patch] = {
                'U': 'timeVaryingMappedFixedValue',
                'p': 'zeroGradient',
            }
        else:
            recommended[patch] = {
                'U': 'inletOutlet',
                'p': 'fixedValue',
            }
    return {
        'patches': recommended,
        'inlets': balance['inlets'],
        'outlets': balance['outlets'],
        'balance': balance,
    }


def format_boundary_flux_report(
    wind_data: dict[str, dict[str, float]],
    *,
    warnings: list[str] | None = None,
    compare_wind_data: dict[str, dict[str, float]] | None = None,
    compare_label: str = "关注高度带平均",
    flux_wind_data: dict[str, dict[str, float]] | None = None,
    flux_label: str = "全柱平均",
    level_note: str | None = None,
) -> str:
    """生成边界风向与出入口诊断文本（与 CLI 打印格式一致）。

    compare_wind_data: 对照数据（通常 0–z_max 带平均），仅用于检查与近地面
    主判定是否一致；不一致才报警。
    flux_wind_data: 质量通量数据（通常全柱平均）。OpenFOAM 的
    timeVaryingMappedFixedValue 会映射整根廓线，因此用全柱估计更贴近求解器。
    若未提供，则回退到 compare_wind_data。
    """
    if flux_wind_data is None:
        flux_wind_data = compare_wind_data

    lines: list[str] = []
    lines.append("=" * 65)
    lines.append(" 🌬️  WRF 边界风向与出入口自动识别 (Boundary Flux Detector)")
    lines.append("=" * 65)
    if level_note:
        lines.append(f"   {level_note}")

    if warnings:
        for msg in warnings:
            lines.append(msg)

    if not wind_data:
        lines.append("❌ 错误: 未能在任何边界读取到风速数据。")
        lines.append("=" * 65)
        return "\n".join(lines)

    avg_u = float(np.mean([v['U'] for v in wind_data.values()]))
    avg_v = float(np.mean([v['V'] for v in wind_data.values()]))
    wind_dir_str = classify_wind_direction(avg_u, avg_v)

    lines.append("")
    lines.append("📊 [宏观风速分析]")
    lines.append(f"   • 全域平均纬向风速 (U, 西-东): {avg_u:+.3f} m/s")
    lines.append(f"   • 全域平均经向风速 (V, 南-北): {avg_v:+.3f} m/s")
    lines.append(f"   • 气象学宏观风向判别: {wind_dir_str}")
    lines.append("")
    lines.append("🎯 [OpenFOAM 边界属性判定]")

    inlets: list[str] = []
    outlets: list[str] = []
    for patch, vel in wind_data.items():
        u, v = vel['U'], vel['V']
        is_inlet = is_inlet_patch(patch, u, v)
        if is_inlet:
            inlets.append(patch)
            lines.append(
                f"   🟢 {patch.upper():<5} 边界: 入气口 (Inlet)   [法向通量指向域内]"
            )
        else:
            outlets.append(patch)
            lines.append(
                f"   🔴 {patch.upper():<5} 边界: 出气口 (Outlet)  [法向通量指向域外]"
            )

        if compare_wind_data and patch in compare_wind_data:
            cu, cv = compare_wind_data[patch]['U'], compare_wind_data[patch]['V']
            is_inlet_cmp = is_inlet_patch(patch, cu, cv)
            if is_inlet_cmp != is_inlet:
                cmp_label = "入气口 (Inlet)" if is_inlet_cmp else "出气口 (Outlet)"
                lines.append(
                    f"        ⚠️  与{compare_label}判定不一致 "
                    f"(U={cu:+.3f}, V={cv:+.3f} → {cmp_label})，建议人工复核该边界"
                )

    lines.append("")
    lines.append("🛠️ [0/U 配置文件推荐行动 (Action Items)]")
    for patch in inlets:
        lines.append(f"   - {patch}: 设置为 timeVaryingMappedFixedValue (强迫入流)")
    for patch in outlets:
        lines.append(
            f"   - {patch}: 设置为 inletOutlet, inletValue uniform (0 0 0) (防御性自由出流)"
        )

    if flux_wind_data:
        balance = compute_mass_balance(wind_data, flux_wind_data)
        lines.append(format_mass_balance_section(balance, flux_label=flux_label))

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)


def analyze_boundary_flux(bd_dir, *, z_max: float = DEFAULT_Z_MAX):
    warnings: list[str] = []
    wind_data = read_boundary_wind_data(bd_dir, level="surface")
    compare_data = read_boundary_wind_data(bd_dir, level="band", z_max=z_max)
    flux_data = read_boundary_wind_data(bd_dir, level="full")
    compare_label = f"0–{z_max:g} m 带平均"
    for patch in PATCHES:
        if patch not in wind_data:
            warnings.append(f"⚠️  警告: 找不到或无法读取 {patch} 边界的 0/U 数据。")
    print(format_boundary_flux_report(
        wind_data,
        warnings=warnings,
        compare_wind_data=compare_data,
        compare_label=compare_label,
        flux_wind_data=flux_data,
        flux_label="全柱平均",
        level_note=(
            f"（判定依据: 近地面层 z≈10 m；一致性对照: {compare_label}；"
            f"质量平衡: 全柱，对应 mapped BC 整根廓线）"
        ),
    ))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='WRF 边界风向与出入口自动识别（可设关注高度上限）。',
    )
    parser.add_argument('bd_dir', help='OpenFOAM constant/boundaryData 目录')
    parser.add_argument(
        '--z-max', type=float, default=DEFAULT_Z_MAX,
        help=f'对照/质量平衡用的高度上限 (m)，默认 {DEFAULT_Z_MAX:g}',
    )
    args = parser.parse_args()

    if not os.path.exists(args.bd_dir):
        print(f"❌ 找不到指定的 boundaryData 目录: {args.bd_dir}")
        sys.exit(1)

    analyze_boundary_flux(args.bd_dir, z_max=args.z_max)


if __name__ == "__main__":
    main()

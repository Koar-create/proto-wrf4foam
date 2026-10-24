"""Height-layer bins, names, and QC thresholds."""

from __future__ import annotations

HEIGHT_BINS = [0, 300, 1000, 2100]
LAYER_NAMES_EN = [
    "Low (52–300 m)",
    "Mid (300–1000 m)",
    "High (1000–2000 m)",
]
LAYER_SHORT = {
    "Low (52–300 m)": "Low",
    "Mid (300–1000 m)": "Mid",
}

WS_MAX_OBS = 30.0
WS_MAX_CFD = 20.0


def summary_layers(*, skip_high: bool = False) -> list[str]:
    if skip_high:
        return LAYER_NAMES_EN[:-1]
    return list(LAYER_NAMES_EN)

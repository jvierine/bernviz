#!/usr/bin/env python3
"""
Export Sanya IGRF magnetic-meridian field lines for the interactive web view.

Run:
  conda run -n base python export_sanya_igrf_fieldlines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from render_sanya_isr_earth import (
    bperp_f_region_beam,
    IGRF_DATE,
    SANYA_LAT_DEG,
    SANYA_LON_DEG,
    EARTH_RADIUS_KM,
    F_REGION_ALT_KM,
    magnetic_meridian_seed_points,
    trace_igrf_field_line,
    xyz_to_lat_lon_alt,
    xyz_to_local_km,
)


OUT_PATH = Path("assets/sanya_igrf_fieldlines.js")


def curved_ground_z(east_km: np.ndarray, north_km: np.ndarray) -> np.ndarray:
    return -((east_km**2 + north_km**2) / (2.0 * EARTH_RADIUS_KM))


def existing_beam_payload() -> dict | None:
    if not OUT_PATH.exists():
        return None
    text = OUT_PATH.read_text(encoding="utf-8")
    payload = json.loads(text[text.index("=") + 1 :].strip().rstrip(";"))
    return payload.get("beam")


def generate_beam_payload() -> dict:
    beam_axis_xyz, target_xyz, b_unit_xyz, beam_azimuth, beam_elevation = bperp_f_region_beam()
    beam_axis_local = xyz_to_local_km(beam_axis_xyz)
    beam_alt = np.array([xyz_to_lat_lon_alt(row)[2] for row in beam_axis_xyz])
    order = np.argsort(beam_alt)
    beam_axis_local = beam_axis_local[order]
    beam_alt = beam_alt[order]
    keep = (beam_alt >= 0) & (beam_alt <= 1000)
    beam_axis_local = beam_axis_local[keep]
    beam_alt = beam_alt[keep]
    beam_axis_local = beam_axis_local[np.linspace(0, len(beam_axis_local) - 1, 90).astype(int)]
    beam_alt = beam_alt[np.linspace(0, len(beam_alt) - 1, 90).astype(int)]
    beam_axis_local[:, 2] = beam_alt + curved_ground_z(beam_axis_local[:, 0], beam_axis_local[:, 1])

    target_local = xyz_to_local_km(np.array([target_xyz]))[0]
    target_alt = xyz_to_lat_lon_alt(target_xyz)[2]
    target_local[2] = target_alt + curved_ground_z(target_local[0], target_local[1])
    b_local = xyz_to_local_km(
        np.array(
            [
                target_xyz,
                target_xyz + (140.0 / EARTH_RADIUS_KM) * b_unit_xyz,
            ]
        )
    )
    b_dir_local = b_local[1] - b_local[0]
    b_dir_local = b_dir_local / np.linalg.norm(b_dir_local)

    return {
        "axis": [[round(float(v), 3) for v in row] for row in beam_axis_local],
        "target": [round(float(v), 3) for v in target_local],
        "bUnitAtTarget": [round(float(v), 6) for v in b_dir_local],
        "targetAltitudeKm": F_REGION_ALT_KM,
        "axisAltitudeKm": [round(float(v), 3) for v in beam_alt],
        "maxAltitudeKm": round(float(beam_alt[-1]), 3),
        "azimuthDeg": round(float(beam_azimuth), 3),
        "elevationDeg": round(float(beam_elevation), 3),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(exist_ok=True)

    offsets_km = np.r_[np.linspace(-1400, -250, 5), np.linspace(0, 2700, 16)]
    lines = []
    for seed_lat, seed_lon in magnetic_meridian_seed_points(offsets_km):
        field_line = trace_igrf_field_line(seed_lat, seed_lon, alt_km=0.0)
        local = xyz_to_local_km(field_line)
        altitude = local[:, 2] - curved_ground_z(local[:, 0], local[:, 1])
        keep = (
            (local[:, 0] > -2200)
            & (local[:, 0] < 2800)
            & (local[:, 1] > -1800)
            & (local[:, 1] < 3100)
            & (altitude > -2)
            & (altitude < 1800)
        )
        local = local[keep]
        if local.shape[0] >= 2:
            local_altitude = local[:, 2] - curved_ground_z(local[:, 0], local[:, 1])
            min_index = int(np.argmin(local_altitude))
            if local_altitude[min_index] > 0.5:
                surface = local[min_index].copy()
                surface[2] = curved_ground_z(surface[0], surface[1])
                if min_index == 0:
                    local = np.vstack([surface, local])
                elif min_index == local.shape[0] - 1:
                    local = np.vstack([local, surface])
            lines.append([[round(float(v), 3) for v in row] for row in local])

    beam_payload = existing_beam_payload() or generate_beam_payload()

    payload = {
        "date": str(IGRF_DATE.date()),
        "sanya": {"lat": SANYA_LAT_DEG, "lon": SANYA_LON_DEG},
        "units": "local east,north,up kilometers relative to Sanya ISR",
        "fieldLines": lines,
        "beam": beam_payload,
    }

    OUT_PATH.write_text(
        "export const SANYA_IGRF_FIELDLINES = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH} with {len(lines)} field lines")


if __name__ == "__main__":
    main()

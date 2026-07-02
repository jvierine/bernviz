#!/usr/bin/env python3
"""
Render an open-source NASA Earth texture with the Sanya ISR radar location.

Data provenance:
  Earth texture: NASA Earth Observatory, Blue Marble Next Generation,
  "world.topo.bathy.200412.3x5400x2700.jpg"
  https://visibleearth.nasa.gov/images/73934/topography

Run:
  conda run -n base python render_sanya_isr_earth.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
import ppigrf
from PIL import Image


TEXTURE_URL = (
    "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/"
    "bmng-topography-bathymetry/december/world.topo.bathy.200412.3x5400x2700.jpg"
)

OUT_DIR = Path("figures")
ASSET_DIR = Path("assets")
TEXTURE_PATH = ASSET_DIR / "nasa_blue_marble_2004_12_5400x2700.jpg"
OUTPUT_PATH = OUT_DIR / "sanya_isr_west_meridian_fieldlines_altitude_scale.png"

# Sanya ISR approximate location, Hainan, China.
SANYA_LAT_DEG = 18.3
SANYA_LON_DEG = 109.6
EARTH_RADIUS_KM = 6371.0
IGRF_DATE = pd.Timestamp("2026-07-02")
F_REGION_ALT_KM = 300.0


def download_texture() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    if not TEXTURE_PATH.exists():
        print(f"Downloading NASA Blue Marble texture to {TEXTURE_PATH}")
        urlretrieve(TEXTURE_URL, TEXTURE_PATH)


def unit_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.array(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )


def local_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    up = unit_xyz(lat_deg, lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return east, north, up


def xyz_to_lat_lon_alt(xyz: np.ndarray) -> tuple[float, float, float]:
    """Convert render coordinates to spherical lat/lon/alt."""
    r = np.linalg.norm(xyz)
    lat = np.rad2deg(np.arcsin(xyz[2] / r))
    lon = np.rad2deg(np.arctan2(xyz[1], xyz[0]))
    alt_km = (r - 1.0) * EARTH_RADIUS_KM
    return lat, lon, alt_km


def enu_to_xyz(
    east_comp: float,
    north_comp: float,
    up_comp: float,
    lat_deg: float,
    lon_deg: float,
) -> np.ndarray:
    east, north, up = local_basis(lat_deg, lon_deg)
    return east_comp * east + north_comp * north + up_comp * up


def igrf_unit_xyz(xyz: np.ndarray) -> np.ndarray:
    """IGRF14 magnetic-field unit vector in render coordinates."""
    lat, lon, alt_km = xyz_to_lat_lon_alt(xyz)
    be, bn, bu = ppigrf.igrf(lon, lat, max(0.0, alt_km), IGRF_DATE)
    field = enu_to_xyz(float(be[0]), float(bn[0]), float(bu[0]), lat, lon)
    return field / np.linalg.norm(field)


def trace_igrf_direction(
    start_xyz: np.ndarray,
    direction_sign: float,
    step_km: float = 25.0,
    max_steps: int = 420,
    max_alt_km: float = 2600.0,
) -> np.ndarray:
    """Trace one side of an IGRF field line with a fixed-step RK4 integrator."""
    point = start_xyz.astype(float).copy()
    points = [point.copy()]
    h = step_km / EARTH_RADIUS_KM

    def rhs(p: np.ndarray) -> np.ndarray:
        return direction_sign * igrf_unit_xyz(p)

    for _ in range(max_steps):
        k1 = rhs(point)
        k2 = rhs(point + 0.5 * h * k1)
        k3 = rhs(point + 0.5 * h * k2)
        k4 = rhs(point + h * k3)
        point = point + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        points.append(point.copy())

        _, _, alt_km = xyz_to_lat_lon_alt(point)
        if alt_km < 0.0 or alt_km > max_alt_km:
            break

    return np.array(points)


def trace_igrf_field_line(lat_deg: float, lon_deg: float, alt_km: float = 120.0) -> np.ndarray:
    radius = 1.0 + alt_km / EARTH_RADIUS_KM
    start = radius * unit_xyz(lat_deg, lon_deg)
    backward = trace_igrf_direction(start, -1.0)
    forward = trace_igrf_direction(start, 1.0)
    return np.vstack((backward[::-1], forward[1:]))


def magnetic_meridian_seed_points(offsets_km: np.ndarray) -> list[tuple[float, float]]:
    """Seed ground points along Sanya's local magnetic meridian."""
    be, bn, _ = ppigrf.igrf(SANYA_LON_DEG, SANYA_LAT_DEG, 0.0, IGRF_DATE)
    east, north, up = local_basis(SANYA_LAT_DEG, SANYA_LON_DEG)
    horizontal = float(be[0]) * east + float(bn[0]) * north
    horizontal = horizontal / np.linalg.norm(horizontal)

    seeds = []
    for offset_km in offsets_km:
        surface_point = up + (offset_km / EARTH_RADIUS_KM) * horizontal
        surface_point = surface_point / np.linalg.norm(surface_point)
        lat_deg, lon_deg, _ = xyz_to_lat_lon_alt(surface_point)
        seeds.append((lat_deg, lon_deg))
    return seeds


def magnetic_meridian_azimuth_deg() -> float:
    be, bn, _ = ppigrf.igrf(SANYA_LON_DEG, SANYA_LAT_DEG, 0.0, IGRF_DATE)
    return np.rad2deg(np.arctan2(float(be[0]), float(bn[0])))


def beam_direction_xyz(lat_deg: float, lon_deg: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    east, north, up = local_basis(lat_deg, lon_deg)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    direction = np.sin(el) * up + np.cos(el) * (np.cos(az) * north + np.sin(az) * east)
    return direction / np.linalg.norm(direction)


def slant_range_for_altitude_km(elevation_deg: float, altitude_km: float) -> float:
    """Slant range from local ground to a spherical altitude shell."""
    sin_el = np.sin(np.deg2rad(elevation_deg))
    return -EARTH_RADIUS_KM * sin_el + np.sqrt(
        (EARTH_RADIUS_KM * sin_el) ** 2
        + 2.0 * EARTH_RADIUS_KM * altitude_km
        + altitude_km**2
    )


def bperp_f_region_beam() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Find the Sanya beam in the magnetic meridian perpendicular to IGRF B at F-region height."""
    origin = unit_xyz(SANYA_LAT_DEG, SANYA_LON_DEG)
    meridian_az = magnetic_meridian_azimuth_deg()
    candidates = []
    for azimuth in [meridian_az, meridian_az + 180.0]:
        for elevation in np.linspace(15.0, 88.0, 600):
            direction = beam_direction_xyz(SANYA_LAT_DEG, SANYA_LON_DEG, azimuth, elevation)
            slant_range_km = slant_range_for_altitude_km(elevation, F_REGION_ALT_KM)
            target = origin + (slant_range_km / EARTH_RADIUS_KM) * direction
            b_unit = igrf_unit_xyz(target)
            candidates.append((abs(np.dot(direction, b_unit)), azimuth, elevation, slant_range_km, target, b_unit))

    _, azimuth, elevation, slant_range_km, target, b_unit = min(candidates, key=lambda item: item[0])
    beam = beam_points(
        SANYA_LAT_DEG,
        SANYA_LON_DEG,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        max_alt_km=slant_range_for_altitude_km(elevation, 1000.0),
        samples=240,
    )
    return beam, target, b_unit, azimuth, elevation


def project_points_to_figure(fig: plt.Figure, ax: plt.Axes, points: np.ndarray) -> np.ndarray:
    x2, y2, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
    display_xy = ax.transData.transform(np.column_stack((x2, y2)))
    return fig.transFigure.inverted().transform(display_xy)


def add_projected_line(
    fig: plt.Figure,
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    figure_xy = project_points_to_figure(fig, ax, points)
    fig.add_artist(
        Line2D(
            figure_xy[:, 0],
            figure_xy[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            transform=fig.transFigure,
            zorder=1000,
        )
    )


def xyz_to_local_km(xyz: np.ndarray) -> np.ndarray:
    east, north, up = local_basis(SANYA_LAT_DEG, SANYA_LON_DEG)
    origin = unit_xyz(SANYA_LAT_DEG, SANYA_LON_DEG)
    delta_km = (xyz - origin[None, :]) * EARTH_RADIUS_KM
    return np.column_stack((delta_km @ east, delta_km @ north, delta_km @ up))


def local_km_to_lat_lon_alt(east_km: np.ndarray, north_km: np.ndarray, alt_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    east, north, up = local_basis(SANYA_LAT_DEG, SANYA_LON_DEG)
    local_xyz = (
        unit_xyz(SANYA_LAT_DEG, SANYA_LON_DEG)[None, None, :]
        + (east_km[:, :, None] * east + north_km[:, :, None] * north + alt_km[:, :, None] * up)
        / EARTH_RADIUS_KM
    )
    local_xyz = local_xyz / np.linalg.norm(local_xyz, axis=2)[:, :, None]
    lat = np.rad2deg(np.arcsin(local_xyz[:, :, 2]))
    lon = np.rad2deg(np.arctan2(local_xyz[:, :, 1], local_xyz[:, :, 0]))
    return lat, lon, alt_km


def sample_earth_texture(texture: Image.Image, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    tex = np.asarray(texture.convert("RGB")) / 255.0
    h, w, _ = tex.shape
    lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
    col = np.clip(((lon_wrapped + 180.0) / 360.0 * (w - 1)).astype(int), 0, w - 1)
    row = np.clip(((90.0 - lat_deg) / 180.0 * (h - 1)).astype(int), 0, h - 1)
    rgb = tex[row, col]
    shade = 0.72 + 0.28 * np.clip((lat_deg - lat_deg.min()) / (lat_deg.max() - lat_deg.min()), 0, 1)
    return np.clip(rgb * shade[:, :, None], 0, 1)


def beam_points(
    lat_deg: float,
    lon_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    max_alt_km: float = 1200.0,
    samples: int = 80,
) -> np.ndarray:
    """Straight local beam in Earth-radius units."""
    east, north, up = local_basis(lat_deg, lon_deg)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    direction = np.sin(el) * up + np.cos(el) * (np.cos(az) * north + np.sin(az) * east)
    direction = direction / np.linalg.norm(direction)

    p0 = up
    alt = np.linspace(0.0, max_alt_km / EARTH_RADIUS_KM, samples)
    return p0[None, :] + alt[:, None] * direction[None, :]


def render() -> None:
    download_texture()
    OUT_DIR.mkdir(exist_ok=True)

    texture = Image.open(TEXTURE_PATH).convert("RGB")

    east_km = np.linspace(-1650, 2450, 460)
    north_km = np.linspace(-1450, 2200, 360)
    east2, north2 = np.meshgrid(east_km, north_km)
    ground_z = -((east2**2 + north2**2) / (2.0 * EARTH_RADIUS_KM))
    lat2, lon2, _ = local_km_to_lat_lon_alt(east2, north2, np.zeros_like(east2))
    ground_rgba = np.dstack((sample_earth_texture(texture, lat2, lon2), np.full(east2.shape, 0.98)))

    fig = plt.figure(figsize=(18, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.set_position([0.00, 0.04, 1.00, 0.93])
    ax.plot_surface(
        east2,
        north2,
        ground_z,
        rstride=1,
        cstride=1,
        facecolors=ground_rgba,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    field_line_offsets_km = np.r_[
        np.linspace(-1250, -250, 5),
        np.linspace(0, 2400, 13),
    ]
    for seed_lat, seed_lon in magnetic_meridian_seed_points(field_line_offsets_km):
        field_line = trace_igrf_field_line(seed_lat, seed_lon, alt_km=120.0)
        local_line = xyz_to_local_km(field_line)
        keep = (
            (local_line[:, 0] > east_km.min())
            & (local_line[:, 0] < east_km.max())
            & (local_line[:, 1] > north_km.min())
            & (local_line[:, 1] < north_km.max())
            & (local_line[:, 2] > -40)
            & (local_line[:, 2] < 1600)
        )
        local_line = local_line[keep]
        if local_line.shape[0] < 2:
            continue
        ax.plot(
            local_line[:, 0],
            local_line[:, 1],
            local_line[:, 2],
            color="#ff4d1f",
            linewidth=2.2,
            alpha=0.82,
        )

    ax.scatter([0], [0], [15], s=95, c="#ff3b30", edgecolors="white", linewidths=1.1, depthshade=False)
    ax.text(35, -10, 30, "Sanya ISR", color="white", fontsize=12)

    be, bn, _ = ppigrf.igrf(SANYA_LON_DEG, SANYA_LAT_DEG, 0.0, IGRF_DATE)
    meridian = np.array([float(be[0]), float(bn[0])])
    meridian = meridian / np.linalg.norm(meridian)
    s = np.linspace(-1500, 1500, 280)
    mx = s * meridian[0]
    my = s * meridian[1]
    mz = 55.0 - ((mx**2 + my**2) / (2.0 * EARTH_RADIUS_KM))
    meridian_points = np.column_stack((mx, my, mz))
    ax.plot(mx, my, mz, color="#ffd84d", linewidth=2.4, alpha=0.92)
    ax.text(mx[-1] - 360, my[-1] + 40, mz[-1] + 35, "magnetic meridian", color="#ffd84d", fontsize=13)

    scale_x = -1320.0
    scale_y = -1120.0
    ax.plot(
        [scale_x, scale_x],
        [scale_y, scale_y],
        [100, 1000],
        color="white",
        linewidth=1.6,
        alpha=0.85,
    )
    for altitude in range(100, 1001, 100):
        tick_len = 42.0 if altitude % 500 else 70.0
        ax.plot(
            [scale_x, scale_x + tick_len],
            [scale_y, scale_y],
            [altitude, altitude],
            color="white",
            linewidth=1.1,
            alpha=0.8,
        )
        if altitude % 200 == 0 or altitude in (100, 1000):
            ax.text(
                scale_x - 155,
                scale_y,
                altitude,
                f"{altitude} km",
                color="white",
                fontsize=9,
                alpha=0.82,
                ha="right",
                va="center",
            )

    ax.set_xlim(-1450, 1750)
    ax.set_ylim(-1250, 2050)
    ax.set_zlim(-130, 1600)
    ax.set_box_aspect((2.8, 2.1, 1.25))
    ax.set_axis_off()

    # View from east looking west toward Sanya; magnetic meridian lies left-right.
    ax.view_init(elev=8, azim=0, roll=0)
    ax.dist = 5.4
    fig.canvas.draw()
    add_projected_line(fig, ax, meridian_points, color="#ffd84d", linewidth=2.4, alpha=0.95)

    sanya_xy = project_points_to_figure(fig, ax, np.array([[0.0, 0.0, 80.0]]))[0]
    fig.add_artist(
        Line2D(
            [sanya_xy[0]],
            [sanya_xy[1]],
            marker="o",
            markersize=8,
            markerfacecolor="#ff3b30",
            markeredgecolor="white",
            linestyle="None",
            transform=fig.transFigure,
            zorder=1001,
        )
    )
    fig.text(
        sanya_xy[0] + 0.008,
        sanya_xy[1] + 0.004,
        "Sanya",
        color="white",
        fontsize=12,
        transform=fig.transFigure,
        zorder=1001,
    )

    fig.text(
        0.02,
        0.025,
        "Earth texture: NASA Blue Marble Next Generation / Earth Observatory. "
        f"Magnetic meridian and field lines from IGRF14 via ppigrf, {IGRF_DATE.date()}. "
        "Script: render_sanya_isr_earth.py",
        color="0.75",
        fontsize=8,
    )
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    render()

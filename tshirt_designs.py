"""
Race t-shirt designs — Marathon du Mont-Blanc 23 km.

Standalone. Regenerates the shirt designs and the 2025 activity sunbursts:

    shirt_route           route line art (race route + ghosted training tracks)
    shirt_badge           stat badge (elevation profile as the mountain)
    shirt_stats           "the work" (weekly distance bars + headline totals)
    sunburst_2025         year wheel — 12 months, activity types stacked outward
    sunburst_2025_donut   the same, with an empty centre

Data sources, in order of preference:
  * strava_activities.json / strava_streams.json  — caches written by the
    main StravaAnalysis notebook (used if present)
  * the Strava API                                — fallback, needs
    strava_tokens.json in this folder
  * the official Google My Map                    — fetched for the course

Run with:   python tshirt_designs.py
"""

import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# --- config -----------------------------------------------------------------
FOLDER = "/Users/lois/Documents/Jupyter/Strava"
COURSE_MID = "1Yb5fI78r0UNk6m2eqKVTRUA46EWC1jvz"   # official 23 km du Mont-Blanc map
DEEP_START = pd.Timestamp("2026-03-01")            # training window for the shirts
RACE_RELEVANT = ["Run", "TrailRun", "Hike", "Walk"]

SHIRT_INK = "#f2ede1"     # warm off-white
SHIRT_PAPER = "#16161c"   # near-black
SHIRT_ACCENT = "#e4513f"  # alpine red

# per-activity-type colours for the sunburst
TYPE_COLORS = {
    "Run": "#e4513f", "Hike": "#22c55e", "Walk": "#a3b18a",
    "NordicSki": "#4c78a8", "AlpineSki": "#7fc7ff", "BackcountrySki": "#14b8a6",
    "Ride": "#a855f7", "RockClimbing": "#d98c5f", "Workout": "#8a8a8a",
}

# folder layout (matches the notebook)
IMAGES_DIR = os.path.join(FOLDER, "Images")
STRAVA_STREAMS_DIR = os.path.join(FOLDER, "StravaStreams")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STRAVA_STREAMS_DIR, exist_ok=True)

TOKEN_FILE = os.path.join(FOLDER, "strava_tokens.json")
ACT_CACHE = os.path.join(STRAVA_STREAMS_DIR, "strava_activities.json")
STREAM_CACHE = os.path.join(STRAVA_STREAMS_DIR, "strava_streams.json")


# --- Strava data ------------------------------------------------------------
def strava_access_token() -> str:
    """Valid access token from strava_tokens.json, refreshing if expired."""
    with open(TOKEN_FILE) as fh:
        tok = json.load(fh)
    if tok["expires_at"] - time.time() < 300:
        resp = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": tok["client_id"],
                "client_secret": tok["client_secret"],
                "refresh_token": tok["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=20,
        ).json()
        tok.update(
            access_token=resp["access_token"],
            refresh_token=resp["refresh_token"],
            expires_at=resp["expires_at"],
        )
        with open(TOKEN_FILE, "w") as fh:
            json.dump(tok, fh, indent=2)
    return tok["access_token"]


def load_activities() -> pd.DataFrame:
    """All Strava activity summaries — from cache, or fetched via the API."""
    if os.path.exists(ACT_CACHE):
        with open(ACT_CACHE) as fh:
            raw = json.load(fh)
    else:
        headers = {"Authorization": f"Bearer {strava_access_token()}"}
        raw, page = [], 1
        while True:
            batch = requests.get(
                "https://www.strava.com/api/v3/athlete/activities",
                headers=headers,
                params={"per_page": 200, "page": page},
                timeout=30,
            ).json()
            if not batch:
                break
            raw += batch
            page += 1
        with open(ACT_CACHE, "w") as fh:
            json.dump(raw, fh)

    df = pd.DataFrame(raw)
    df["start"] = pd.to_datetime(df["start_date"], utc=True)
    df["start_local"] = pd.to_datetime(df["start_date_local"]).dt.tz_localize(None)
    df["dist_km"] = df["distance"] / 1000
    return df.sort_values("start").reset_index(drop=True)


def load_streams(activity_ids) -> dict:
    """Per-activity streams — from cache, or fetched (latlng only) via the API."""
    if os.path.exists(STREAM_CACHE):
        with open(STREAM_CACHE) as fh:
            return json.load(fh)
    headers = {"Authorization": f"Bearer {strava_access_token()}"}
    out = {}
    for aid in activity_ids:
        resp = requests.get(
            f"https://www.strava.com/api/v3/activities/{aid}/streams",
            headers=headers,
            params={"keys": "latlng", "key_by_type": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            out[str(aid)] = {k: v["data"] for k, v in resp.json().items()}
    with open(STREAM_CACHE, "w") as fh:
        json.dump(out, fh)
    return out


def build_records(streams_raw: dict) -> dict:
    """{activity_id: DataFrame of GPS points} — just what the route shirt needs."""
    records = {}
    for aid, s in streams_raw.items():
        if "latlng" not in s or not s["latlng"]:
            continue
        ll = pd.DataFrame(s["latlng"],
                          columns=["position_lat_deg", "position_long_deg"])
        records[aid] = ll
    return records


def build_act(strava: pd.DataFrame) -> pd.DataFrame:
    """Per-activity table for the training window (since DEEP_START)."""
    deep = strava[strava["start_local"] >= DEEP_START].copy()
    act = pd.DataFrame({
        "type": deep["type"].values,
        "date": deep["start_local"].values,
        "dist_km": deep["dist_km"].values,
        "ascent_strava_m": deep["total_elevation_gain"].values,
        "moving_min": deep["moving_time"].values / 60,
    })
    act["week"] = pd.to_datetime(act["date"]).dt.to_period("W-SUN").dt.start_time
    return act


# --- the race course --------------------------------------------------------
def load_course() -> pd.DataFrame:
    """Race route from the official Google My Map: lat, lon, ele, dist_km."""
    kml = requests.get(
        f"https://www.google.com/maps/d/kml?mid={COURSE_MID}&forcekml=1",
        timeout=20,
    ).text
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.fromstring(kml)
    line = next(
        pm.find(".//k:LineString/k:coordinates", ns)
        for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark")
        if pm.find(".//k:LineString/k:coordinates", ns) is not None
    )
    pts = np.array([list(map(float, c.split(","))) for c in line.text.split()])
    course = pd.DataFrame({"lon": pts[:, 0], "lat": pts[:, 1], "ele": pts[:, 2]})

    lat, lon = course["lat"].values, course["lon"].values
    radius = 6371000.0
    dlat = np.radians(np.diff(lat))
    dlon = np.radians(np.diff(lon))
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat[:-1])) * np.cos(np.radians(lat[1:]))
         * np.sin(dlon / 2) ** 2)
    seg = 2 * radius * np.arcsin(np.sqrt(a))
    course["dist_km"] = np.concatenate([[0], np.cumsum(seg)]) / 1000
    return course


# --- the three shirts -------------------------------------------------------
def make_route_shirt(course: pd.DataFrame, records: dict) -> None:
    """Route line art — race route as one bold stroke, training tracks behind."""
    lat0 = np.radians(course["lat"].mean())

    def project(lat, lon):
        return np.asarray(lon) * np.cos(lat0), np.asarray(lat)

    fig = plt.figure(figsize=(9, 11))
    fig.patch.set_facecolor(SHIRT_PAPER)
    ax = fig.add_axes([0.20, 0.33, 0.60, 0.63])
    ax.set_facecolor(SHIRT_PAPER)
    ax.axis("off")
    ax.set_aspect("equal")

    # race route first, to fix the frame on it
    rx, ry = project(course["lat"], course["lon"])
    pad = 0.12
    xw, yh = rx.max() - rx.min(), ry.max() - ry.min()
    ax.set_xlim(rx.min() - xw * pad, rx.max() + xw * pad)
    ax.set_ylim(ry.min() - yh * pad, ry.max() + yh * pad)

    # ghosted training routes (clipped to the race frame)
    for rec in records.values():
        if "position_lat_deg" not in rec.columns:
            continue
        la = rec["position_lat_deg"].dropna()
        lo = rec["position_long_deg"].dropna()
        if len(la) < 10:
            continue
        gx, gy = project(la, lo)
        ax.plot(gx, gy, color=SHIRT_INK, lw=0.7, alpha=0.16,
                solid_capstyle="round", clip_on=True)

    # the race route — hero stroke (drawn on top)
    ax.plot(rx, ry, color=SHIRT_INK, lw=3.6, zorder=5,
            solid_capstyle="round", solid_joinstyle="round")
    ax.scatter([rx[0]], [ry[0]], s=110, color=SHIRT_ACCENT, zorder=6,
               edgecolor=SHIRT_PAPER, linewidth=1.8)
    ax.scatter([rx[-1]], [ry[-1]], s=110, color=SHIRT_ACCENT, zorder=6,
               edgecolor=SHIRT_PAPER, linewidth=1.8, marker="s")

    # typography
    fig.text(0.5, 0.245, "23 KM DU MONT-BLANC", color=SHIRT_INK,
             fontsize=31, fontweight="bold", ha="center")
    fig.text(0.5, 0.196, "M O N T R O C   →   F L É G È R E",
             color=SHIRT_ACCENT, fontsize=12.5, ha="center", fontweight="bold")
    fig.add_artist(Line2D([0.34, 0.66], [0.158, 0.158],
                          color=SHIRT_INK, lw=1, alpha=0.45))
    fig.text(0.5, 0.118, "+1477 M D+        28 . 06 . 2026",
             color=SHIRT_INK, fontsize=12.5, ha="center", alpha=0.85)

    fig.savefig(os.path.join(IMAGES_DIR, "shirt_route.svg"), facecolor=SHIRT_PAPER)
    fig.savefig(os.path.join(IMAGES_DIR, "shirt_route.png"), dpi=200,
                facecolor=SHIRT_PAPER)
    plt.close(fig)


def make_badge_shirt(course: pd.DataFrame) -> None:
    """Stat badge — the real elevation profile as the mountain, numbers big."""
    fig = plt.figure(figsize=(9, 9.6))
    fig.patch.set_facecolor(SHIRT_PAPER)

    # double frame
    fig.add_artist(Rectangle((0.055, 0.05), 0.89, 0.90, fill=False,
                             edgecolor=SHIRT_INK, lw=2.2))
    fig.add_artist(Rectangle((0.075, 0.068), 0.85, 0.864, fill=False,
                             edgecolor=SHIRT_INK, lw=0.8, alpha=0.5))

    # top label + title
    fig.text(0.5, 0.875, "C H A M O N I X  —  M O N T - B L A N C",
             color=SHIRT_ACCENT, fontsize=12, ha="center", fontweight="bold")
    fig.text(0.5, 0.80, "23 KM", color=SHIRT_INK, fontsize=58,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.745, "M A R A T H O N   D U   M O N T - B L A N C",
             color=SHIRT_INK, fontsize=12.5, ha="center", alpha=0.8)

    # elevation profile as the mountain silhouette
    axp = fig.add_axes([0.135, 0.45, 0.73, 0.21])
    axp.fill_between(course["dist_km"], course["ele"], course["ele"].min(),
                     color=SHIRT_INK, alpha=0.95)
    axp.plot(course["dist_km"], course["ele"], color=SHIRT_INK, lw=1)
    axp.set_xlim(0, course["dist_km"].max())
    axp.set_ylim(course["ele"].min(), course["ele"].max() * 1.02)
    axp.axis("off")
    fig.text(0.135, 0.43, "MONTROC", color=SHIRT_INK, fontsize=9, alpha=0.7)
    fig.text(0.865, 0.43, "FLÉGÈRE", color=SHIRT_INK, fontsize=9,
             alpha=0.7, ha="right")

    # the headline number
    fig.text(0.5, 0.33, "+1477", color=SHIRT_ACCENT, fontsize=66,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.275, "M E T R E S   O F   C L I M B I N G",
             color=SHIRT_INK, fontsize=12.5, ha="center", alpha=0.8)

    # divider + date
    fig.add_artist(Line2D([0.30, 0.70], [0.205, 0.205],
                          color=SHIRT_INK, lw=1, alpha=0.45))
    fig.text(0.5, 0.135, "28 . 06 . 2026", color=SHIRT_INK, fontsize=22,
             fontweight="bold", ha="center")

    fig.savefig(os.path.join(IMAGES_DIR, "shirt_badge.svg"), facecolor=SHIRT_PAPER)
    fig.savefig(os.path.join(IMAGES_DIR, "shirt_badge.png"), dpi=200,
                facecolor=SHIRT_PAPER)
    plt.close(fig)


def make_stats_shirt(act: pd.DataFrame) -> None:
    """The work — weekly distance bars + the headline training totals."""
    rel = act[act["type"].isin(RACE_RELEVANT)]
    km = rel["dist_km"].sum()
    vert = rel["ascent_strava_m"].sum()
    hrs = rel["moving_min"].sum() / 60
    n = len(rel)

    # weekly distance, gaps filled so the bars are continuous
    wk = rel.groupby("week")["dist_km"].sum()
    wk = wk.reindex(
        pd.date_range(wk.index.min(), wk.index.max(), freq="7D"), fill_value=0)

    fig = plt.figure(figsize=(9, 11))
    fig.patch.set_facecolor(SHIRT_PAPER)

    # title
    fig.text(0.5, 0.90, "M A R A T H O N   D U   M O N T - B L A N C",
             color=SHIRT_ACCENT, fontsize=12, ha="center", fontweight="bold")
    fig.text(0.5, 0.825, "THE WORK", color=SHIRT_INK, fontsize=66,
             fontweight="bold", ha="center")

    # weekly bars — the build
    ax = fig.add_axes([0.13, 0.45, 0.74, 0.28])
    ax.set_facecolor(SHIRT_PAPER)
    bar_colors = [SHIRT_ACCENT if v >= wk.max() else SHIRT_INK
                  for v in wk.values]
    ax.bar(range(len(wk)), wk.values, color=bar_colors, width=0.74)
    ax.axhline(0, color=SHIRT_INK, lw=1.4)
    ax.set_ylim(0, wk.max() * 1.18)
    ax.set_xlim(-0.8, len(wk) + 2.6)
    ax.axis("off")
    ax.annotate("", xy=(len(wk) + 1.7, wk.max() * 0.45),
                xytext=(len(wk) - 0.3, wk.max() * 0.45),
                arrowprops=dict(arrowstyle="-|>", color=SHIRT_ACCENT, lw=2.2))
    ax.text(len(wk) + 1.9, wk.max() * 0.45, "RACE", color=SHIRT_ACCENT,
            fontsize=11, fontweight="bold", va="center")
    fig.text(0.5, 0.415,
             "W E E K L Y   K I L O M E T R E S   ·   S I N C E   1   M A R C H",
             color=SHIRT_INK, fontsize=9.5, ha="center", alpha=0.6)

    # headline stats — three columns
    stats = [(f"{km:.0f}", "KILOMETRES"),
             (f"{vert:,.0f}", "METRES  D+"),
             (f"{hrs:.0f}", "HOURS")]
    for sx, (big, lab) in zip([0.27, 0.5, 0.73], stats):
        fig.text(sx, 0.305, big, color=SHIRT_INK, fontsize=44,
                 fontweight="bold", ha="center")
        fig.text(sx, 0.258, lab, color=SHIRT_ACCENT, fontsize=10.5,
                 fontweight="bold", ha="center")

    # divider + footer
    fig.add_artist(Line2D([0.30, 0.70], [0.205, 0.205],
                          color=SHIRT_INK, lw=1, alpha=0.45))
    fig.text(0.5, 0.150, f"{n} SESSIONS   →   28 . 06 . 2026",
             color=SHIRT_INK, fontsize=16, fontweight="bold", ha="center")

    fig.savefig(os.path.join(IMAGES_DIR, "shirt_stats.svg"), facecolor=SHIRT_PAPER)
    fig.savefig(os.path.join(IMAGES_DIR, "shirt_stats.png"), dpi=200,
                facecolor=SHIRT_PAPER)
    plt.close(fig)


# --- 2025 sunburst ----------------------------------------------------------
def make_sunburst(strava: pd.DataFrame, inner_hole: float = 0.0,
                  filename: str = "sunburst_2025") -> None:
    """2025 activity wheel — 12 months around the circle, activity types
    stacked outward; radial length = number of activities that month.
    inner_hole > 0 leaves an empty centre (the donut variant)."""
    y25 = strava[strava["start_local"].dt.year == 2025].copy()
    y25["month"] = y25["start_local"].dt.month
    ct = (y25.groupby(["month", "type"]).size()
          .unstack(fill_value=0).reindex(range(1, 13), fill_value=0))
    order = ct.sum().sort_values(ascending=False).index.tolist()  # biggest in centre

    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    width = 2 * np.pi / 12 * 0.92

    fig = plt.figure(figsize=(10, 10.5))
    fig.patch.set_facecolor("#16161c")
    ax = fig.add_subplot(projection="polar")
    ax.set_facecolor("#16161c")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bottom = np.full(12, float(inner_hole))
    for typ in order:
        vals = ct[typ].values
        ax.bar(theta, vals, width=width, bottom=bottom,
               color=TYPE_COLORS.get(typ, "#888888"),
               edgecolor="#16161c", linewidth=1.3, label=typ)
        bottom += vals

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(theta)
    ax.set_xticklabels(months, color="#f2ede1", fontsize=13, fontweight="bold")
    rmax = int(bottom.max() - inner_hole)
    ax.set_yticks([inner_hole + v for v in range(5, rmax + 5, 5)])
    ax.set_yticklabels([str(v) for v in range(5, rmax + 5, 5)],
                       color="#f2ede1", fontsize=8, alpha=0.45)
    ax.set_ylim(0, bottom.max() * 1.06)
    ax.set_rlabel_position(15)
    ax.grid(color="#f2ede1", alpha=0.12)
    ax.spines["polar"].set_visible(False)

    for t, tot in zip(theta, bottom):
        ax.text(t, tot + 0.9, str(int(tot - inner_hole)), color="#f2ede1",
                fontsize=9, ha="center", va="center", alpha=0.8)

    fig.suptitle("2025  ·  A YEAR OF ACTIVITY", color="#f2ede1",
                 fontsize=22, fontweight="bold", y=0.97)
    fig.text(0.5, 0.925,
             f"{len(y25)} activities  ·  radial length = how many that month",
             color="#f2ede1", fontsize=11, ha="center", alpha=0.6)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                    ncol=5, frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color("#f2ede1")

    fig.savefig(os.path.join(IMAGES_DIR, f"{filename}.png"), dpi=170,
                facecolor="#16161c", bbox_inches="tight")
    plt.close(fig)


# --- run --------------------------------------------------------------------
def main() -> None:
    print("Loading Strava activities...")
    strava = load_activities()
    act = build_act(strava)

    print("Loading GPS streams...")
    deep_ids = strava[strava["start_local"] >= DEEP_START]["id"].tolist()
    records = build_records(load_streams(deep_ids))

    print("Fetching the race course...")
    course = load_course()

    print("Drawing shirts...")
    make_route_shirt(course, records)
    make_badge_shirt(course)
    make_stats_shirt(act)

    print("Drawing 2025 sunbursts (full Strava history)...")
    make_sunburst(strava, inner_hole=0.0, filename="sunburst_2025")
    make_sunburst(strava, inner_hole=4.0, filename="sunburst_2025_donut")

    for name in ("shirt_route", "shirt_badge", "shirt_stats"):
        print(f"  {name}.svg / {name}.png")
    for name in ("sunburst_2025", "sunburst_2025_donut"):
        print(f"  {name}.png")
    print("Done.")


if __name__ == "__main__":
    main()

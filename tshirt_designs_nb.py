import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
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

    return ET, Line2D, Rectangle, json, np, os, pd, plt, requests, time


@app.cell
def _(os, pd):
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
    return (
        ACT_CACHE,
        COURSE_MID,
        DEEP_START,
        IMAGES_DIR,
        RACE_RELEVANT,
        SHIRT_ACCENT,
        SHIRT_INK,
        SHIRT_PAPER,
        STRAVA_STREAMS_DIR,
        STREAM_CACHE,
        TOKEN_FILE,
        TYPE_COLORS,
    )


@app.cell
def _(
    ACT_CACHE,
    DEEP_START,
    STRAVA_STREAMS_DIR,
    STREAM_CACHE,
    TOKEN_FILE,
    json,
    os,
    pd,
    requests,
    time,
):
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


    def load_streams_2025(strava: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """GPS latlng streams for 2025 runs & hikes (year-in-routes grid).

        Cached to StravaStreams/strava_streams_2025.json. Format is flatter than
        the since-March cache: ``{activity_id: [[lat, lon], ...]}``.

        Returns (routes_df, streams_dict). routes_df only contains activities
        we actually have a track for, sorted by date.
        """
        cache = os.path.join(STRAVA_STREAMS_DIR, "strava_streams_2025.json")
        y25 = (strava[(strava["start_local"].dt.year == 2025)
                      & (strava["type"].isin(["Run", "TrailRun", "Hike"]))]
               .copy())
        if os.path.exists(cache):
            with open(cache) as fh:
                raw = json.load(fh)
        else:
            headers = {"Authorization": f"Bearer {strava_access_token()}"}
            raw = {}
            for aid in y25["id"]:
                resp = requests.get(
                    f"https://www.strava.com/api/v3/activities/{aid}/streams",
                    headers=headers,
                    params={"keys": "latlng", "key_by_type": "true"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    latlng = resp.json().get("latlng", {}).get("data")
                    if latlng:
                        raw[str(aid)] = latlng
            with open(cache, "w") as fh:
                json.dump(raw, fh)

        routes = (y25[y25["id"].astype(str).isin(raw)]
                  .sort_values("start_local").reset_index(drop=True))
        return routes, raw

    return (
        build_act,
        build_records,
        load_activities,
        load_streams,
        load_streams_2025,
    )


@app.cell
def _(COURSE_MID, ET, np, pd, requests):
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

    return (load_course,)


@app.cell
def _(
    IMAGES_DIR,
    Line2D,
    SHIRT_ACCENT,
    SHIRT_INK,
    SHIRT_PAPER,
    np,
    os,
    pd,
    plt,
):
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

    return


@app.cell
def _(
    IMAGES_DIR,
    Line2D,
    Rectangle,
    SHIRT_ACCENT,
    SHIRT_INK,
    SHIRT_PAPER,
    os,
    pd,
    plt,
):
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

    return


@app.cell
def _(
    IMAGES_DIR,
    Line2D,
    RACE_RELEVANT,
    SHIRT_ACCENT,
    SHIRT_INK,
    SHIRT_PAPER,
    os,
    pd,
    plt,
):
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

    return


@app.cell
def _(IMAGES_DIR, TYPE_COLORS, np, os, pd, plt):
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

    return


@app.cell
def _(np, pd, plt):
    def make_routes_grid_2025(routes: pd.DataFrame, streams: dict) -> None:
        """Every 2025 run & hike rendered as its own little route shape, in a
        chronological grid coloured Jan -> Dec. Saved to Images/routes_2025.png."""
        n = len(routes)
        ncol = 10
        nrow = int(np.ceil(n / ncol))

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.45, nrow * 1.55))
        fig.patch.set_facecolor("#16161c")
        axes = np.array(axes).flatten()

        for i, r in enumerate(routes.itertuples()):
            ax = axes[i]
            ax.set_facecolor("#16161c")
            ax.axis("off")
            ll = np.array(streams[str(r.id)])
            lat, lon = ll[:, 0], ll[:, 1]
            x = lon * np.cos(np.radians(lat.mean()))
            color = plt.cm.turbo(r.start_local.dayofyear / 366)
            ax.plot(x, lat, color=color, lw=1.2, solid_capstyle="round")
            ax.set_aspect("equal")
            ax.set_title(f"{r.start_local:%-d %b} \u00b7 {r.dist_km:.0f}k",
                         color="#f2ede1", fontsize=6.5, pad=2)

        for j in range(n, len(axes)):
            axes[j].axis("off")
            axes[j].set_facecolor("#16161c")

        fig.suptitle(f"2025  \u00b7  {n} RUNS & HIKES", color="#f2ede1",
                     fontsize=22, fontweight="bold", y=0.985)
        fig.text(0.5, 0.945,
                 "each shape is one activity   \u00b7   colour runs January \u2192 December",
                 color="#f2ede1", fontsize=10, ha="center", alpha=0.65)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01,
                            wspace=0.15, hspace=0.55)
       # fig.savefig(os.path.join(IMAGES_DIR, "routes_2025.png"), dpi=170,
        #            facecolor="#16161c")
       # plt.close(fig)
        return fig


    return (make_routes_grid_2025,)


@app.cell
def _(IMAGES_DIR, np, os, pd, plt, requests):
    # --- 2D route over satellite imagery ----------------------------------------
    from PIL import Image
    from io import BytesIO
    from matplotlib.transforms import Affine2D

    ESRI_EXPORT_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/export")

    def fetch_aerial(west: float, south: float, east: float, north: float,
                     size_px: int = 1024) -> Image.Image:
        """ESRI World Imagery PNG for a lat/lon bbox (no API key needed)."""
        r = requests.get(ESRI_EXPORT_URL, params={
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": 4326, "imageSR": 4326,
            "size": f"{size_px},{size_px}", "format": "png", "f": "image",
        }, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")


    def make_course_aerial(course: pd.DataFrame, *,
                           pad: float = 0.006, size_px: int = 1024,
                           linewidth: float = 3.2,
                           rotation_deg: float = 0.0, save: bool = False):
        """2D route over satellite imagery, route coloured by elevation.

        rotation_deg rotates the whole map COUNTER-clockwise. So:
            0     north up           (default)
           -45    NORTH-WEST up      (NW corner of the bbox points up)
           +45    NORTH-EAST up
        A small compass-rose inset shows where north ends up.
        """
        west  = course["lon"].min() - pad
        east  = course["lon"].max() + pad
        south = course["lat"].min() - pad
        north = course["lat"].max() + pad

        img = np.asarray(fetch_aerial(west, south, east, north, size_px))

        # work in local metres so rotation is geographically correct (1 deg
        # lat \u2248 111 km; 1 deg lon depends on latitude)
        lat_ref = course["lat"].mean()
        lon_ref = course["lon"].mean()
        M_LAT = 111000.0
        M_LON = 111000.0 * np.cos(np.radians(lat_ref))

        def to_xy(lon, lat):
            return (np.asarray(lon) - lon_ref) * M_LON, \
                   (np.asarray(lat) - lat_ref) * M_LAT

        fig = plt.figure(figsize=(11, 11))
        fig.patch.set_facecolor("#16161c")
        ax = fig.add_axes([0.05, 0.05, 0.78, 0.88])
        ax.set_facecolor("#16161c")
        rot = Affine2D().rotate_deg(rotation_deg) + ax.transData

        # aerial floor — extent in pre-rotation metres, transform applies the spin
        w_m, s_m = to_xy(west, south)
        e_m, n_m = to_xy(east, north)
        w_m, s_m, e_m, n_m = float(w_m), float(s_m), float(e_m), float(n_m)
        ax.imshow(img, extent=(w_m, e_m, s_m, n_m), origin="upper",
                  interpolation="bilinear", transform=rot, zorder=1)

        # route coloured by elevation
        lats = course["lat"].values
        lons = course["lon"].values
        eles = course["ele"].values
        xr, yr = to_xy(lons, lats)
        norm = (eles - eles.min()) / (eles.max() - eles.min())
        for i in range(len(xr) - 1):
            ax.plot(xr[i:i+2], yr[i:i+2],
                    color=plt.cm.plasma(norm[i]),
                    lw=linewidth, solid_capstyle="round",
                    solid_joinstyle="round", zorder=4, transform=rot)

        ax.scatter([xr[0]],  [yr[0]],  s=220, color="#22c55e",
                   edgecolor="#16161c", linewidth=2, zorder=10,
                   label="Start (Montroc)", transform=rot)
        ax.scatter([xr[-1]], [yr[-1]], s=220, color="#e4513f", marker="s",
                   edgecolor="#16161c", linewidth=2, zorder=10,
                   label="Finish (Fl\u00e9g\u00e8re)", transform=rot)

        # auto-fit axis limits to the rotated bbox + margin
        a = np.radians(rotation_deg)
        ca, sa = np.cos(a), np.sin(a)
        cx = np.array([w_m, e_m, e_m, w_m])
        cy = np.array([s_m, s_m, n_m, n_m])
        rx = ca * cx - sa * cy
        ry = sa * cx + ca * cy
        m = (rx.max() - rx.min()) * 0.02
        ax.set_xlim(rx.min() - m, rx.max() + m)
        ax.set_ylim(ry.min() - m, ry.max() + m)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("23 km du Mont-Blanc \u2014 route over satellite imagery",
                     color="#f2ede1", fontsize=14, fontweight="bold", pad=12)

        leg = ax.legend(loc="upper left", frameon=False, fontsize=10)
        for t in leg.get_texts():
            t.set_color("#f2ede1")

        # compass-rose inset showing where north ends up after rotation
        comp = fig.add_axes([0.78, 0.84, 0.08, 0.08])
        comp.set_facecolor("#16161c")
        comp.set_xlim(-1.2, 1.2); comp.set_ylim(-1.2, 1.2)
        comp.set_aspect("equal"); comp.axis("off")
        n_dx, n_dy = -np.sin(a), np.cos(a)   # where (0,1) maps under the rotation
        comp.annotate("", xy=(n_dx, n_dy), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="#f2ede1", lw=2.2))
        comp.text(n_dx * 1.55, n_dy * 1.55, "N", color="#f2ede1",
                  fontsize=13, fontweight="bold", ha="center", va="center")

        # elevation colour bar
        cax = fig.add_axes([0.88, 0.22, 0.025, 0.55])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                   norm=plt.Normalize(eles.min(), eles.max()))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Elevation (m)", color="#f2ede1")
        cb.outline.set_edgecolor("#333")
        cb.ax.yaxis.set_tick_params(color="#f2ede1")
        plt.setp(cb.ax.get_yticklabels(), color="#f2ede1")

        if save:
            fig.savefig(os.path.join(IMAGES_DIR, "course_aerial.png"),
                        dpi=170, facecolor="#16161c", bbox_inches="tight")
        return fig

    return (make_course_aerial,)


@app.cell
def _(IMAGES_DIR, np, os, plt):
    # --- variants of the routes plot --------------------------------------------
    # Two new designs that reuse the same month-colour scheme as the grid:
    #   * make_routes_random      every route scattered randomly on one canvas
    #   * make_routes_spirograph  every route at a common centre, fanning out

    def projected_xy_centered(latlng):
        """latlng list/array -> (x, y) numpy arrays centred on the route's mean."""
        ll = np.asarray(latlng)
        lat, lon = ll[:, 0], ll[:, 1]
        lat0 = np.radians(lat.mean())
        x = (lon - lon.mean()) * np.cos(lat0)
        y = lat - lat.mean()
        return x, y


    def rotate2d(x, y, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return c * x - s * y, s * x + c * y


    def make_routes_random(routes, streams, *, seed=0,
                           canvas_radius=10.0, route_size=1.6, alpha=0.8,
                           save=False):
        """Every 2025 run & hike at a random position on one canvas, coloured by
        month, each normalised to roughly the same size and randomly rotated.
        Returns the fig so it renders inline; pass save=True to also write to disk."""
        rng = np.random.default_rng(seed)
        n = len(routes)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#16161c")
        ax.set_facecolor("#16161c")
        ax.set_aspect("equal")
        ax.axis("off")
        for r in routes.itertuples():
            x, y = projected_xy_centered(streams[str(r.id)])
            extent = max(x.max() - x.min(), y.max() - y.min())
            if extent > 0:
                x = x / extent * route_size
                y = y / extent * route_size
            x, y = rotate2d(x, y, rng.uniform(0, 2 * np.pi))
            cx, cy = rng.uniform(-canvas_radius, canvas_radius, size=2)
            color = plt.cm.turbo(r.start_local.dayofyear / 366)
            ax.plot(x + cx, y + cy, color=color, lw=1.1, alpha=alpha,
                    solid_capstyle="round")
        lim = canvas_radius + route_size
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        fig.suptitle(f"2025  \u00b7  {n} RUNS & HIKES",
                     color="#f2ede1", fontsize=20, fontweight="bold", y=0.96)
        if save:
            fig.savefig(os.path.join(IMAGES_DIR, "routes_2025_random.png"),
                        dpi=170, facecolor="#16161c", bbox_inches="tight")
        return fig


    def make_routes_spirograph(routes, streams, *, route_size=4.0, alpha=0.55,
                               save=False):
        """Every route at a common centre, each rotated by an evenly-spaced angle
        so they fan out like spirograph petals. Coloured by month, sorted by date."""
        n = len(routes)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#16161c")
        ax.set_facecolor("#16161c")
        ax.set_aspect("equal")
        ax.axis("off")
        for i, r in enumerate(routes.itertuples()):
            x, y = projected_xy_centered(streams[str(r.id)])
            extent = max(x.max() - x.min(), y.max() - y.min())
            if extent > 0:
                x = x / extent * route_size
                y = y / extent * route_size
            theta = 2 * np.pi * i / n
            x, y = rotate2d(x, y, theta)
            color = plt.cm.turbo(r.start_local.dayofyear / 366)
            ax.plot(x, y, color=color, lw=1.0, alpha=alpha, solid_capstyle="round")
        lim = route_size * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        fig.suptitle(f"2025  \u00b7  {n} ROUTES",
                     color="#f2ede1", fontsize=20, fontweight="bold", y=0.96)
        if save:
            fig.savefig(os.path.join(IMAGES_DIR, "routes_2025_spirograph.png"),
                        dpi=170, facecolor="#16161c", bbox_inches="tight")
        return fig

    return make_routes_random, make_routes_spirograph


@app.cell
def _(make_routes_grid_2025, routes_2025, streams_2025):
    # Render the 2025 routes grid -> Images/routes_2025.png
    make_routes_grid_2025(routes_2025, streams_2025)
    return


@app.cell
def _():
    return


@app.cell
def _(
    DEEP_START,
    build_act,
    build_records,
    load_activities,
    load_course,
    load_streams,
    load_streams_2025,
):
    # Load the data once — exposes `strava`, `act`, `records`, `course` to
    # every cell below. Uses the StravaStreams JSON caches if present, else
    # fetches via the Strava API (needs strava_tokens.json).
    strava = load_activities()
    act = build_act(strava)
    _deep_ids = strava[strava["start_local"] >= DEEP_START]["id"].tolist()
    records = build_records(load_streams(_deep_ids))
    course = load_course()
    routes_2025, streams_2025 = load_streams_2025(strava)
    print(f"{len(strava)} activities  ·  {len(records)} streams  ·  "
          f"course {course['dist_km'].iloc[-1]:.1f} km")
    return course, routes_2025, strava, streams_2025


@app.cell
def _(IMAGES_DIR, TYPE_COLORS, np, os, plt, strava):
    # === Sunburst — HOURS of activity (not count) ===
    # Same wheel as make_sunburst, but radial length = total moving hours per
    # month per type. Saved to Images/sunburst_2025_hours.png.
    _y25 = strava[strava["start_local"].dt.year == 2025].copy()
    _y25["hours"] = _y25["moving_time"] / 3600
    _y25["month"] = _y25["start_local"].dt.month
    _ct = (_y25.groupby(["month", "type"])["hours"].sum()
           .unstack(fill_value=0).reindex(range(1, 13), fill_value=0))
    _order = _ct.sum().sort_values(ascending=False).index.tolist()

    _theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    _width = 2 * np.pi / 12 * 0.92

    _fig = plt.figure(figsize=(10, 10.5))
    _fig.patch.set_facecolor("#16161c")
    _ax = _fig.add_subplot(projection="polar")
    _ax.set_facecolor("#16161c")
    _ax.set_theta_zero_location("N")
    _ax.set_theta_direction(-1)

    _bottom = np.zeros(12)
    for _typ in _order:
        _vals = _ct[_typ].values
        _ax.bar(_theta, _vals, width=_width, bottom=_bottom,
                color=TYPE_COLORS.get(_typ, "#888888"),
                edgecolor="#16161c", linewidth=1.3, label=_typ)
        _bottom += _vals

    _months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _ax.set_xticks(_theta)
    _ax.set_xticklabels(_months, color="#f2ede1", fontsize=13, fontweight="bold")

    _rmax = _bottom.max()
    _step = 5 if _rmax > 20 else 2
    _ticks = list(range(_step, int(_rmax) + _step, _step))
    _ax.set_yticks(_ticks)
    _ax.set_yticklabels([f"{v}h" for v in _ticks],
                        color="#f2ede1", fontsize=8, alpha=0.45)
    _ax.set_ylim(0, _rmax * 1.10)
    _ax.set_rlabel_position(15)
    _ax.grid(color="#f2ede1", alpha=0.12)
    _ax.spines["polar"].set_visible(False)

    for _t, _tot in zip(_theta, _bottom):
        _ax.text(_t, _tot + _rmax * 0.05, f"{_tot:.0f}h",
                 color="#f2ede1", fontsize=9, ha="center", va="center", alpha=0.85)

    _fig.suptitle("2025  \u00b7  HOURS OF ACTIVITY", color="#f2ede1",
                  fontsize=22, fontweight="bold", y=0.97)
    _fig.text(0.5, 0.925,
              f"{_y25['hours'].sum():.0f} total hours  \u00b7  "
              f"radial length = hours that month",
              color="#f2ede1", fontsize=11, ha="center", alpha=0.6)
    _leg = _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                      ncol=5, frameon=False, fontsize=10)
    for _txt in _leg.get_texts():
        _txt.set_color("#f2ede1")

    _fig.savefig(os.path.join(IMAGES_DIR, "sunburst_2025_hours.png"), dpi=170,
                 facecolor="#16161c", bbox_inches="tight")
    _fig
    return


@app.cell
def _(make_routes_random, routes_2025, streams_2025):
    # Render — every 2025 route, scattered randomly. Bump `seed` for a different
    # arrangement, or `save=True` to also write Images/routes_2025_random.png.
    make_routes_random(routes_2025, streams_2025, seed=0)
    return


@app.cell
def _(make_routes_spirograph, routes_2025, streams_2025):
    # Render — every 2025 route, fanning out from a common centre.
    make_routes_spirograph(routes_2025, streams_2025)
    return


@app.cell
def _(course, make_course_aerial):
    # Render with NorthWest pointing straight up (rotation_deg = -45).
    # Pass rotation_deg=0 for north-up, +45 for NE-up, etc.
    make_course_aerial(course, rotation_deg=135)
    return


if __name__ == "__main__":
    app.run()

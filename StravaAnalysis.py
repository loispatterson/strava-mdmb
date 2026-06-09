import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import glob
    import json
    import os
    import time
    import xml.etree.ElementTree as ET

    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import requests
    import srtm
    from garmin_fit_sdk import Decoder, Stream


    return (
        Decoder,
        ET,
        Stream,
        alt,
        glob,
        json,
        mo,
        np,
        os,
        pd,
        requests,
        srtm,
        time,
    )


@app.cell(hide_code=True)
def _(Decoder, Stream, glob, os, pd):
    # Decode every .fit file in the folder into raw FIT message dicts.
    FIT_FOLDER = "/Users/lois/Documents/Jupyter/Strava"
    FIT_FILES_DIR = os.path.join(FIT_FOLDER, "FIT files")
    IMAGES_DIR = os.path.join(FIT_FOLDER, "Images")
    STRAVA_STREAMS_DIR = os.path.join(FIT_FOLDER, "StravaStreams")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(STRAVA_STREAMS_DIR, exist_ok=True)
    RACE_DATE = pd.Timestamp("2026-06-28", tz="UTC")

    def load_fit(path: str) -> dict:
        messages, errors = Decoder(Stream.from_file(path)).read()
        if errors:
            print(f"{os.path.basename(path)}: {len(errors)} decode error(s)")
        return messages

    fit_files = sorted(glob.glob(os.path.join(FIT_FILES_DIR, "*.fit")))
    raw = {os.path.basename(p): load_fit(p) for p in fit_files}
    # (silenced "Loaded N FIT files" prints for a clean intro)
    return FIT_FOLDER, RACE_DATE, STRAVA_STREAMS_DIR, raw


@app.cell(hide_code=True)
def _(pd, raw):
    def fit_dict_to_dfs(messages: dict) -> dict[str, pd.DataFrame]:
        """Normalise each FIT message type (list of dicts) into a DataFrame."""
        dfs = {}
        for msg_type, records in messages.items():
            if not isinstance(records, list) or not records:
                continue
            df = pd.json_normalize(records, sep="_")
            df.insert(0, "message_type", str(msg_type))
            dfs[str(msg_type)] = df
        return dfs

    activities = {name: fit_dict_to_dfs(msgs) for name, msgs in raw.items()}
    return (activities,)


@app.cell(hide_code=True)
def _(activities, np, pd):
    # One row per activity: the session-level summary plus a clean date/label.
    def clean_label(filename: str) -> str:
        return (
            filename.removesuffix(".fit")
            .replace("_", " ")
            .strip()
            .capitalize()
        )

    def session_row(name: str, dfs: dict) -> dict:
        s = dfs["session_mesgs"].iloc[0]
        rec = dfs.get("record_mesgs")
        dist_km = s.get("total_distance", np.nan) / 1000
        elapsed_min = s.get("total_elapsed_time", np.nan) / 60
        moving_min = s.get("total_timer_time", np.nan) / 60
        ascent = s.get("total_ascent", np.nan)
        return {
            "activity": clean_label(name),
            "file": name,
            "date": pd.Timestamp(s["start_time"]),
            "sport": f"{s.get('sport','?')}/{s.get('sub_sport','?')}",
            "dist_km": dist_km,
            "elapsed_min": elapsed_min,
            "moving_min": moving_min,
            "ascent_watch_m": ascent,
            "descent_m": s.get("total_descent", np.nan),
            "pace_min_km": moving_min / dist_km if dist_km else np.nan,
            "avg_hr": s.get("avg_heart_rate", np.nan),
            "max_hr": s.get("max_heart_rate", np.nan),
            "avg_cadence": s.get("avg_running_cadence", s.get("avg_cadence", np.nan)),
            "total_calories": s.get("total_calories", np.nan),
            "n_records": 0 if rec is None else len(rec),
        }

    summary = (
        pd.DataFrame([session_row(n, d) for n, d in activities.items()])
        .sort_values("date")
        .reset_index(drop=True)
    )
    #summary
    return (summary,)


@app.cell(hide_code=True)
def _(activities, mo, np, pd, srtm):
    # --- Altitude correction -------------------------------------------------
    # The Venu 3\'s barometric altimeter over-reads vertical gain. We re-derive
    # elevation from each GPS point using the SRTM digital elevation model, then
    # recompute ascent on a fixed DISTANCE grid with smoothing — so the result
    # doesn\'t depend on how slowly you moved.
    #
    # ELEVATION_SMOOTH_M is the one knob: smaller = more gain, larger = less.
    # Calibrate it against Strava\'s corrected numbers once we have them.
    ELEVATION_SMOOTH_M = 250.0  # calibrated against Strava's corrected D+
    # (see the calibration cell below) — ~5% mean error, unbiased
    GRID_STEP_M = 10.0
    SEMICIRCLE = 180 / 2**31  # FIT lat/long unit -> degrees

    dem_data = srtm.get_data()

    def dem_altitude(rec: pd.DataFrame) -> pd.Series:
        """Ground elevation (m) from SRTM at each GPS point; voids interpolated."""
        lat = rec["position_lat"] * SEMICIRCLE
        lon = rec["position_long"] * SEMICIRCLE
        elev = [
            dem_data.get_elevation(la, lo)
            if np.isfinite(la) and np.isfinite(lo) else np.nan
            for la, lo in zip(lat, lon)
        ]
        return pd.Series(elev, index=rec.index, dtype="float64").interpolate(
            limit_direction="both"
        )

    def total_ascent(distance: pd.Series, elevation: pd.Series,
                     grid_step: float = GRID_STEP_M,
                     smooth_m: float = ELEVATION_SMOOTH_M) -> float:
        """Positive elevation gain, resampled to a fixed distance grid + smoothed."""
        m = np.isfinite(distance) & np.isfinite(elevation)
        d, e = np.asarray(distance[m]), np.asarray(elevation[m])
        if len(d) < 2:
            return np.nan
        grid = np.arange(d[0], d[-1], grid_step)
        eg = np.interp(grid, d, e)
        win = max(1, int(smooth_m / grid_step))
        eg = pd.Series(eg).rolling(win, center=True, min_periods=1).mean().values
        diffs = np.diff(eg)
        return float(diffs[diffs > 0].sum())

    dem_alt = {name: dem_altitude(dfs["record_mesgs"])
               for name, dfs in activities.items()}
    dem_ascent = pd.Series(
        {name: total_ascent(dfs["record_mesgs"]["distance"], dem_alt[name])
         for name, dfs in activities.items()},
        name="ascent_dem_m",
    )
    dem_ascent.round().astype(int)

    mo.output.clear()
    return ELEVATION_SMOOTH_M, dem_alt, dem_ascent, dem_data, total_ascent


@app.cell(hide_code=True)
def _(activities, dem_ascent, mo, np, pd, summary):
    # Aerobic efficiency + decoupling per activity.
    # Efficiency factor (EF) = speed / heart-rate while moving: higher = fitter.
    # Decoupling = how much EF fades from the 1st to 2nd half of the activity;
    # < 5% is a sign of a solid aerobic base (Friel\'s aerobic decoupling test).
    def efficiency_metrics(rec: pd.DataFrame) -> pd.Series:
        d = rec[["timestamp", "enhanced_speed", "heart_rate"]].dropna()
        # keep only genuinely-moving samples so stops don\'t skew the ratio
        d = d[d["enhanced_speed"] > 0.5]
        if len(d) < 60:
            return pd.Series({"ef": np.nan, "decoupling_pct": np.nan})
        ef = d["enhanced_speed"].mean() / d["heart_rate"].mean()
        mid = len(d) // 2
        h1, h2 = d.iloc[:mid], d.iloc[mid:]
        ef1 = h1["enhanced_speed"].mean() / h1["heart_rate"].mean()
        ef2 = h2["enhanced_speed"].mean() / h2["heart_rate"].mean()
        return pd.Series({"ef": ef, "decoupling_pct": (ef1 - ef2) / ef1 * 100})

    eff = pd.DataFrame(
        {name: efficiency_metrics(dfs["record_mesgs"]) for name, dfs in activities.items()}
    ).T
    eff.index.name = "file"

    # Assemble the master table. ascent_m uses the DEM-corrected value; the raw
    # barometric figure stays as ascent_watch_m for comparison.
    summary_full = (
        summary
        .merge(eff, on="file", how="left")
        .merge(dem_ascent.rename_axis("file").reset_index(), on="file", how="left")
    )
    summary_full["ascent_m"] = summary_full["ascent_dem_m"]
    summary_full["vert_per_km"] = summary_full["ascent_m"] / summary_full["dist_km"]

    summary_full[["activity", "date", "dist_km", "ascent_watch_m", "ascent_dem_m",
                  "vert_per_km", "pace_min_km", "avg_hr", "ef"]]

    mo.output.clear()

    return efficiency_metrics, summary_full


@app.cell(hide_code=True)
def _(alt, mo, summary_full):
    # "Am I getting fitter?" — terrain confounds raw pace & EF (a flat run always
    # looks "fitter" than a steep one), so we plot against terrain difficulty.
    # For a given vert/km, more recent activities sitting LOWER (faster) = fitter.
    _base = alt.Chart(summary_full).encode(
        x=alt.X("vert_per_km:Q", title="Climb per km (m/km) — terrain difficulty"),
        color=alt.Color("date:T", title="Date", scale=alt.Scale(scheme="viridis")),
    )
    pace_vs_terrain = (
        _base.mark_point(size=160, filled=True).encode(
            y=alt.Y("pace_min_km:Q", title="Pace (min/km)"),
            shape=alt.Shape("sport:N", title="Sport"),
            tooltip=["activity", "date:T", "dist_km", "ascent_m",
                     "pace_min_km", "avg_hr", "ef"],
        )
        + _base.mark_text(dy=-14, fontSize=9).encode(
            y="pace_min_km:Q", text="activity:N"
        )
    ).properties(height=260, width=560, title="Pace vs terrain difficulty (lower = faster)")

    ef_over_time = (
        alt.Chart(summary_full).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("ef:Q", title="Efficiency factor (speed / HR)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("sport:N", title="Sport"),
            tooltip=["activity", "date:T", "ef", "avg_hr", "pace_min_km"],
        ).properties(height=220, width=560,
                     title="Aerobic efficiency over time (higher = fitter, but terrain-sensitive)")
    )

    mo.vstack([pace_vs_terrain, ef_over_time])

    mo.output.clear()

    return


@app.cell(hide_code=True)
def _(alt, mo, summary_full):
    # Training load by week. NOTE: only the .fit files in this folder are
    # included — once more activities are pulled from Strava this fills out.
    _s = summary_full.copy()
    _s["week_start"] = (
        _s["date"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time
    )
    weekly = (
        _s.groupby("week_start")
        .agg(activities=("activity", "count"),
             dist_km=("dist_km", "sum"),
             ascent_m=("ascent_m", "sum"),
             hours=("moving_min", lambda x: x.sum() / 60))
        .round(1)
        .reset_index()
    )

    _dist = alt.Chart(weekly).mark_bar(size=40, color="#4c78a8").encode(
        x=alt.X("week_start:T", title="Week"),
        y=alt.Y("dist_km:Q", title="Distance (km)"),
        tooltip=["week_start:T", "activities", "dist_km", "ascent_m", "hours"],
    ).properties(height=200, width=560, title="Distance per week")

    _vert = alt.Chart(weekly).mark_bar(size=40, color="#e45756").encode(
        x=alt.X("week_start:T", title="Week"),
        y=alt.Y("ascent_m:Q", title="Climb (m D+)"),
    
        tooltip=["week_start:T", "activities", "ascent_m", "hours"],
    ).properties(height=200, width=560, title="Vertical gain per week")

    mo.vstack([weekly, mo.hstack([_dist, _vert], widths="equal")])

    mo.output.clear()

    return


@app.cell
def _(activities, alt, mo, pd, summary_full):
    # Time-in-HR-zone per activity, using the watch's REAL configured zones —
    # read straight from the .fit files' time_in_zone_mesgs (the Strava streams
    # don't carry zone config). Endurance training should be "polarised":
    # mostly easy (Z1-Z2), a little hard (Z4-Z5), not much grey-zone Z3.
    _tiz = activities["Afternoon_Run.fit"]["time_in_zone_mesgs"]
    _sess = _tiz[_tiz["reference_mesg"] == "session"].iloc[0]
    GARMIN_HR_BOUNDS = list(_sess["hr_zone_high_boundary"])  # [88,106,123,141,158,176]
    MAX_HR = int(_sess["max_heart_rate"])       # 176 - Garmin's configured max
    HR_REST = int(_sess["resting_heart_rate"])  # 72

    # 5 zones from the boundary array; sub-88 folds into Z1, 176+ into Z5.
    _b = GARMIN_HR_BOUNDS
    ZONES = [
        ("Z1 recovery",   0,     _b[1], "#3b82f6"),   # < 106
        ("Z2 endurance",  _b[1], _b[2], "#22c55e"),   # 106-123
        ("Z3 tempo",      _b[2], _b[3], "#eab308"),   # 123-141
        ("Z4 threshold",  _b[3], _b[4], "#f97316"),   # 141-158
        ("Z5 VO2max",     _b[4], 999,   "#ef4444"),   # 158+
    ]
    ZONE_ORDER = [z[0] for z in ZONES]

    def zone_minutes(rec: pd.DataFrame) -> pd.Series:
        """Minutes in each Garmin HR zone (absolute bpm boundaries)."""
        d = rec[["timestamp", "heart_rate"]].dropna().copy()
        # seconds each sample represents (clip long gaps from auto-pause)
        dt = d["timestamp"].diff().dt.total_seconds().clip(upper=10).fillna(1.0)
        out = {}
        for name, lo, hi, _c in ZONES:
            mask = (d["heart_rate"] >= lo) & (d["heart_rate"] < hi)
            out[name] = dt[mask].sum() / 60
        return pd.Series(out)

    zone_tbl = (
        pd.DataFrame({n: zone_minutes(d["record_mesgs"])
                      for n, d in activities.items()})
        .T.rename_axis("file").reset_index()
        .merge(summary_full[["file", "activity", "date"]], on="file")
        .melt(id_vars=["file", "activity", "date"],
              var_name="zone", value_name="minutes")
    )

    zone_chart = alt.Chart(zone_tbl).mark_bar().encode(
        y=alt.Y("activity:N", sort=alt.SortField("date"), title=None),
        x=alt.X("minutes:Q", stack="normalize", title="Share of time"),
        color=alt.Color("zone:N", sort=ZONE_ORDER,
                        scale=alt.Scale(domain=ZONE_ORDER,
                                        range=[z[3] for z in ZONES])),
        order=alt.Order("zone:N", sort="ascending"),
        tooltip=["activity", "zone", alt.Tooltip("minutes:Q", format=".0f")],
    ).properties(height=240, width=560,
                 title=f"HR zone distribution \u2014 Garmin zones, max HR {MAX_HR}")
    zone_chart

    mo.output.clear()

    return GARMIN_HR_BOUNDS, MAX_HR, ZONES, ZONE_ORDER, zone_minutes


@app.cell(hide_code=True)
def _(ET, dem_data, mo, np, pd, requests):
    # --- The race course ------------------------------------------------------
    # Pulled from the official "23km du Mont-Blanc" Google My Map. The KML
    # LineString carries Google-provided (pre-smoothed) elevation, which matches
    # the official race spec better than raw SRTM-along-a-GPS-line — so we use it
    # for the course, with SRTM kept alongside for reference.

    COURSE_MID = "1Yb5fI78r0UNk6m2eqKVTRUA46EWC1jvz"
    _KML_NS = {"k": "http://www.opengis.net/kml/2.2"}

    def haversine_cumulative(lat, lon):
        R = 6371000.0
        dlat = np.radians(np.diff(lat))
        dlon = np.radians(np.diff(lon))
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat[:-1])) * np.cos(np.radians(lat[1:]))
             * np.sin(dlon / 2) ** 2)
        return np.concatenate([[0.0], np.cumsum(2 * R * np.arcsin(np.sqrt(a)))])

    _kml = requests.get(
        f"https://www.google.com/maps/d/kml?mid={COURSE_MID}&forcekml=1", timeout=20
    ).text
    _root = ET.fromstring(_kml)

    _line = next(pm.find(".//k:LineString/k:coordinates", _KML_NS)
                 for pm in _root.iter("{http://www.opengis.net/kml/2.2}Placemark")
                 if pm.find(".//k:LineString/k:coordinates", _KML_NS) is not None)
    _pts = np.array([list(map(float, c.split(","))) for c in _line.text.split()])

    course = pd.DataFrame({"lon": _pts[:, 0], "lat": _pts[:, 1], "ele": _pts[:, 2]})
    course["dist_km"] = haversine_cumulative(course["lat"].values,
                                             course["lon"].values) / 1000
    course["ele_dem"] = pd.Series(
        [dem_data.get_elevation(la, lo) for la, lo in zip(course["lat"], course["lon"])],
        dtype="float64",
    ).interpolate(limit_direction="both")

    # Aid-station / waypoint markers, snapped to nearest point on the course.
    aid_stations = []
    for pm in _root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        coord = pm.find(".//k:Point/k:coordinates", _KML_NS)
        if coord is None:
            continue
        lo, la, *_ = map(float, coord.text.strip().split(","))
        i = ((course["lat"] - la) ** 2 + (course["lon"] - lo) ** 2).idxmin()
        aid_stations.append({
            "name": pm.findtext("k:name", "", _KML_NS),
            "dist_km": course.loc[i, "dist_km"],
            "ele": course.loc[i, "ele"],
        })
    aid_stations = pd.DataFrame(aid_stations).sort_values("dist_km").reset_index(drop=True)
    aid_stations

    mo.output.clear()
    return aid_stations, course


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Strava / Garmin training analysis 2026

    On the 28th June 2026, I'm running the Marathon du Mont Blanc 23 km. I began to wonder if I was doing the right training and getting any fitter. Since there is such a lot of vertical in the race which I can't actually run, it is very hard to tell.

    Now before you judge me, I am a very long way from being any kind of elite althlete. Think active person with a lot of other things going on. All I know is that I am going further, I will finish the race and I'm sure training could be improved.

    I thought I would use this goal as an opportunity to see what Marimo is like and code pair with Claude. Also, with all the tools we now have available, why can't the average person get personalised traning for free.

    Goal: are we getting fitter, and how to train better for the race.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the race profile. A lot of up!
    """)
    return


@app.cell
def _(aid_stations, alt, course):
    # Race elevation profile. Shuttle-logistics markers are dropped — only real
    # course features (start/finish, aid stations, landmarks) are shown.
    course_markers = aid_stations[
        ~aid_stations["name"].str.contains("navettes", case=False)
    ].copy()
    course_markers["label"] = (
        course_markers["name"]
        .str.replace(r'"23km du Mont-Blanc"', "", regex=True)
        .str.replace("Ravitaillement", "Ravito", regex=False)
        .str.strip()
    )

    _area = alt.Chart(course).mark_area(
        line={"color": "#e45756"}, color="#fbe1de", opacity=0.8,
    ).encode(
        x=alt.X("dist_km:Q", title="Distance (km)"),
        y=alt.Y("ele:Q", title="Elevation (m)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("dist_km:Q", format=".1f"),
                 alt.Tooltip("ele:Q", format=".0f")],
    )
    _rules = alt.Chart(course_markers).mark_rule(
        color="#444", strokeDash=[3, 3]
    ).encode(x="dist_km:Q")
    _pts = alt.Chart(course_markers).mark_point(
        color="#444", size=55, filled=True
    ).encode(x="dist_km:Q", y="ele:Q",
             tooltip=["label", alt.Tooltip("dist_km:Q", format=".1f"),
                      alt.Tooltip("ele:Q", format=".0f")])
    _txt = alt.Chart(course_markers).mark_text(
        angle=270, align="left", dx=6, fontSize=9, color="#444",
    ).encode(x="dist_km:Q", y=alt.value(6), text="label:N")

    course_profile = (_area + _rules + _pts + _txt).properties(
        width=620, height=280,
        title="23 km du Mont-Blanc — elevation profile",
    )
    course_profile
    return


@app.cell
def _(course, mo, np, pd):
    # Race demands vs what the training files show.
    def climb_segments(c: pd.DataFrame, min_gain: float = 30.0) -> pd.DataFrame:
        """Sustained climbs/descents on a smoothed profile (for naming the big ones)."""
        g = np.arange(0, c["dist_km"].iloc[-1] * 1000, 10.0)
        e = np.interp(g, c["dist_km"] * 1000, c["ele"])
        e = pd.Series(e).rolling(15, center=True, min_periods=1).mean().values
        de = np.diff(e)
        up = de > 0
        segs, i = [], 0
        while i < len(up):
            j = i
            while j < len(up) and up[j] == up[i]:
                j += 1
            gain = e[j] - e[i]
            if abs(gain) >= min_gain:
                segs.append({"start_km": g[i] / 1000, "end_km": g[j] / 1000,
                             "length_km": (g[j] - g[i]) / 1000,
                             "gain_m": gain, "kind": "climb" if up[i] else "descent"})
            i = j
        return pd.DataFrame(segs)

    course_segs = climb_segments(course)
    RACE_DIST_KM = course["dist_km"].iloc[-1]

    # Headline D+/D- = the OFFICIAL published course figures. Our KML-derived
    # profile, lightly smoothed, gives ~1690 / ~685 (D+ matches well); the route
    # geometry from Google My Maps differs slightly from the organisers' own
    # measurement, so the published numbers are the ones to train against.
    RACE_ASCENT_M = 1680    # published D+
    RACE_DESCENT_M = 870    # published D-
    RACE_LOW_M = course["ele"].min()
    RACE_HIGH_M = course["ele"].max()

    biggest = course_segs[course_segs.kind == "climb"].nlargest(3, "gain_m")
    mo.md(
        f"""
        ### Race demands \u2014 23 km du Mont-Blanc

        | Demand | Figure |
        |---|---|
        | Distance | **{RACE_DIST_KM:.1f} km** (measured route; race is branded "23 km") |
        | Vertical gain | **{RACE_ASCENT_M:,.0f} m D+** *(published)* |
        | Vertical loss | **{RACE_DESCENT_M:,.0f} m D\u2212** *(published)* |
        | Low / high point | {RACE_LOW_M:,.0f} m / {RACE_HIGH_M:,.0f} m |
        | Net | +{RACE_ASCENT_M - RACE_DESCENT_M:,.0f} m (uphill finish at Fl\u00e9g\u00e8re) |

        **Three biggest climbs** (from the course profile):
        {(chr(10) + "    ").join(f"- {r.start_km:.1f}\u2013{r.end_km:.1f} km: +{r.gain_m:.0f} m over {r.length_km:.1f} km" for r in biggest.itertuples())}

        It's a net-uphill race finishing at altitude \u2014 front-load nothing, the
        work is continuous and the final 5 km from Fl\u00e9g\u00e8re climbs to the line.
        """
    )
    return RACE_ASCENT_M, RACE_DIST_KM, RACE_HIGH_M, course_segs


@app.cell
def _(mo):
    mo.md("""
    I used the Strava API to extract all my data and start with some simple analysis.
    """)
    return


@app.cell
def _(FIT_FOLDER, STRAVA_STREAMS_DIR, json, os, pd, requests, time):
    # Tokens live in strava_tokens.json (credentials — keep private). Activity
    # summaries are cached to strava_activities.json so we don\'t hit the API every
    # run; set STRAVA_REFRESH = False to re-pull.
    STRAVA_REFRESH = True
    _TOKEN_FILE = os.path.join(FIT_FOLDER, "strava_tokens.json")
    _CACHE_FILE = os.path.join(STRAVA_STREAMS_DIR, "strava_activities.json")

    def strava_access_token() -> str:
        """Return a valid access token, refreshing + re-saving if it has expired."""
        with open(_TOKEN_FILE) as fh:
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
            with open(_TOKEN_FILE, "w") as fh:
                json.dump(tok, fh, indent=2)
        return tok["access_token"]

    def _fetch_all_activities() -> list[dict]:
        headers = {"Authorization": f"Bearer {strava_access_token()}"}
        out, page = [], 1
        while True:
            resp = requests.get(
                "https://www.strava.com/api/v3/athlete/activities",
                headers=headers,
                params={"per_page": 200, "page": page},
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            out.extend(batch)
            page += 1
        return out

    if STRAVA_REFRESH or not os.path.exists(_CACHE_FILE):
        _raw_acts = _fetch_all_activities()
        with open(_CACHE_FILE, "w") as fh:
            json.dump(_raw_acts, fh)
    else:
        with open(_CACHE_FILE) as fh:
            _raw_acts = json.load(fh)

    strava = pd.DataFrame(_raw_acts)
    strava["start"] = pd.to_datetime(strava["start_date"], utc=True)
    strava["start_local"] = pd.to_datetime(
        strava["start_date_local"]
    ).dt.tz_localize(None)
    strava["dist_km"] = strava["distance"] / 1000
    strava["moving_hr"] = strava["moving_time"] / 3600
    strava = strava.sort_values("start").reset_index(drop=True)
    print(f"{len(strava)} activities  |  "
          f"{strava['start'].min().date()} -> {strava['start'].max().date()}")
    strava[["start_local", "name", "type", "dist_km",
            "total_elevation_gain", "average_heartrate"]].tail(10)

    a=1
    return strava, strava_access_token


@app.cell
def _(mo, pd, strava):
    # Long-view training load. Trail-relevant activities only (the engine that
    # matters for a mountain race), aggregated by month. The date picker below
    # alters the timeline on the charts — defaults to this year.
    TRAIL_TYPES = ["Run", "TrailRun", "Hike", "Walk"]
    _trail = strava[strava["type"].isin(TRAIL_TYPES)].copy()
    _trail["month"] = _trail["start_local"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        _trail.groupby("month")
        .agg(activities=("id", "count"),
             dist_km=("dist_km", "sum"),
             vert_m=("total_elevation_gain", "sum"),
             hours=("moving_hr", "sum"))
        .reset_index()
    )

    _min_month = monthly["month"].min().date()
    _today = pd.Timestamp.now().date()
    month_range = mo.ui.date_range(
        start=_min_month, stop=_today,
        value=(pd.Timestamp("2026-01-01").date(), _today),
        label="Timeline",
    )
    month_range
    return month_range, monthly


@app.cell
def _(alt, mo, month_range, monthly):
    # Monthly trail/run charts — reactive to the `month_range` picker above.
    # Drag the dates to zoom (e.g. just this year, or back to 2018).
    _lo, _hi = month_range.value
    _win = monthly[(monthly["month"].dt.date >= _lo)
                   & (monthly["month"].dt.date <= _hi)].copy()
    _win["month_lbl"] = _win["month"].dt.strftime("%b %Y")
    _span = f"{_lo:%b %Y} \u2013 {_hi:%b %Y}"

    # ordinal month axis -> one label per month, and bars fill the band (wide)
    _x = alt.X("month_lbl:O", title="Month", sort=list(_win["month_lbl"]),
               scale=alt.Scale(paddingInner=0.08),
               axis=alt.Axis(labelAngle=-45))

    _dist_bar = alt.Chart(_win).mark_bar(color="#4c78a8").encode(
        x=_x,
        y=alt.Y("dist_km:Q", title="Distance (km)"),
        tooltip=["month_lbl", "activities", "dist_km", "vert_m", "hours"],
    ).properties(height=200, width=620,
                 title=f"Monthly trail/run distance  \u00b7  {_span}")
    _vert_bar = alt.Chart(_win).mark_bar(color="#e45756").encode(
        x=_x,
        y=alt.Y("vert_m:Q", title="Vertical gain (m D+)"),
        tooltip=["month_lbl", "activities", "vert_m", "hours"],
    ).properties(height=200, width=620,
                 title=f"Monthly vertical gain  \u00b7  {_span}")

    mo.vstack([_dist_bar, _vert_bar])
    return


@app.cell
def _(mo, records):
    mo.md(
        """
        ---
        # Deep analysis — every activity since March

        Per-activity streams are pulled
        from Strava — heart rate, GPS, speed, altitude at 1 Hz — giving the same
        depth as the sample `.fit` files I initally downloaded but across all {n} activities.
        """.replace("{n}", str(len(records)))
    )
    return


@app.cell
def _(STRAVA_STREAMS_DIR, json, os, pd, requests, strava, strava_access_token):
    # Per-activity streams since DEEP_START, cached to strava_streams.json.
    # Incremental: loads the cache and only fetches streams for activities not
    # already cached (set STREAMS_REFRESH=True to re-pull every activity).
    DEEP_START = pd.Timestamp("2026-03-01")
    STREAMS_REFRESH = False
    _STREAMS_FILE = os.path.join(STRAVA_STREAMS_DIR, "strava_streams.json")
    _STREAM_KEYS = ("time,distance,altitude,latlng,heartrate,"
                    "velocity_smooth,cadence,grade_smooth")

    deep_acts = strava[strava["start_local"] >= DEEP_START].reset_index(drop=True)

    def _fetch_streams(activity_id: int) -> dict:
        headers = {"Authorization": f"Bearer {strava_access_token()}"}
        resp = requests.get(
            f"https://www.strava.com/api/v3/activities/{activity_id}/streams",
            headers=headers,
            params={"keys": _STREAM_KEYS, "key_by_type": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        return {k: v["data"] for k, v in resp.json().items()}

    # Start from the existing cache (unless a full refresh is requested), then
    # top up any since-March activity whose streams we don't have yet.
    if not STREAMS_REFRESH and os.path.exists(_STREAMS_FILE):
        with open(_STREAMS_FILE) as _fh:
            _streams_raw = json.load(_fh)
    else:
        _streams_raw = {}

    _missing = [int(a) for a in deep_acts["id"] if str(a) not in _streams_raw]
    if _missing:
        print(f"fetching streams for {len(_missing)} new activit"
              f"{'y' if len(_missing) == 1 else 'ies'}...")
        for _aid in _missing:
            try:
                _streams_raw[str(_aid)] = _fetch_streams(int(_aid))
            except Exception as exc:  # skip activities with no stream data
                print(f"  skipped {_aid}: {exc}")
        with open(_STREAMS_FILE, "w") as _fh:
            json.dump(_streams_raw, _fh)

    def _streams_to_records(aid: str, start_utc: pd.Timestamp) -> pd.DataFrame:
        """Strava streams -> a record DataFrame with .fit-compatible columns."""
        s = _streams_raw.get(aid, {})
        if "time" not in s:
            return pd.DataFrame()
        df = pd.DataFrame({"t": s["time"]})
        df["timestamp"] = start_utc + pd.to_timedelta(df["t"], unit="s")
        df["distance"] = s.get("distance")
        df["enhanced_speed"] = s.get("velocity_smooth")
        df["enhanced_altitude"] = s.get("altitude")
        df["heart_rate"] = s.get("heartrate")
        df["cadence"] = s.get("cadence")
        df["grade"] = s.get("grade_smooth")
        if "latlng" in s:
            ll = pd.DataFrame(s["latlng"], columns=["position_lat_deg", "position_long_deg"])
            df["position_lat_deg"] = ll["position_lat_deg"]
            df["position_long_deg"] = ll["position_long_deg"]
        return df

    records = {
        str(r.id): _streams_to_records(str(r.id), r.start)
        for r in deep_acts.itertuples()
    }
    records = {k: v for k, v in records.items() if not v.empty}
    print(f"{len(records)} activities with stream data "
          f"({deep_acts['start_local'].min():%d %b} -> {deep_acts['start_local'].max():%d %b %Y})")

    #
    return deep_acts, records


@app.cell
def _(
    deep_acts,
    dem_data,
    efficiency_metrics,
    np,
    pd,
    records,
    total_ascent,
    zone_minutes,
):
    # Master per-activity table — every streamed activity since March, with
    # DEM-corrected ascent, efficiency, HR-zone split. Reuses the functions built
    # for the .fit files (total_ascent, efficiency_metrics, zone_minutes).
    def _dem_ascent_from_stream(rec: pd.DataFrame) -> float:
        if "position_lat_deg" not in rec.columns:
            return np.nan
        elev = pd.Series(
            [dem_data.get_elevation(la, lo)
             for la, lo in zip(rec["position_lat_deg"], rec["position_long_deg"])],
            dtype="float64",
        ).interpolate(limit_direction="both")
        return total_ascent(rec["distance"], elev)

    _rows = []
    for _a in deep_acts.itertuples():
        _rec = records.get(str(_a.id))
        if _rec is None or _rec.empty:
            continue
        _eff = efficiency_metrics(_rec)
        _zm = zone_minutes(_rec)
        _moving_min = _a.moving_time / 60
        _easy = _zm[["Z1 recovery", "Z2 endurance"]].sum()
        _rows.append({
            "id": _a.id,
            "name": _a.name,
            "type": _a.type,
            "date": _a.start_local,
            "dist_km": _a.dist_km,
            "moving_min": _moving_min,
            "ascent_strava_m": _a.total_elevation_gain,
            "ascent_dem_m": _dem_ascent_from_stream(_rec),
            "pace_min_km": _moving_min / _a.dist_km if _a.dist_km else np.nan,
            "avg_hr": _a.average_heartrate,
            "max_hr": _a.max_heartrate,
            "ef": _eff["ef"],
            "easy_pct": 100 * _easy / _zm.sum() if _zm.sum() else np.nan,
        })

    act = pd.DataFrame(_rows).sort_values("date").reset_index(drop=True)
    act["vert_per_km"] = act["ascent_dem_m"] / act["dist_km"]
    act["week"] = act["date"].dt.to_period("W-SUN").dt.start_time
    RACE_RELEVANT = ["Run", "TrailRun", "Hike", "Walk"]
    act
    return RACE_RELEVANT, act


@app.cell
def _(RACE_RELEVANT, act, alt, mo):
    # Weekly load since March — race-relevant activities (run/hike/walk).
    _wl = act[act["type"].isin(RACE_RELEVANT)].copy()
    weekly_deep = (
        _wl.groupby("week")
        .agg(activities=("id", "count"),
             dist_km=("dist_km", "sum"),
             ascent_m=("ascent_dem_m", "sum"),
             hours=("moving_min", lambda s: s.sum() / 60),
             easy_pct=("easy_pct", "mean"))
        .reset_index()
    )
    weekly_deep["week_lbl"] = weekly_deep["week"].dt.strftime("%d %b")

    # ordinal week axis -> bars fill the band (wide); small padding keeps a thin gap
    _wx = alt.X("week_lbl:O", title="Week", sort=list(weekly_deep["week_lbl"]),
                scale=alt.Scale(paddingInner=0.08),
                axis=alt.Axis(labelAngle=-45))

    _wdist = alt.Chart(weekly_deep).mark_bar(color="#4c78a8").encode(
        x=_wx,
        y=alt.Y("dist_km:Q", title="Distance (km)"),
        tooltip=["week_lbl", "activities", "dist_km", "ascent_m", "hours"],
    ).properties(height=210, width=360, title="Distance / week")
    _wvert = alt.Chart(weekly_deep).mark_bar(color="#e45756").encode(
        x=_wx,
        y=alt.Y("ascent_m:Q", title="Climb (m D+, corrected)"),
        tooltip=["week_lbl", "activities", "ascent_m", "hours"],
    ).properties(height=210, width=360, title="Vertical / week")

    mo.vstack([mo.hstack([_wdist, _wvert], widths="equal")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Altitude correction

    ### Why smooth the elevation?

    I know that my watch's (Venu 3) barometric altimeter, tends to over-read vertical gain. On some runs it massively overestimates the elevation, so for a few recent efforts I have used Strava's elevation correction to re-adjust this.

    In order to do the analysis properly, I need to make sure I am comparing like for like. That means using a more accurate and consistent elevation source rather than relying only on the watch data.

    To do this, I re-derive the elevation from each GPS point using the SRTM (Shuttle Radar Topography Mission) digital elevation model (DEM), then recompute the ascent on a fixed distance grid with smoothing. This means the final elevation gain is based on the terrain itself, rather than being affected by how slowly or quickly I was moving.

    ### What is SRTM?

    SRTM stands for Shuttle Radar Topography Mission. It was a NASA project from February 2000, when the space shuttle Endeavour flew an 11-day mission carrying two radar antennas: one in the cargo bay and one on a 60-metre mast extended out from the shuttle.

    The two antennas captured slightly different radar views of the Earth's surface at the same time. By comparing the phase difference between the two radar returns, NASA was able to calculate ground elevation. In simple terms, it created a near-global 3D map of the Earth's surface.

    The result was one of the first high-quality global elevation datasets:

    - Coverage: around 80% of Earth's land surface, between roughly 60° N and 56° S
    - Resolution: around 30 m horizontally, also known as 1 arc-second
    - Vertical accuracy: a few metres relative, and around 10 m absolute
    - Limitations: better on flatter terrain, less accurate in steep mountains where radar shadowing can leave gaps
    - Status: still one of the standard free global elevation models, used by many elevation APIs, hiking apps, and elevation correction tools

    Behind the scenes, I download the relevant SRTM tiles when needed and look up the elevation for each GPS coordinate. These DEM-corrected elevation values are then used to calculate the corrected D+ figure. In effect, the watch's barometric altimeter is being cross-checked against ground elevation from the SRTM dataset.

    However, re-deriving elevation from GPS and SRTM still gives one elevation reading per track point, and those readings are not perfectly clean. If I simply summed every tiny rise between points, two types of noise would inflate the total climb:

    - **GPS jitter**: the recorded position can wander by a few metres, even when standing still. This means a point can hop between neighbouring SRTM cells and create small fake ups and downs.
    - **DEM noise**: SRTM itself is only accurate to within a few metres vertically, so tiny changes between points are not always real terrain changes.

    If those small ups and downs are summed raw, they can make a real 1,500 m climb look more like 2,000 m (great for the ego). Smoothing the elevation profile over a fixed distance window helps remove this sub-cell noise, so the cumulative gain better reflects the actual terrain rather than the jitter in the data.

    The smoothing window is a trade-off. If it is too small, noise still inflates the D+ figure. If it is too large, genuine short climbs and rolling terrain get flattened away. For that reason, `ELEVATION_SMOOTH_M` is calibrated against Strava's corrected elevation figures below.
    """)
    return


@app.cell
def _(
    ELEVATION_SMOOTH_M,
    activities,
    alt,
    dem_alt,
    mo,
    np,
    pd,
    strava,
    total_ascent,
):
    # How ELEVATION_SMOOTH_M was chosen. We match each .fit file to its Strava
    # activity by start time, keep only the ones Strava actually elevation-
    # corrected (Strava D+ != the raw watch D+), then find the smoothing window
    # whose DEM ascent best matches Strava across that set.
    _cal = []
    for _name, _dfs in activities.items():
        _fit_start = pd.Timestamp(
            _dfs["session_mesgs"].iloc[0]["start_time"]
        ).tz_convert("UTC")
        _i = (strava["start"] - _fit_start).abs().idxmin()
        _watch = _dfs["session_mesgs"].iloc[0].get("total_ascent")
        _strava_v = strava.loc[_i, "total_elevation_gain"]
        _cal.append({
            "file": _name,
            "watch_m": _watch,
            "strava_m": _strava_v,
            "corrected": abs(_strava_v - _watch) > 1,  # Strava actually corrected it
        })
    calibration_set = pd.DataFrame(_cal)

    def _fit_error(smooth_m: float) -> float:
        errs = []
        for r in calibration_set[calibration_set["corrected"]].itertuples():
            pred = total_ascent(
                activities[r.file]["record_mesgs"]["distance"],
                dem_alt[r.file], smooth_m=smooth_m,
            )
            errs.append((pred - r.strava_m) / r.strava_m)
        return float(np.abs(errs).mean() * 100)

    _sweep = pd.DataFrame({"smooth_m": list(range(50, 601, 25))})
    _sweep["mean_abs_pct_err"] = _sweep["smooth_m"].map(_fit_error)
    _best_smooth = _sweep.loc[_sweep["mean_abs_pct_err"].idxmin(), "smooth_m"]

    calibration_set["dem_m"] = [
        round(total_ascent(activities[r.file]["record_mesgs"]["distance"],
                           dem_alt[r.file], smooth_m=ELEVATION_SMOOTH_M))
        for r in calibration_set.itertuples()
    ]

    mo.vstack([
        mo.md(
            f"""
            ### Elevation-correction calibration

            Best-fit smoothing: **{_best_smooth:.0f} m**
            (currently using `ELEVATION_SMOOTH_M = {ELEVATION_SMOOTH_M:.0f}`).
            Calibrated on the **{calibration_set['corrected'].sum()}** activities
            Strava elevation-corrected;
            """
        ),
        calibration_set,
        alt.Chart(_sweep).mark_line(point=True).encode(
            x=alt.X("smooth_m:Q", title="Smoothing window (m)"),
            y=alt.Y("mean_abs_pct_err:Q", title="Mean abs error vs Strava (%)"),
        ).properties(height=180, width=620, title="Calibration sweep"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reading the pace-vs-terrain plot

    Raw pace tells me almost nothing about trail fitness. A steep mountain run looks "slow" because of the terrain, not necessarily because I am unfit. To understand fitness properly, I need to **control for terrain**.

    So I plot every activity since March as:

    - **x — terrain difficulty**: climb per km, in m/km, using DEM-corrected elevation.
      Flat road sits on the left, steep mountain terrain sits on the right.
    - **y — pace**: min/km. Lower means faster.
    - **colour — date**, **shape — activity type**.

    The cloud naturally trends up and to the right: the steeper the terrain gets, the slower I go. That is just the terrain doing its thing, not automatically a sign of fitness.

    The actual fitness signal lives *within a vertical slice* of the plot. At a given terrain difficulty, the points sitting **lower** are faster. So if my recent, brighter points sit below the older ones at the same terrain difficulty, that suggests I am getting fitter for that type of terrain.
    """)
    return


@app.cell
def _(act, alt):
    # Pace vs terrain difficulty — 2026 activities, DEM-corrected climb.
    # DEM correction only covers the streamed window (since 1 March) — which is
    # also the post-anaemia data worth analysing, so "2026" here means Mar onward.
    _ft = act[act["type"].isin(["Run", "TrailRun", "Hike", "Walk"])].copy()

    pace_terrain_chart = alt.Chart(_ft).mark_point(
        size=120, filled=True, opacity=0.85,
    ).encode(
        x=alt.X("vert_per_km:Q",
                title="Terrain difficulty \u2014 climb per km (m/km, DEM-corrected)"),
        y=alt.Y("pace_min_km:Q", title="Pace (min/km) \u2014 lower is faster"),
        color=alt.Color("date:T", title="Date",
                        scale=alt.Scale(range=["#052e16", "#07f059"])),
        shape=alt.Shape("type:N", title="Activity"),
        tooltip=["name", "date:T", "type",
                 alt.Tooltip("dist_km:Q", format=".1f", title="km"),
                 alt.Tooltip("ascent_dem_m:Q", format=".0f", title="D+ (DEM)"),
                 alt.Tooltip("vert_per_km:Q", format=".0f", title="m/km"),
                 alt.Tooltip("pace_min_km:Q", format=".2f", title="pace")],
    ).properties(
        height=430, width=640,
        title=f"Pace vs terrain  \u00b7  {len(_ft)} activities  \u00b7  "
              f"2026 (since 1 Mar), DEM-corrected climb",
    ).interactive()

    pace_terrain_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting HR zones

    I plot the HR zones to see what that reveals.
    """)
    return


@app.cell
def _(
    GARMIN_HR_BOUNDS,
    MAX_HR,
    ZONES,
    ZONE_ORDER,
    act,
    alt,
    mo,
    pd,
    records,
    zone_minutes,
):
    # HR-zone split for every run/hike since March. The aerobic base for a
    # mountain race is built in Z1-Z2 — too much Z3-Z4 is the classic trap.
    _za = act[act["type"].isin(["Run", "Hike", "TrailRun"])].copy()
    _zrows = []
    for _a in _za.itertuples():
        _zm = zone_minutes(records[str(_a.id)])
        for _z in ZONE_ORDER:
            _zrows.append({"date": _a.date, "label": f"{_a.date:%d %b} · {_a.name[:22]}",
                           "type": _a.type, "zone": _z, "minutes": _zm[_z]})
    zone_long = pd.DataFrame(_zrows)

    _pooled = (zone_long.groupby("zone")["minutes"].sum()
               .reindex(ZONE_ORDER))
    _pooled_pct = (_pooled / _pooled.sum() * 100).round(1)

    _zone_colors = alt.Scale(domain=ZONE_ORDER, range=[z[3] for z in ZONES])

    zone_bars = alt.Chart(zone_long).mark_bar().encode(
        y=alt.Y("label:N", sort=alt.SortField("date"), title=None,
                axis=alt.Axis(labelFontSize=8)),
        x=alt.X("minutes:Q", stack="normalize", title="Share of time"),
        color=alt.Color("zone:N", sort=ZONE_ORDER, scale=_zone_colors),
        order=alt.Order("zone:N"),
        tooltip=["label", "zone", alt.Tooltip("minutes:Q", format=".0f")],
    ).properties(height=420, width=560,
                 title=f"HR zones per activity  ·  pooled easy (Z1-Z2) = "
                       f"{_pooled_pct['Z1 recovery'] + _pooled_pct['Z2 endurance']:.0f}%")

    # Reference ruler: the watch's real Garmin zone boundaries in bpm.
    _zone_ref = pd.DataFrame([
        {"zone": _name,
         "lo_bpm": max(_lo, GARMIN_HR_BOUNDS[0]),
         "hi_bpm": min(_hi, GARMIN_HR_BOUNDS[-1])}
        for _name, _lo, _hi, _c in ZONES
    ])
    _zone_ref["mid"] = (_zone_ref["lo_bpm"] + _zone_ref["hi_bpm"]) / 2
    _zone_ref["band"] = (_zone_ref["zone"].str.split().str[0] + "  "
                         + _zone_ref["lo_bpm"].astype(str) + "–"
                         + _zone_ref["hi_bpm"].astype(str))

    _ruler_bar = alt.Chart(_zone_ref).mark_bar(height=22, stroke="white").encode(
        x=alt.X("lo_bpm:Q", title="Heart rate (bpm)  —  max HR " + str(MAX_HR),
                scale=alt.Scale(zero=False, nice=False)),
        x2="hi_bpm:Q",
        color=alt.Color("zone:N", sort=ZONE_ORDER, scale=_zone_colors, legend=None),
        tooltip=["zone", "lo_bpm", "hi_bpm"],
    )
    _ruler_txt = alt.Chart(_zone_ref).mark_text(fontSize=9, color="white").encode(
        x="mid:Q", text="band:N",
    )
    zone_ruler = (_ruler_bar + _ruler_txt).properties(
        height=60, width=560, title="Where the zones are")

    mo.vstack([zone_ruler, zone_bars])
    return (zone_long,)


@app.cell(hide_code=True)
def _(
    RACE_ASCENT_M,
    RACE_DATE,
    RACE_DIST_KM,
    RACE_HIGH_M,
    ZONE_ORDER,
    act,
    course_segs,
    mo,
    monthly,
    zone_long,
):
    # Race-readiness assessment — driven by the streams dataset (since March).
    _runs = act[act["type"] == "Run"].copy()
    _flat = (_runs[_runs["vert_per_km"] < 20]
             .dropna(subset=["ef"]).sort_values("date"))  # truly road runs
    _ef_early = _flat.head(4)["ef"].mean()
    _ef_late = _flat.tail(4)["ef"].mean()
    _ef_gain = (_ef_late / _ef_early - 1) * 100
    _fastest_recent = _flat.iloc[-1]

    _pz = zone_long.groupby("zone")["minutes"].sum().reindex(ZONE_ORDER)
    _easy_pct = 100 * _pz[["Z1 recovery", "Z2 endurance"]].sum() / _pz.sum()
    _z34_pct = 100 * _pz[["Z3 tempo", "Z4 threshold"]].sum() / _pz.sum()

    _y26 = monthly[(monthly.month.dt.year == 2026) & (monthly.month.dt.month <= 5)]
    _y25 = monthly[(monthly.month.dt.year == 2025) & (monthly.month.dt.month <= 5)]
    _dist_vs_ly = (_y26.dist_km.sum() / _y25.dist_km.sum() - 1) * 100
    _longest = act.loc[act["dist_km"].idxmax()]
    _weeks_to_race = (RACE_DATE.tz_localize(None) - act["date"].max()).days / 7

    mo.md(
        f"""
        ## Race readiness — 23 km du Mont-Blanc (28 Jun 2026)

        {_weeks_to_race:.0f} weeks out. Built on **{len(act)} activities since
        1 March** — {len(_runs)} of them runs, full HR + GPS streams.*

        ### Where I stand so far
        The data above tells a fairly clean story:

        - **My base is back.** 2026 Jan–May is **{_dist_vs_ly:+.0f}%** on the
          same months in 2025 — a strong aerobic base is rebuilt.
        - **I'm getting fitter, modestly.** On flat runs my pace-at-fixed-HR is
          roughly stable, but my most recent run is the best of the set — speed
          is just starting to follow the base.
        - **I've already done a race-sized day** ({_longest['name']},
          {_longest['dist_km']:.1f} km) — the mountain legs and time-on-feet exist.
        - **The single biggest lever from here is the HR-zone balance** — see #2.

        ### The race · {RACE_DIST_KM:.1f} km / ≈ {RACE_ASCENT_M:,.0f} m D+
        Net-uphill, finishing at altitude (Flégère, {RACE_HIGH_M:,.0f} m). One
        {course_segs.gain_m.max():.0f} m climb mid-race; then after a fast ~250 m
        descent the final ~7 km climbs ~480 m to the line. No flat recovery.

        ### 1. Am I getting fitter? — base yes, speed starting to
        - **Efficiency:** on truly flat road runs, speed-per-heartbeat is roughly
          stable ({_ef_gain:+.1f}% early-March → now) — but my **most recent run
          is the best of the set** ({_fastest_recent['pace_min_km']:.2f} min/km at
          HR {_fastest_recent['avg_hr']:.0f}).
        - **Volume:** 2026 Jan–May is **{_dist_vs_ly:+.0f}%** vs the same months
          in 2025 — a real aerobic base rebuilt.
        - The honest read: the base is back; the race-specific speed is the work
          of the next 6 weeks.

        ### 2. Biggest opportunity: train *easier* — **{_easy_pct:.0f}% easy**
        Across every run & hike since March, only **{_easy_pct:.0f}%** of my time
        is truly easy (Z1–Z2); **{_z34_pct:.0f}%** sits in Z3–Z4. My flat runs are
        almost entirely tempo-or-harder. A {RACE_DIST_KM:.0f} km mountain race is
        a 2.5–4 h *aerobic* effort — that engine is built slow. I need to shift
        to **~80% easy / 20% hard**. Highest-leverage change on this page.

        ### 3. Volume trend — building well, hold it
        April was a big block; May is continuing. I'm on a good ramp — the job
        now is to peak the long mountain day, then taper the last ~10 days.

        ### 4. Already strong
        My **{_longest['name']}** day ({_longest['dist_km']:.1f} km) is essentially
        race distance. The mountain legs and time-on-feet exist — the gap is doing
        that terrain at a controlled *run*, and the easy/hard balance above.

        *Caveat: SRTM-based D+ under-reads short steep efforts (hill repeats) where
        the climbs are smaller than the ~30 m DEM grid — trust Strava's D+ for those.*
        """
    )
    return


@app.cell(hide_code=True)
def _(GARMIN_HR_BOUNDS, RACE_RELEVANT, act, alt, mo, pd, records):
    # Weekly debrief — the last COMPLETE Monday–Sunday week vs the week before.
    # Filters to race-relevant activities (Run / TrailRun / Hike / Walk).
    # Z3 (123–141 bpm) split at its midpoint; zone mix compared vs the prior week.
    _now = pd.Timestamp.now()
    # Monday 00:00 of the current week; the last complete week ends there (exclusive).
    _this_mon = _now.normalize() - pd.Timedelta(days=_now.weekday())
    _w1_end,  _w1_start = _this_mon,             _this_mon - pd.Timedelta(days=7)
    _w0_end,  _w0_start = _w1_start,             _w1_start - pd.Timedelta(days=7)
    _w1_last = _w1_end - pd.Timedelta(days=1)    # inclusive Sunday, for display

    def _window(df, start, end):
        return df[(df["date"] >= start) & (df["date"] < end)
                  & (df["type"].isin(RACE_RELEVANT))].copy()

    _last7  = _window(act, _w1_start, _w1_end)
    _prior7 = _window(act, _w0_start, _w0_end)

    def _agg(df):
        return dict(n=len(df),
                    km=float(df["dist_km"].sum()),
                    ascent=float(df["ascent_dem_m"].sum()),
                    hours=float(df["moving_min"].sum() / 60))

    _cur, _prv = _agg(_last7), _agg(_prior7)

    def _delta(a, b, unit="", prec=1):
        if b == 0:
            return f"({a:.{prec}f}{unit} vs 0 prior)"
        pct = (a / b - 1) * 100
        arrow = "↑" if pct >= 0 else "↓"
        return f"{arrow} {abs(pct):.0f}%  (prior: {b:.{prec}f}{unit})"

    _summary = mo.md(f"""
    ## Week of {_w1_start:%a %d %b} – {_w1_last:%a %d %b %Y}

    | Metric | This week | vs prior week |
    |---|---|---|
    | Sessions | **{_cur['n']}** | {_delta(_cur['n'], _prv['n'], '', 0)} |
    | Distance | **{_cur['km']:.1f} km** | {_delta(_cur['km'], _prv['km'], ' km')} |
    | Ascent | **{_cur['ascent']:,.0f} m** | {_delta(_cur['ascent'], _prv['ascent'], ' m', 0)} |
    | Moving time | **{_cur['hours']:.1f} h** | {_delta(_cur['hours'], _prv['hours'], ' h')} |
    """)

    _detail = (
        _last7.sort_values("date")
        .assign(day=lambda d: d["date"].dt.strftime("%a %d %b"))
        [["day", "name", "type", "dist_km", "ascent_dem_m",
          "moving_min", "avg_hr", "easy_pct"]]
        .rename(columns={"dist_km": "km", "ascent_dem_m": "ascent_m",
                         "moving_min": "min", "easy_pct": "easy_%"})
        .round({"km": 1, "ascent_m": 0, "min": 0, "avg_hr": 0, "easy_%": 0})
    )

    # HR-zone breakdown with Z3 split into low / high halves at its midpoint
    _z3_lo, _z3_hi = GARMIN_HR_BOUNDS[2], GARMIN_HR_BOUNDS[3]   # 123, 141
    _z3_mid = (_z3_lo + _z3_hi) / 2                               # 132
    _split_zones = [
        ("Z1 recovery",    0,                    GARMIN_HR_BOUNDS[1], "#3b82f6"),
        ("Z2 endurance",   GARMIN_HR_BOUNDS[1],  _z3_lo,              "#22c55e"),
        ("Z3a low tempo",  _z3_lo,               _z3_mid,             "#facc15"),
        ("Z3b high tempo", _z3_mid,              _z3_hi,              "#eab308"),
        ("Z4 threshold",   _z3_hi,               GARMIN_HR_BOUNDS[4], "#f97316"),
        ("Z5 VO2max",      GARMIN_HR_BOUNDS[4],  999,                 "#ef4444"),
    ]
    _split_order  = [z[0] for z in _split_zones]
    _split_colors = [z[3] for z in _split_zones]

    def _zmin(df):
        """Total minutes per (split) zone across a window's activities."""
        out = {n: 0.0 for n, _, _, _ in _split_zones}
        for _row in df.itertuples():
            _rec = records[str(_row.id)][["timestamp", "heart_rate"]].dropna()
            _dt = _rec["timestamp"].diff().dt.total_seconds().clip(upper=10).fillna(1.0)
            for _n, _lo, _hi, _c in _split_zones:
                out[_n] += float(_dt[(_rec["heart_rate"] >= _lo)
                                     & (_rec["heart_rate"] < _hi)].sum() / 60)
        return out

    def _pcts(m):
        _t = sum(m.values())
        return ({k: 100 * v / _t for k, v in m.items()} if _t else
                {k: 0.0 for k in m}), _t

    _m1, _t1 = _pcts(_zmin(_last7))
    _m0, _t0 = _pcts(_zmin(_prior7))

    _zt = pd.DataFrame({"zone": _split_order})
    _zt["minutes"] = _zt["zone"].map(_zmin(_last7))
    _zt["pct"]     = _zt["zone"].map(_m1)
    _zt["pp"]      = _zt["zone"].map(lambda z: _m1[z] - _m0[z])   # pts vs prior wk
    _zt["pct_lbl"] = _zt.apply(
        lambda r: f"{r['pct']:.0f}%  ({r['pp']:+.0f}pp)", axis=1)

    _bars = alt.Chart(_zt).mark_bar().encode(
        y=alt.Y("zone:N", sort=_split_order, title=None),
        x=alt.X("minutes:Q", title="Minutes", scale=alt.Scale(padding=60)),
        color=alt.Color("zone:N", sort=_split_order, legend=None,
                        scale=alt.Scale(domain=_split_order, range=_split_colors)),
        tooltip=["zone",
                 alt.Tooltip("minutes:Q", format=".0f"),
                 alt.Tooltip("pct:Q", format=".1f", title="% this week"),
                 alt.Tooltip("pp:Q", format="+.1f", title="pp vs prior week")],
    )
    _labels = alt.Chart(_zt).mark_text(align="left", baseline="middle",
                                       dx=4, fontWeight="bold").encode(
        y=alt.Y("zone:N", sort=_split_order),
        x="minutes:Q",
        text="pct_lbl:N",
    )
    _zone_chart = (_bars + _labels).properties(
        height=210, width=520,
        title=f"HR-zone minutes, week of {_w1_start:%d %b}  (Z3 split at {_z3_mid:.0f} bpm · pp = pts vs prior week)",
    )

    _easy1 = _m1["Z1 recovery"] + _m1["Z2 endurance"]
    _easy0 = _m0["Z1 recovery"] + _m0["Z2 endurance"]
    _hard1 = _m1["Z4 threshold"] + _m1["Z5 VO2max"]

    _mix = mo.md(f"""
    **Aerobic mix (vs prior week):**

    - **{_easy1:.0f}% easy** — Z1–Z2 ({_easy1 - _easy0:+.0f}pp) · of which **Z2 endurance {_m1['Z2 endurance']:.0f}%** ({_m1['Z2 endurance'] - _m0['Z2 endurance']:+.0f}pp)
    - **{_m1['Z3a low tempo'] + _m1['Z3b high tempo']:.0f}% tempo** — Z3: **{_m1['Z3a low tempo']:.0f}% low** ({_z3_lo}–{_z3_mid:.0f}) + **{_m1['Z3b high tempo']:.0f}% high** ({_z3_mid:.0f}–{_z3_hi})
    - **{_hard1:.0f}% hard** — Z4–Z5 ({_hard1 - (_m0['Z4 threshold'] + _m0['Z5 VO2max']):+.0f}pp)

    Mountain-ultra base target ≈ 80% easy. Z2 is the engine — more of it is the goal.
    """)

    mo.vstack([_summary, _detail, _zone_chart, _mix])
    return


@app.cell(hide_code=True)
def _(act, alt, mo, np, pd):
    # Fitness trend — aerobic efficiency (EF = speed / HR). EF is terrain- and
    # intensity-sensitive, so the trend is fit ONLY on comparable *steady road
    # runs* (flat < 20 m/km, run at a real aerobic effort, avg HR ≥ 148). Easy,
    # interval and hilly runs are plotted faded for context — their low EF is
    # expected, not a fitness drop.
    _runs = act[act["type"] == "Run"].dropna(subset=["ef"]).copy()
    _runs["steady"] = (_runs["vert_per_km"] < 20) & (_runs["avg_hr"] >= 148)
    _runs["kind"] = np.where(_runs["steady"], "steady road run",
                             "easy / interval / hilly")

    _steady = _runs[_runs["steady"]].sort_values("date")
    _d0 = _steady["date"].min()
    _xday = (_steady["date"] - _d0).dt.total_seconds() / 86400
    _coef = np.polyfit(_xday, _steady["ef"], 1)
    _pct_wk = (_coef[0] * 7) / _steady["ef"].mean() * 100
    _trend = pd.DataFrame({
        "date": [_steady["date"].min(), _steady["date"].max()],
        "ef":   [np.polyval(_coef, _xday.min()), np.polyval(_coef, _xday.max())],
    })
    _best = _steady.loc[_steady["ef"].idxmax()]
    _six_ago = pd.Timestamp.now() - pd.Timedelta(weeks=6)

    _pts = alt.Chart(_runs).mark_point(filled=True, opacity=0.85).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("ef:Q", title="Efficiency factor (speed / HR)",
                scale=alt.Scale(zero=False)),
        size=alt.Size("vert_per_km:Q", title="Climb (m/km)",
                      scale=alt.Scale(range=[40, 450])),
        color=alt.Color("kind:N", title=None,
                        scale=alt.Scale(
                            domain=["steady road run", "easy / interval / hilly"],
                            range=["#2563eb", "#cbd5e1"])),
        tooltip=["name", "date:T", alt.Tooltip("ef:Q", format=".5f"),
                 alt.Tooltip("avg_hr:Q", title="avg HR"),
                 alt.Tooltip("vert_per_km:Q", format=".0f", title="m/km"),
                 alt.Tooltip("pace_min_km:Q", format=".2f", title="pace")],
    )
    _trend_line = alt.Chart(_trend).mark_line(
        color="#2563eb", strokeDash=[6, 4], size=2.5).encode(x="date:T", y="ef:Q")
    _six_rule = alt.Chart(pd.DataFrame({"date": [_six_ago]})).mark_rule(
        color="#94a3b8", strokeDash=[2, 2]).encode(x="date:T")

    _ef_chart = (_pts + _trend_line + _six_rule).properties(
        height=260, width=580,
        title="Fitness trend — EF on comparable steady runs  (dashed grey = 6 weeks ago)",
    )

    _dir = ("improving" if _pct_wk > 0.1
            else "declining" if _pct_wk < -0.1 else "holding steady")
    _caption = mo.md(f"""
    **Am I getting fitter?** On comparable **steady road runs** (flat, HR ≥ 148)
    my speed-per-heartbeat is **{_dir}** — fitted slope **{_pct_wk:+.1f}% / week**
    across the base period. Best steady EF so far: **{_best['ef']:.4f}** on
    {_best['date']:%d %b}.

    The faded points are easy / interval / hilly runs — low EF there is *by design*
    (easy pace, or climbing), not lost fitness. Aerobic gains from this month's Z2
    block take **2–6 weeks** to show up as EF, so the payoff of the easy work lands
    in June. Watch the blue line.
    """)

    mo.vstack([_ef_chart, _caption])

    return


@app.cell
def _(RACE_DATE, ZONE_ORDER, mo, pd, zone_long):
    # Week-by-week plan to race day, regenerated from today's date. Built around
    # the findings above: lock in the easy/Z2 balance, peak the long mountain day,
    # practise descents + tired climbing, then taper. Phases are keyed to
    # weeks-to-race, so re-running later still lays out the correct remaining plan.
    _race = RACE_DATE.tz_localize(None).normalize()
    _today = pd.Timestamp.now().normalize()
    _mon0 = _today - pd.Timedelta(days=_today.weekday())   # Monday of this week
    _starts = []
    _m = _mon0
    while _m <= _race:
        _starts.append(_m)
        _m = _m + pd.Timedelta(weeks=1)

    # Live easy/Z2 share for the current 7 days (original 5-zone scheme)
    _zc = zone_long[zone_long["date"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
    _zsum = _zc.groupby("zone")["minutes"].sum().reindex(ZONE_ORDER)
    _zt = _zsum.sum()
    _easy_now = 100 * _zsum[["Z1 recovery", "Z2 endurance"]].sum() / _zt if _zt else 0
    _z2_now   = 100 * _zsum["Z2 endurance"] / _zt if _zt else 0

    # Content keyed by weeks-to-race (0 = race week)
    _tmpl = {
        0: dict(phase="Race",  easy="race",  long_day="Race — 23 km du Mont-Blanc 🏔️",
                quality="Mon/Tue: 20–30 min easy + 3 short strides",
                focus="Volume ~50% down. Stay off the legs, fuel well, arrive fresh."),
        1: dict(phase="Taper", easy="~80%", long_day="16–18 km / 900 m, relaxed",
                quality="2 × 10 min at race-climb effort",
                focus="Cut volume ~25%. Legs start to feel springy — keep every easy run in Z2."),
        2: dict(phase="Peak",  easy="~75%", long_day="24 km / 1400 m — race simulation",
                quality="Climb intervals when already tired (end of a medium run)",
                focus="Biggest week. Rehearse race fuelling + kit. Easy runs strictly Z2."),
        3: dict(phase="Build", easy="~70%", long_day="20 km / 1200 m on race-type terrain",
                quality="Sustained climb 20–25 min at threshold",
                focus="Add controlled downhill running on the long day. Easy days stay conversational."),
        4: dict(phase="Build", easy="≥ 65%", long_day="18–20 km / ~1100 m, easy",
                quality="Hill repeats 6–8 × 2 min uphill hard",
                focus="Hold the Z2 reset — only the hill session is hard; everything else is Z2."),
    }
    _default = dict(phase="Build", easy="≥ 65%", long_day="16–20 km / ~1000 m, easy",
                    quality="Hill repeats 6–8 × 2 min uphill hard",
                    focus="Build aerobic volume; keep ~80% of running time easy (Z2).")

    _rows = []
    _n = len(_starts)
    for _i, _ws in enumerate(_starts):
        _out = _n - 1 - _i                       # weeks to race
        _c = _tmpl.get(_out, _default)
        _is_now = _ws <= _today <= _ws + pd.Timedelta(days=6)
        _wk_lbl = "Race wk" if _out == 0 else f"{_out} wk{'s' if _out != 1 else ''} out"
        if _is_now:
            _wk_lbl += " ← now"
        _rows.append(dict(week=_wk_lbl, week_of=_ws.strftime("%d %b"),
                          phase=_c["phase"], easy_target=_c["easy"],
                          long_day=_c["long_day"], quality=_c["quality"],
                          focus=_c["focus"]))
    training_plan = pd.DataFrame(_rows)

    _wks_to_race = (_race - _today).days / 7

    mo.vstack([
        mo.md(
            f"""
            ---
            ## My plan → 28 June  ({_wks_to_race:.0f} weeks out)

            **The Z2 reset is working.** This week I'm at **{_easy_now:.0f}% easy**
            with **Z2 endurance at {_z2_now:.0f}%** of my training time — up
            sharply, and exactly the engine a 2.5–4 h mountain race runs on.
            Steady-run efficiency is holding with a slight upward tilt; the real
            aerobic payoff of this block lands over the next few weeks.

            My weekly shape: **2 easy/Z2 runs · 1 long mountain day ·
            1 quality session · 1–2 rest / cross-train days.** The one
            non-negotiable rule, all the way through the taper: ***easy days are
            actually easy (Z2, conversational).*** Each week's **easy target**
            ramps toward ~80% — that is what turns my volume into race fitness.
            """
        ),
        training_plan,
        mo.md(
            """
            **Threaded through every week (incl. the taper):**

            - **Z2 discipline** — only the one flagged quality session goes hard.
              If a run drifts above Z2 on the flat, I ease off. This holds *through*
              the taper, not just the build.
            - **Descents** — the race drops ~250 m fast mid-course and finishes
              on tired quads. I *run* downhills on the long days, not just hike them.
            - **Tired climbing** — the last 7 km is all up. Climb efforts go at
              the *end* of sessions, not the start.
            - **Altitude + even climb pacing** — my Jun-7 breathlessness only hit
              above ~1750 m, and mostly because I climbed hard low down then met the
              altitude ceiling. Put key long days **up high (1800–2200 m)** and climb
              from the bottom at an **even, conversational effort**, lifting only near
              the top — familiarity and pacing, not heroics.
            - **Fuel & feet** — rehearse eating on the move + race shoes / kit on
              the peak long day, never on race day.
            - **Recheck** — re-run this notebook weekly to watch easy-%, Z2 share
              and the EF trend move.
            """
        ),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same climb, two days — May 1 vs Jun 7 (breathlessness check)

    Using the **shared Strava segment "Marathon du Mont-Blanc · 2ème montée"** (+19 %,
    the actual race climb) — a fixed bit of trail that both the **1 May** run and the
    **7 Jun** session climbed. A true like-for-like, ~290 m of ascent from 1424 → 1715 m.

    Note the context: on **Jun 7** you reached this climb **~65 % into a 16 km outing
    (already tired)**, whereas on **May 1** you hit it **fresh, near the start** — so any
    edge on Jun 7 is *despite* being more fatigued.
    """)
    return


@app.cell(hide_code=True)
def _(pd, records):
    # Like-for-like on the shared Strava segment (MdMB 2ème montée, +19%). The
    # index ranges are the Strava segment-effort start/end into each activity's
    # stream (records[aid] is 1:1 with the stream samples).
    cmp_J = records["18824580666"].iloc[3609:4360].copy()   # Jun 7 effort
    cmp_M = records["18330726193"].iloc[250:868].copy()      # May 1 effort

    def _stats(s):
        _t = (s["timestamp"].iloc[-1] - s["timestamp"].iloc[0]).total_seconds()
        _d = s["distance"].iloc[-1] - s["distance"].iloc[0]
        _g = s["enhanced_altitude"].iloc[-1] - s["enhanced_altitude"].iloc[0]
        return dict(gain=_g, dist=_d, mins=_t / 60, vam=_g / (_t / 3600),
                    pace=(_t / 60) / (_d / 1000), hr=s["heart_rate"].mean(),
                    hrmax=s["heart_rate"].max(), eff=(_g / (_t / 3600)) / s["heart_rate"].mean())
    cmp_j, cmp_m = _stats(cmp_J), _stats(cmp_M)

    cmp_table = pd.DataFrame({
        "Metric": ["Altitude gain", "Distance", "Time", "Avg HR", "Max HR",
                   "Vertical speed", "Pace", "Climb efficiency", "Where in the run"],
        "Jun 7": [f"+{cmp_j['gain']:.0f} m", f"{cmp_j['dist']:.0f} m", f"{cmp_j['mins']:.1f} min",
            f"{cmp_j['hr']:.0f} bpm", f"{cmp_j['hrmax']:.0f} bpm", f"{cmp_j['vam']:.0f} m/h",
            f"{cmp_j['pace']:.1f} min/km", f"{cmp_j['eff']:.1f} m/h·bpm", "~65% in (tired)"],
        "May 1": [f"+{cmp_m['gain']:.0f} m", f"{cmp_m['dist']:.0f} m", f"{cmp_m['mins']:.1f} min",
            f"{cmp_m['hr']:.0f} bpm", f"{cmp_m['hrmax']:.0f} bpm", f"{cmp_m['vam']:.0f} m/h",
            f"{cmp_m['pace']:.1f} min/km", f"{cmp_m['eff']:.1f} m/h·bpm", "near start (fresh)"],
    })
    cmp_table
    return cmp_J, cmp_M, cmp_j, cmp_m


@app.cell(hide_code=True)
def _(cmp_J, mo, records):
    # Map: faint = full routes (grey Jun 7, blue May 1); bold red = the shared
    # race climb (Strava "MdMB 2ème montée"). Green = bottom, red pin = top.
    import folium
    cmp_fullJ = records["18824580666"]; cmp_fullM = records["18330726193"]
    cmp_map = folium.Map(
        location=[float(cmp_J["position_lat_deg"].mean()), float(cmp_J["position_long_deg"].mean())],
        zoom_start=15, tiles=None)
    folium.TileLayer("OpenStreetMap").add_to(cmp_map)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                     attr="OpenTopoMap", name="Terrain").add_to(cmp_map)
    folium.PolyLine(cmp_fullJ[["position_lat_deg", "position_long_deg"]].dropna().values.tolist(),
                    color="#777", weight=2.5, opacity=0.5, tooltip="Jun 7 — full route").add_to(cmp_map)
    folium.PolyLine(cmp_fullM[["position_lat_deg", "position_long_deg"]].dropna().values.tolist(),
                    color="#3b82f6", weight=2.5, opacity=0.45, tooltip="May 1 — full route").add_to(cmp_map)
    folium.PolyLine(cmp_J[["position_lat_deg", "position_long_deg"]].values.tolist(),
                    color="#e4513f", weight=6, opacity=0.95,
                    tooltip="Shared race climb (compared)").add_to(cmp_map)
    folium.Marker([cmp_J["position_lat_deg"].iloc[0], cmp_J["position_long_deg"].iloc[0]],
        tooltip=f"Bottom ~{cmp_J['enhanced_altitude'].iloc[0]:.0f} m",
        icon=folium.Icon(color="green", icon="play")).add_to(cmp_map)
    folium.Marker([cmp_J["position_lat_deg"].iloc[-1], cmp_J["position_long_deg"].iloc[-1]],
        tooltip=f"Top ~{cmp_J['enhanced_altitude'].iloc[-1]:.0f} m",
        icon=folium.Icon(color="red", icon="flag")).add_to(cmp_map)
    folium.LayerControl().add_to(cmp_map)
    cmp_map.fit_bounds([
        [float(cmp_J["position_lat_deg"].min()), float(cmp_J["position_long_deg"].min())],
        [float(cmp_J["position_lat_deg"].max()), float(cmp_J["position_long_deg"].max())]])
    mo.iframe(cmp_map._repr_html_(), height="460px")
    return


@app.cell(hide_code=True)
def _(alt, cmp_J, cmp_M, cmp_j, cmp_m, mo, np, pd):
    # HR vs altitude on the shared climb + the verdict.
    _alts = np.linspace(max(cmp_J["enhanced_altitude"].iloc[0], cmp_M["enhanced_altitude"].iloc[0]),
                        min(cmp_J["enhanced_altitude"].iloc[-1], cmp_M["enhanced_altitude"].iloc[-1]), 40)
    def _hr_vs_alt(seg):
        _o = seg.sort_values("enhanced_altitude")
        return np.interp(_alts, _o["enhanced_altitude"], _o["heart_rate"])
    cmp_curve = pd.concat([
        pd.DataFrame({"altitude": _alts, "hr": _hr_vs_alt(cmp_J), "run": "Jun 7 (tired, 65% in)"}),
        pd.DataFrame({"altitude": _alts, "hr": _hr_vs_alt(cmp_M), "run": "May 1 (fresh)"}),
    ])
    _chart = alt.Chart(cmp_curve).mark_line(point=True).encode(
        x=alt.X("altitude:Q", title="Altitude (m)", scale=alt.Scale(zero=False)),
        y=alt.Y("hr:Q", title="Heart rate (bpm)", scale=alt.Scale(zero=False)),
        color=alt.Color("run:N", title=None,
            scale=alt.Scale(domain=["Jun 7 (tired, 65% in)", "May 1 (fresh)"], range=["#e4513f", "#3b82f6"])),
        tooltip=["run", alt.Tooltip("altitude:Q", format=".0f"), alt.Tooltip("hr:Q", format=".0f")],
    ).properties(width=560, height=300, title="HR vs altitude — shared race climb (1424→1715 m)")

    _verdict = mo.md(f"""
    **On the identical race climb, Jun 7 was *not* the weaker effort:**

    - Heart rate **{cmp_j['hr']:.0f} vs {cmp_m['hr']:.0f} bpm** — slightly **lower** on Jun 7.
    - Slightly **faster**: {cmp_j['mins']:.1f} vs {cmp_m['mins']:.1f} min ({cmp_j['vam']:.0f} vs {cmp_m['vam']:.0f} m/h vertical).
    - And this was **despite hitting the climb tired** (65 % into the run) vs fresh on May 1.

    So on this lower climb (≤1715 m) the effort held up or improved — no sign of a
    limiter here.

    **The breathlessness was higher up.** This climb tops 1715 m, but on Jun 7 you carried
    on to **1994 m**. Breathlessness that is *ventilatory* (air hunger at altitude) sits
    alongside a normal/low HR rather than a high one — so it won't show on this lower
    section. The next cells zoom in on that high ground.
    """)
    mo.vstack([_chart, _verdict])
    return


@app.cell(hide_code=True)
def _(alt, mo, pd, records):
    # Jun 7 FIRST climb — Argentière (1266 m) up to the 1994 m high point: the steep,
    # high one where the breathlessness hit. HR (red) over the altitude profile (grey),
    # with her Z4 (141) and Z5 (158) thresholds dashed. Wrist-optical HR is noisy when
    # power-hiking, so read single-second dips with caution; the repeated threshold
    # spikes at near-walking speed are the real signal.
    cmp_climb = records["18824580666"].copy()
    cmp_climb["km"] = cmp_climb["distance"] / 1000
    cmp_climb = cmp_climb.loc[:cmp_climb["enhanced_altitude"].idxmax()].reset_index(drop=True)

    _up = cmp_climb[cmp_climb["enhanced_altitude"] >= 1750]
    _up_hr = _up["heart_rate"].mean()
    _up_z4 = (_up["heart_rate"] >= 141).mean() * 100
    _up_t = (_up["timestamp"].iloc[-1] - _up["timestamp"].iloc[0]).total_seconds() / 60

    _base = alt.Chart(cmp_climb)
    _altA = _base.mark_area(color="#cfc8b8", opacity=0.7).encode(
        x=alt.X("km:Q", title="Distance (km)"),
        y=alt.Y("enhanced_altitude:Q", title="Altitude (m)", scale=alt.Scale(zero=False)))
    _hrL = _base.mark_line(color="#e4513f", strokeWidth=1.5).encode(
        x="km:Q",
        y=alt.Y("heart_rate:Q", title="Heart rate (bpm)", scale=alt.Scale(domain=[80, 170])))
    _z4 = alt.Chart(pd.DataFrame({"y": [141]})).mark_rule(
        color="#f97316", strokeDash=[5, 4]).encode(y="y:Q")
    _z5 = alt.Chart(pd.DataFrame({"y": [158]})).mark_rule(
        color="#ef4444", strokeDash=[5, 4]).encode(y="y:Q")
    cmp_climb_chart = alt.layer(_altA, _hrL + _z4 + _z5).resolve_scale(
        y="independent").properties(width=640, height=330,
        title="Jun 7 first climb — HR (red) vs altitude (grey).  Dashed: Z4 141 · Z5 158")

    _note = mo.md(f"""
    **Above 1750 m** ({_up_t:.0f} min for the last +{_up['enhanced_altitude'].iloc[-1]-_up['enhanced_altitude'].iloc[0]:.0f} m,
    ~22 min/km): HR averaged **{_up_hr:.0f}** but spiked to **{_up['heart_rate'].max():.0f}**, with
    **{_up_z4:.0f}% of the time at Z4+ (≥141)** — a push-then-recover grind at near-walking pace.

    Compare the **2ème montée lower down** (≤1715 m), where you held a smooth steady **154**.
    The difference isn't fitness — it's that the limiter showed up **only up high**, which is the
    **altitude / ventilation** signature, not a whole-day cardiac or anaemia one.
    """)
    mo.vstack([cmp_climb_chart, _note])
    return (cmp_climb,)


@app.cell(hide_code=True)
def _(alt, cmp_climb, mo, pd):
    # HR vs ALTITUDE on the Jun 7 first climb — does HR break at a specific height?
    # Faint dots = every second; bold line = mean HR per 50 m band. Note HR peaks
    # around 1700 m then FALLS toward the 1994 m summit while you keep climbing —
    # the engine can't be driven up high (ventilation-limited), not an anaemia look.
    cmp_band = cmp_climb.assign(band=(cmp_climb["enhanced_altitude"] // 50 * 50))
    cmp_bin = cmp_band.groupby("band").agg(
        hr=("heart_rate", "mean"), spd=("enhanced_speed", "mean")).reset_index()

    _dots = alt.Chart(cmp_climb).mark_circle(size=14, opacity=0.22, color="#e4513f").encode(
        x=alt.X("enhanced_altitude:Q", title="Altitude (m)", scale=alt.Scale(zero=False)),
        y=alt.Y("heart_rate:Q", title="Heart rate (bpm)", scale=alt.Scale(domain=[90, 170])))
    _meanl = alt.Chart(cmp_bin).mark_line(point=True, color="#b3271a", strokeWidth=3).encode(
        x="band:Q", y="hr:Q",
        tooltip=[alt.Tooltip("band:Q", title="alt"), alt.Tooltip("hr:Q", format=".0f"),
                 alt.Tooltip("spd:Q", format=".2f", title="m/s")])
    _z4 = alt.Chart(pd.DataFrame({"y": [141]})).mark_rule(color="#f97316", strokeDash=[5, 4]).encode(y="y:Q")
    _z5 = alt.Chart(pd.DataFrame({"y": [158]})).mark_rule(color="#ef4444", strokeDash=[5, 4]).encode(y="y:Q")
    cmp_hralt = (_dots + _meanl + _z4 + _z5).properties(width=640, height=340,
        title="Jun 7 first climb — HR vs altitude (bold = 50 m mean).  Dashed: Z4 141 · Z5 158")

    mo.vstack([cmp_hralt, mo.md(
        "HR rises to **~148 by 1700 m**, then **declines** (136 → 131 → 122) as you push on to "
        "1994 m — classic altitude ceiling: you're breathing hard but can't lift HR further.")])
    return


@app.cell(hide_code=True)
def _(alt, mo, pd, records):
    # Altitude theory, second day: Jun 7 (pushed hard) vs 29 May Posettes (easy hike,
    # which went HIGHER, to 2192 m). Mean HR per 50 m band. At matched altitude they
    # converge (~130 above 1750 m) and BOTH taper near the top — but on Posettes you
    # rode an easy ~121 lower down, so it never felt like a struggle.
    def _hr_bands(aid, lo=1400):
        _d = records[aid]
        _d = _d[_d["enhanced_altitude"] >= lo]
        return (_d.assign(band=(_d["enhanced_altitude"] // 50 * 50))
                  .groupby("band").agg(hr=("heart_rate", "mean")).reset_index())
    cmp_two = pd.concat([
        _hr_bands("18824580666").assign(run="Jun 7 — pushed (top 1994 m)"),
        _hr_bands("18701189577").assign(run="29 May Posettes — easy (top 2192 m)"),
    ])
    cmp_two_chart = alt.Chart(cmp_two).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("band:Q", title="Altitude (m)", scale=alt.Scale(zero=False)),
        y=alt.Y("hr:Q", title="Mean HR (bpm)", scale=alt.Scale(zero=False)),
        color=alt.Color("run:N", title=None, scale=alt.Scale(
            domain=["Jun 7 — pushed (top 1994 m)", "29 May Posettes — easy (top 2192 m)"],
            range=["#e4513f", "#2563eb"])),
        tooltip=["run", "band", alt.Tooltip("hr:Q", format=".0f")],
    ).properties(width=640, height=340, title="HR vs altitude — Jun 7 vs 29 May Posettes")

    cmp_verdict2 = mo.md(f"""
    ### Altitude test — verdict

    - **Matched altitude + pace** (1750–1990 m, both ~0.8 m/s): HR **131 (Jun 7) ≈ 130 (Posettes)**.
      Jun 7 was *not* running a high HR for the altitude — it lines up with another high day.
    - **Both days HR tapers near the summit** — the altitude ceiling, not specific to Jun 7.
    - The difference in *feel*: on Jun 7 you **pushed hard low down** (HR 141–148 at 1500–1750 m) then
      hit that ceiling and got breathless; on Posettes you **cruised easy** (HR ~121) so it never bit.

    **So: this reads as an altitude / ventilation + intensity effect.** The fix is specificity
    and pacing — see the plan: put key long days up high, and climb from the bottom at an
    even, conversational effort rather than going out hard low down.
    """)
    mo.vstack([cmp_two_chart, cmp_verdict2])
    return


if __name__ == "__main__":
    app.run()

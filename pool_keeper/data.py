# -*- coding: utf-8 -*-
"""
Module d'accès aux données LNH et gestion de la cache.
=====================================================
Appels HTTP à l'API LNH, calculs match par match des gardiens, et mise en cache.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import os
import pickle
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import streamlit as st

from pool_keeper.config import (
    CACHE_DIR,
    FLOOR_FACTOR,
    GAMES,
    LEAGUE_SCHEDULE,
    LEAGUE_TRANSLATION,
    PRE_NHL_AGE_CUTOFF,
    PRE_NHL_MAX_GP_EQ,
    PRE_NHL_MIN_GP,
    RULES,
    STATS,
    WEB,
)
from pool_keeper.utils import season_shift

_session = requests.Session()
_session.headers.update({"User-Agent": "pool-keeper-app/1.0"})


def http_get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 30) -> Any:
    """Effectue une requête GET avec réessais en cas d'erreur transitoire."""
    last = None
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code} – {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(1.2 * (i + 1))
    raise last


def _read(path: str) -> Any:
    """Lit un objet sérialisé pickle en toute sécurité."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_get(key: str, ttl: int | None = 3600) -> Any:
    """Fichiers _F = saison terminée (permanents). Fichiers _N = saison en cours (expirent)."""
    obj = _read(os.path.join(CACHE_DIR, f"{key}_F.pkl"))
    if obj is not None:
        return obj
    pn = os.path.join(CACHE_DIR, f"{key}_N.pkl")
    if os.path.exists(pn) and (ttl is None or time.time() - os.path.getmtime(pn) < ttl):
        return _read(pn)
    return None


def cache_put(key: str, obj: Any, final: bool) -> None:
    """Enregistre un objet dans la cache disque."""
    path = os.path.join(CACHE_DIR, f"{key}_{'F' if final else 'N'}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        if final:
            old = os.path.join(CACHE_DIR, f"{key}_N.pkl")
            if os.path.exists(old):
                os.remove(old)
    except Exception:
        pass


def cache_purge_nonfinal() -> None:
    """Supprime les fichiers de cache temporaires de la saison en cours."""
    for fn in os.listdir(CACHE_DIR):
        if fn.endswith("_N.pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, fn))
            except Exception:
                pass


def current_season_id() -> int:
    """Identifie l'ID de la saison en cours (ex: 20242025)."""
    try:
        d = http_get(f"{WEB}/season", retries=2)
        if isinstance(d, list) and d:
            return int(max(d))
    except Exception:
        pass
    t = dt.date.today()
    y = t.year if t.month >= 9 else t.year - 1
    return y * 10000 + y + 1


def fetch_report(kind: str, report: str, season: int, game_type: int = 2) -> pd.DataFrame:
    """Récupère un rapport spécifique de l'API de statistiques LNH."""
    params = {
        "isAggregate": "false",
        "isGame": "false",
        "start": 0,
        "limit": -1,
        "cayenneExp": f"seasonId={season} and gameTypeId={game_type}",
    }
    data = http_get(f"{STATS}/{kind}/{report}", params)
    return pd.DataFrame(data.get("data", []))


def _to_minutes(s: pd.Series) -> pd.Series:
    """Convertit un temps de glace en minutes décimales si nécessaire."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().any() and s.median() > 60:
        s = s / 60.0
    return s


def _bios(kind: str, season: int) -> pd.DataFrame:
    """Récupère la biographie (date de naissance, équipe actuelle) des joueurs."""
    try:
        b = fetch_report(kind, "bios", season)
    except Exception:
        return pd.DataFrame(columns=["pid", "birth", "cur_team"])
    if b.empty:
        return pd.DataFrame(columns=["pid", "birth", "cur_team"])
    b = b.drop_duplicates("playerId")
    return pd.DataFrame({
        "pid": b["playerId"],
        "birth": b["birthDate"] if "birthDate" in b else None,
        "cur_team": b["currentTeamAbbrev"] if "currentTeamAbbrev" in b else None,
    })


def fetch_advanced_stats(season: int) -> pd.DataFrame:
    """Récupère et fusionne les statistiques avancées des patineurs pour une saison donnée."""
    frames: list[pd.DataFrame] = []

    # --- scoringRates ---
    try:
        sr = fetch_report("skater", "scoringRates", season)
        if not sr.empty:
            sr = sr.rename(columns={"playerId": "pid"})
            col_map = {
                "goals5v5": "g5v5",
                "assists5v5": "a5v5",
                "points5v5": "pts5v5",
                "goalsPer605v5": "g_per60_5v5",
                "assistsPer605v5": "a_per60_5v5_tmp",
                "pointsPer605v5": "ipp5v5_per60",
                "primaryAssists5v5": "a1_5v5_tmp",
                "primaryAssistsPer605v5": "a1_per60_5v5",
                "secondaryAssists5v5": "a2_5v5_tmp",
                "secondaryAssistsPer605v5": "a2_per60_5v5",
                "shootingPct5v5": "sh_pct_5v5",
                "offensiveZoneStartPct5v5": "oz_start_pct",
                "satPct": "corsi_pct",
                "satRelative5v5": "corsi_rel",
                "onIceShootingPct5v5": "on_ice_sh_pct",
                "netMinorPenaltiesPer60": "net_pen_per60",
                "timeOnIcePerGame5v5": "toi_5v5",
            }
            keep = ["pid"] + [c for c in col_map if c in sr.columns]
            sr = sr[keep].rename(columns=col_map).drop_duplicates("pid")
            if "toi_5v5" in sr.columns:
                sr["toi_5v5"] = pd.to_numeric(sr["toi_5v5"], errors="coerce") / 60.0
            frames.append(sr)
    except Exception:
        pass

    # --- realtime ---
    try:
        rt = fetch_report("skater", "realtime", season)
        if not rt.empty:
            rt = rt.rename(columns={"playerId": "pid"})
            col_map = {
                "hits": "hits",
                "hitsPer60": "hits_per60",
                "blockedShots": "blocked_shots",
                "blockedShotsPer60": "blocks_per60",
                "giveaways": "giveaways",
                "giveawaysPer60": "give_per60",
                "takeaways": "takeaways",
                "takeawaysPer60": "take_per60",
                "missedShots": "missed_shots",
                "totalShotAttempts": "total_shot_att",
                "firstGoals": "first_goals",
            }
            keep = ["pid"] + [c for c in col_map if c in rt.columns]
            rt = rt[keep].rename(columns=col_map).drop_duplicates("pid")
            frames.append(rt)
    except Exception:
        pass

    # --- percentages ---
    try:
        pct = fetch_report("skater", "percentages", season)
        if not pct.empty:
            pct = pct.rename(columns={"playerId": "pid"})
            col_map = {
                "satPercentage": "corsi_pct",
                "satRelative": "corsi_rel",
                "shootingPct5v5": "sh_pct_5v5",
                "skaterShootingPlusSavePct5v5": "pdo",
                "zoneStartPct5v5": "oz_start_pct",
                "usatPercentage": "fenwick_pct",
                "usatRelative": "fenwick_rel",
            }
            keep = ["pid"] + [c for c in col_map if c in pct.columns]
            pct = pct[keep].rename(columns=col_map).drop_duplicates("pid")
            frames.append(pct)
    except Exception:
        pass

    # --- scoringpergame ---
    try:
        spg = fetch_report("skater", "scoringpergame", season)
        if not spg.empty:
            spg = spg.rename(columns={"playerId": "pid"})
            col_map = {
                "shotsPerGame": "shots_per_gp",
                "primaryAssistsPerGame": "a1_per_gp",
                "secondaryAssistsPerGame": "a2_per_gp",
            }
            keep = ["pid"] + [c for c in col_map if c in spg.columns]
            spg = spg[keep].rename(columns=col_map).drop_duplicates("pid")
            frames.append(spg)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=["pid", "season"])

    merged = frames[0]
    for f in frames[1:]:
        # suffix duplicates; keep left value
        merged = merged.merge(f, on="pid", how="left", suffixes=("", "_dup"))
        for c in list(merged.columns):
            if c.endswith("_dup"):
                orig = c[: -len("_dup")]
                if orig in merged.columns:
                    merged[orig] = merged[orig].fillna(merged[c])
                else:
                    merged = merged.rename(columns={c: orig})
                merged = merged.drop(columns=[c], errors="ignore")

    merged["season"] = season
    return merged


def fetch_skaters(season: int) -> pd.DataFrame:
    """Récupère et fusionne les statistiques des patineurs pour une saison donnée."""
    s = fetch_report("skater", "summary", season)
    if s.empty:
        return s
    try:
        t = fetch_report("skater", "timeonice", season)
    except Exception:
        t = pd.DataFrame()
    ren = {"playerId": "pid", "timeOnIcePerGame": "toi_t", "ppTimeOnIcePerGame": "pp_toi"}
    if not t.empty:
        t = t[[c for c in ren if c in t.columns]].rename(columns=ren).drop_duplicates("pid")
    else:
        t = pd.DataFrame(columns=["pid"])
    df = pd.DataFrame({
        "pid": s["playerId"],
        "name": s["skaterFullName"],
        "pos": s["positionCode"],
        "team": s["teamAbbrevs"].astype(str).str.split(",").str[-1].str.strip(),
        "gp": s["gamesPlayed"],
        "g": s["goals"],
        "a": s["assists"],
        "toi": s["timeOnIcePerGame"] if "timeOnIcePerGame" in s else np.nan,
    })
    df = df.merge(t, on="pid", how="left").merge(_bios("skater", season), on="pid", how="left")
    if "toi_t" in df:
        df["toi"] = df["toi"].fillna(df["toi_t"])
    if "pp_toi" not in df:
        df["pp_toi"] = np.nan
    df["toi"] = _to_minutes(df["toi"])
    df["pp_toi"] = _to_minutes(df["pp_toi"])
    df["team"] = df["cur_team"].fillna(df["team"])
    df["season"] = season
    adv = fetch_advanced_stats(season)
    if not adv.empty:
        df = df.merge(adv, on="pid", how="left", suffixes=("", "_adv"))
        # resolve season duplicate from adv merge
        for c in list(df.columns):
            if c.endswith("_adv"):
                df = df.drop(columns=[c], errors="ignore")
    return df


def fetch_goalies(season: int) -> pd.DataFrame:
    """Récupère et formate les statistiques des gardiens pour une saison donnée."""
    s = fetch_report("goalie", "summary", season)
    if s.empty:
        return s
    df = pd.DataFrame({
        "pid": s["playerId"],
        "name": s["goalieFullName"],
        "pos": "G",
        "team": s["teamAbbrevs"].astype(str).str.split(",").str[-1].str.strip(),
        "gp": s["gamesPlayed"],
        "gs": s["gamesStarted"] if "gamesStarted" in s else s["gamesPlayed"],
        "w": s["wins"],
        "sv": s["saves"],
        "ga": s["goalsAgainst"],
    })
    df = df.merge(_bios("goalie", season), on="pid", how="left")
    df["team"] = df["cur_team"].fillna(df["team"])
    df["season"] = season
    return df[["season", "pid", "name", "pos", "team", "gp", "gs", "w", "sv", "ga", "birth"]]


def fetch_season(season: int, cur: int) -> dict[str, pd.DataFrame]:
    """Récupère les patineurs et gardiens d'une saison avec mise en cache."""
    key = f"season_{season}"
    final = season < cur
    c = cache_get(key, ttl=None if final else 3600)
    if c is not None and isinstance(c, dict) and "sk" in c and "corsi_pct" in c["sk"].columns:
        return c
    try:
        obj = {"sk": fetch_skaters(season), "gl": fetch_goalies(season)}
        cache_put(key, obj, final)
        return obj
    except Exception:
        stale = cache_get(key, ttl=None)
        if stale is not None:
            return stale
        raise


def goalie_floor_points(pid: int, season: int, cur: int) -> dict[str, float] | None:
    """Calcule les points de pool exacts d'un gardien match par match avec plancher à 0."""
    key = f"gl_{season}_{pid}"
    c = cache_get(key, ttl=3600)
    if c is not None:
        return c
    try:
        data = http_get(f"{WEB}/player/{pid}/game-log/{season}/2", retries=2)
    except Exception:
        return None
    logs = data.get("gameLog", [])
    if not logs or not any("decision" in g for g in logs):
        return None
    pts, n = 0.0, 0
    for g in logs:
        sa, ga = g.get("shotsAgainst"), g.get("goalsAgainst")
        if sa is None or ga is None:
            continue
        w = 1 if g.get("decision") == "W" else 0
        p = RULES["G_WIN"] * w + RULES["G_SAVE"] * (sa - ga) + RULES["G_GA"] * ga
        pts += max(0.0, p)
        n += 1
    res = {"pts": pts, "gp": n}
    cache_put(key, res, season < cur)
    return res


def attach_goalie_points(gl: pd.DataFrame, season: int, cur: int, use_logs: bool) -> pd.DataFrame:
    """Attribue les points de pool aux gardiens (match par match ou par approximation)."""
    gl = gl.copy()
    approx = (RULES["G_WIN"] * gl["w"] + RULES["G_SAVE"] * gl["sv"] + RULES["G_GA"] * gl["ga"]).clip(lower=0)
    gl["pool"] = approx * FLOOR_FACTOR
    gl["exact"] = False
    if use_logs:
        ids = gl.loc[gl["gp"] >= 5, "pid"].tolist()
        res = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(goalie_floor_points, pid, season, cur): pid for pid in ids}
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception:
                    r = None
                if r:
                    res[futs[f]] = r["pts"]
        m = gl["pid"].map(res)
        gl["exact"] = m.notna()
        gl["pool"] = m.fillna(gl["pool"])
    return gl


def fetch_pre_nhl_player(pid: int, pos: str = "F") -> dict[str, Any]:
    """Récupère et traduit les statistiques pré-LNH (junior, AHL, Europe) d'un joueur."""
    key = f"pre_nhl_v5_{pid}"
    c = cache_get(key, ttl=86400 * 14)
    if c is not None:
        return c

    default_res = {"pre_rate": 0.0, "pre_gp_eq": 0.0, "pre_summary": "", "top_league": "", "rows": []}
    try:
        data = http_get(f"{WEB}/player/{pid}/landing", retries=2)
    except Exception:
        return default_res

    season_totals = data.get("seasonTotals", [])
    if not isinstance(season_totals, list):
        return default_res

    rows = []
    for s in season_totals:
        if not isinstance(s, dict):
            continue
        if s.get("gameTypeId") != 2:
            continue
        league = s.get("leagueAbbrev", "")
        if league == "NHL" or league not in LEAGUE_TRANSLATION:
            continue
        gp = s.get("gamesPlayed", 0)
        if gp < PRE_NHL_MIN_GP:
            continue
        g = s.get("goals") or 0
        a = s.get("assists") or 0
        pts = (2.0 * g + a) if pos == "D" else (g + a)
        rate = pts / max(gp, 1)
        trans_factor, evid_factor = LEAGUE_TRANSLATION[league]
        rate_nhl = rate * trans_factor

        # Normalisation selon le calendrier réel de la ligue :
        # Ex: 36 matchs en NCAA = 100% d'une saison, pas une demi-saison !
        sched_len = LEAGUE_SCHEDULE.get(league, 68)
        season_fraction = min(1.0, gp / sched_len)
        gp_eq = season_fraction * GAMES * evid_factor

        season_id = s.get("season", 0)
        rows.append({
            "season": season_id,
            "league": league,
            "gp": gp,
            "g": g,
            "a": a,
            "pts": pts,
            "rate_lg": round(rate, 2),
            "rate_nhl": round(rate_nhl, 3),
            "gp_eq": round(gp_eq, 1),
        })

    if not rows:
        cache_put(key, default_res, final=True)
        return default_res

    rows.sort(key=lambda x: x["season"])
    max_yr = rows[-1]["season"] // 10000

    # Prioriser les saisons pré-LNH récentes (D-0, D-1) :
    # Les saisons de 16 ans dans les ligues mineures ne doivent pas diluer le calibre d'entrée
    recent_rows = [r for r in rows if (max_yr - (r["season"] // 10000)) <= 1]
    target_rows = recent_rows if recent_rows else rows

    sum_rate_w, sum_w, total_gp_eq = 0.0, 0.0, 0.0
    lg_weights: dict[str, float] = {}
    for r in target_rows:
        yr = r["season"] // 10000
        recency = 0.70 ** max(0, max_yr - yr)
        w = r["gp_eq"] * recency
        sum_rate_w += r["rate_nhl"] * w
        sum_w += w
        total_gp_eq += r["gp_eq"]
        lg_weights[r["league"]] = lg_weights.get(r["league"], 0.0) + w

    avg_rate = (sum_rate_w / sum_w) if sum_w > 0 else 0.0
    max_rate = max(r["rate_nhl"] for r in target_rows) if target_rows else 0.0
    pre_rate = (0.65 * max_rate + 0.35 * avg_rate) if max_rate > avg_rate else avg_rate

    pre_gp_eq = min(float(total_gp_eq), float(PRE_NHL_MAX_GP_EQ))
    top_lg = max(lg_weights.items(), key=lambda kv: kv[1])[0] if lg_weights else ""
    top_gp = sum(r["gp"] for r in rows if r["league"] == top_lg)
    summary = f"{top_lg} ({top_gp} mj)" if top_lg else ""

    res = {
        "pre_rate": round(pre_rate, 3),
        "pre_gp_eq": round(pre_gp_eq, 1),
        "pre_summary": summary,
        "top_league": top_lg,
        "rows": rows,
    }
    cache_put(key, res, final=True)
    return res


def fetch_pre_nhl_batch(skaters: pd.DataFrame, cur: int, max_workers: int = 10) -> dict[int, dict[str, Any]]:
    """Récupère les stats pré-LNH en parallèle pour les joueurs admissibles (jeunes ou peu d'expérience)."""
    tot_gp = skaters.groupby("pid")["gp"].sum()
    last_season = skaters.groupby("pid")["season"].max()
    pos_map = skaters.groupby("pid")["pos"].last()

    recent_pids = set(last_season[last_season >= season_shift(cur, -1)].index)
    eligible = [pid for pid in recent_pids if tot_gp.get(pid, 0) < 200]

    out: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_pre_nhl_player, pid, pos_map.get(pid, "F")): pid for pid in eligible}
        for f in as_completed(futs):
            pid = futs[f]
            try:
                res = f.result()
                if res and res.get("pre_gp_eq", 0) > 0:
                    out[pid] = res
            except Exception:
                pass
    return out


@st.cache_data(ttl=3600, show_spinner="Chargement des données de la LNH…")
def load_data(n_seasons: int, use_logs: bool, token: int) -> dict[str, Any]:
    """Charge n saisons d'historique (+ saison courante). `token` sert à forcer un rafraîchissement."""
    cur = current_season_id()
    seasons = [season_shift(cur, -i) for i in range(n_seasons - 1, -1, -1)]
    sk_l, gl_l, warns = [], [], []
    for s in seasons:
        try:
            obj = fetch_season(s, cur)
        except Exception as e:
            warns.append(f"Saison {s} indisponible : {e}")
            continue
        if len(obj["sk"]):
            sk_l.append(obj["sk"])
        if len(obj["gl"]):
            gl_l.append(attach_goalie_points(obj["gl"], s, cur, use_logs))
    if not sk_l:
        raise RuntimeError("Aucune donnée reçue de l'API de la LNH. " + " | ".join(warns))
    sk = pd.concat(sk_l, ignore_index=True)
    gl = pd.concat(gl_l, ignore_index=True) if gl_l else pd.DataFrame()
    cur_eff = int(sk["season"].max())
    pre_nhl = fetch_pre_nhl_batch(sk, cur_eff)
    return {
        "skaters": sk,
        "goalies": gl,
        "cur": cur_eff,
        "n_seasons": n_seasons,
        "pre_nhl": pre_nhl,
        "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "warnings": warns,
        "goalie_exact_share": float(gl["exact"].mean()) if len(gl) and "exact" in gl else 0.0,
    }


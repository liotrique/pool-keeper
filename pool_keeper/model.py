# -*- coding: utf-8 -*-
"""
Modélisation statistique et projections de pool keeper (v2 – validée par backtest).
==================================================================================

Ce qui change par rapport à la v1 (et pourquoi) :

1. BACKTEST WALK-FORWARD : le modèle est testé sur le passé (on prédit la saison t+1 avec les
   données ≤ t, courbes d'âge incluses, sans fuite d'information) et comparé à des références
   naïves (dernière saison, moyenne 3 ans). Les métriques sont exposées dans `model["backtest"]`.
2. CALIBRATION : décroissance temporelle, force de régression vers la moyenne et jeu de variables
   du prior (aucune / TOI / TOI + TOI en avantage numérique) sont choisis par le backtest, plus
   un facteur de biais par position.
3. INCERTITUDE : chaque projection a un écart-type calibré sur les erreurs réelles (par position
   et par quantité d'information disponible) ; on en tire intervalles P10/P90 et fiabilité.
4. VORP ESPÉRÉ SUR LE RYTHME : E[max(0, Pace − remplacement)] × Disponibilité.
   Corrige le biais de pénalité double pour les joueurs d'élite fragiles. Le niveau de
   remplacement est calculé sur le rythme 82 matchs.
5. PLUS DE DOUBLE COMPTAGE : le score = valeur espérée en points (keeper + saison actuelle).
   Opportunité, durabilité et upside sont déjà DANS les projections (TOI → prior, disponibilité,
   survie, dispersion) ; ils restent affichés comme diagnostics mais n'entrent plus dans le Score.
6. PLUS d'écrasement au 99e centile : Score = % de la valeur annuelle du meilleur joueur.
7. Version vectorisée (panel joueurs × saisons) : permet le backtest et la grille en quelques secondes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from lightgbm import LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False
import streamlit as st

from pool_keeper.config import (
    AGES,
    GAMES,
    K_PRIOR,
    MAX_YEARS,
    MIN_GP,
    RULES,
    SLOTS,
)
from pool_keeper.utils import norm, season_shift

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DU MODÈLE
# ──────────────────────────────────────────────────────────────────────────────
GROUPS = ("F", "D", "G")
GRP_NAME = {"F": "Attaquants", "D": "Défenseurs", "G": "Gardiens"}
FEATS = {
    "none": [],
    "toi": ["toi", "toi2"],
    "toi_pp": ["toi", "toi2", "pp"],
    "boost": ["boost"],
}
BOOST_FEATURES = [
    "Temps de glace (TOI)",
    "Temps avantage num. (PP)",
    "Âge",
    "Matchs joués (GP)",
    "Rythme points/match",
    "Rythme buts/match",
    "Rythme passes/match",
    "Tendance TOI (an t vs t-1)",
    "Tendance points (an t vs t-1)",
    "Tirs par match",
    "Possession (Corsi CF%)",
    "Déploiement off. (OZ Start%)",
    "% tir 5c5 (Sh%)",
    "PDO (chance)",
    "Production 5c5/60 min",
    "Buts/60 min 5c5",
    "Passes primaires/match",
    "TOI forces égales",
    "Récupérations/60",
    "Pénalités nettes/60",
    "Signal de chance au tir",
    "Ratio production primaire",
]
AV_PRIOR = {"F": (0.88, 0.75), "D": (0.88, 0.75), "G": (0.45, 0.60)}   # (valeur a priori, poids en saisons)
AV_CAP = {"F": 0.97, "D": 0.97, "G": 0.82}
DECL_FROM = {"F": 31, "D": 31, "G": 34}          # âge où la disponibilité s'érode
INFO_BINS = np.array([60.0, 150.0])              # « matchs d'évidence » : faible / moyen / élevé
Z80 = 1.2816                                     # intervalle central à 80 %
RHO_TALENT = 0.6                                 # corrélation des erreurs d'une année à l'autre (hypothèse)
MIN_PAIRS_C = 40                                 # échantillon minimal pour calibrer un écart-type
GRID_DECAY = (0.35, 0.45, 0.55)
GRID_KMULT = (0.5, 1.0, 1.5)
DEFAULT_C = {("F", 0): 2.1, ("F", 1): 1.8, ("F", 2): 1.6,
             ("D", 0): 2.1, ("D", 1): 1.8, ("D", 2): 1.6,
             ("G", 0): 2.2, ("G", 1): 1.9, ("G", 2): 1.7}


@dataclass
class ModelParams:
    """Hyperparamètres du modèle (calibrés par backtest quand c'est possible)."""
    decay: float = 0.35
    k_mult: float = 0.5
    feats: str = "boost"
    cal: dict = field(default_factory=lambda: {g: 1.0 for g in GROUPS})
    sd_c: dict = field(default_factory=lambda: dict(DEFAULT_C))
    sd_growth: float = 1.12
    calibrated: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# PRÉPARATION DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────
def _validate(data: dict[str, Any]) -> None:
    need = {"pid", "name", "pos", "team", "gp", "g", "a", "season", "birth"}
    miss = need - set(data["skaters"].columns)
    if miss:
        raise ValueError(f"Colonnes manquantes (skaters) : {sorted(miss)}")
    if len(data["goalies"]):
        needg = {"pid", "name", "gp", "pool", "season", "birth"}
        missg = needg - set(data["goalies"].columns)
        if missg:
            raise ValueError(f"Colonnes manquantes (goalies) : {sorted(missg)}")


def _add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Âge exact au 1er février de la saison considérée."""
    bm = df.dropna(subset=["birth"]).groupby("pid")["birth"].first()
    df["birth"] = df["pid"].map(bm)
    ref = pd.to_datetime((df["season"] // 10000 + 1).astype(str) + "-02-01")
    df["age"] = (ref - pd.to_datetime(df["birth"], errors="coerce")).dt.days / 365.25
    return df


def prepare_long(data: dict[str, Any]) -> pd.DataFrame:
    """Format long (joueur × saison) avec points de pool historiques selon la Constitution."""
    _validate(data)
    sk = data["skaters"].copy()
    slen = sk.groupby("season")["gp"].max()
    last_pos = sk.sort_values("season").groupby("pid")["pos"].last()
    sk["pos"] = sk["pid"].map(last_pos)
    sk["grp"] = np.where(sk["pos"] == "D", "D", "F")
    is_d = sk["grp"] == "D"
    sk["pool"] = (
        sk["g"] * np.where(is_d, RULES["D_GOAL"], RULES["F_GOAL"])
        + sk["a"] * np.where(is_d, RULES["D_ASSIST"], RULES["F_ASSIST"])
    )
    frames = [sk]
    gl = data["goalies"]
    if len(gl):
        gl = gl.copy()
        gl["grp"] = "G"
        frames.append(gl)
    ps = pd.concat(frames, ignore_index=True)
    for c in ["gs", "w", "toi", "pp_toi",
              "corsi_pct", "corsi_rel", "fenwick_pct", "oz_start_pct",
              "sh_pct_5v5", "on_ice_sh_pct", "pdo",
              "shots_per_gp", "a1_per_gp", "a2_per_gp",
              "g_per60_5v5", "a1_per60_5v5", "a2_per60_5v5", "ipp5v5_per60",
              "toi_5v5", "hits_per60", "blocks_per60",
              "give_per60", "take_per60", "total_shot_att", "first_goals",
              "net_pen_per60"]:
        if c not in ps:
            ps[c] = np.nan
    ps = _add_age(ps)
    ps = ps[ps["gp"] > 0].copy()
    # un seul enregistrement par (joueur, saison) : on garde celui qui a le plus de matchs
    ps = ps.sort_values("gp", ascending=False).drop_duplicates(["pid", "season"]).sort_values(["pid", "season"])
    ps["rate"] = ps["pool"] / ps["gp"]
    ps["slen"] = ps["season"].map(slen).astype(float)
    return ps.reset_index(drop=True)


@dataclass
class Panel:
    """Représentation matricielle joueurs × saisons (toutes les opérations deviennent vectorielles)."""
    pids: np.ndarray
    grp: np.ndarray
    name: np.ndarray
    pos: np.ndarray
    team: np.ndarray
    seasons: list
    yrs: np.ndarray
    slen: np.ndarray
    gp: np.ndarray
    pool: np.ndarray
    rate: np.ndarray
    toi: np.ndarray
    pp: np.ndarray
    gs: np.ndarray
    w: np.ndarray
    age: np.ndarray
    g: np.ndarray | None = None
    a: np.ndarray | None = None
    lg_gpg: np.ndarray | None = None
    corsi_pct: np.ndarray | None = None
    corsi_rel: np.ndarray | None = None
    oz_start_pct: np.ndarray | None = None
    sh_pct_5v5: np.ndarray | None = None
    on_ice_sh_pct: np.ndarray | None = None
    pdo: np.ndarray | None = None
    shots_per_gp: np.ndarray | None = None
    a1_per_gp: np.ndarray | None = None
    ipp5v5_per60: np.ndarray | None = None
    g_per60_5v5: np.ndarray | None = None
    toi_5v5: np.ndarray | None = None
    hits_per60: np.ndarray | None = None
    take_per60: np.ndarray | None = None
    total_shot_att: np.ndarray | None = None
    first_goals: np.ndarray | None = None
    net_pen_per60: np.ndarray | None = None


def make_panel(ps: pd.DataFrame) -> Panel:
    seasons = sorted(int(s) for s in ps["season"].unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    pids = ps["pid"].unique()
    pidx = {p: i for i, p in enumerate(pids)}
    n, T = len(pids), len(seasons)
    ri = ps["pid"].map(pidx).to_numpy()
    ci = ps["season"].map(sidx).to_numpy()

    def mat(col: str, fill: float = np.nan) -> np.ndarray:
        a = np.full((n, T), fill, dtype=float)
        a[ri, ci] = ps[col].to_numpy(dtype=float)
        return a

    gp, pool = mat("gp", 0.0), mat("pool", 0.0)
    g_mat, a_mat = mat("g", 0.0), mat("a", 0.0)
    rate = np.where(gp > 0, pool / np.maximum(gp, 1e-9), 0.0)
    last = ps.loc[ps.groupby("pid")["season"].idxmax()].set_index("pid").loc[pids]
    yrs = np.array([s // 10000 for s in seasons])
    last_idx = np.array([sidx[int(s)] for s in last["season"]])
    age_last = np.where(last["age"].isna(), 26.0, last["age"].to_numpy(dtype=float))
    age = age_last[:, None] - (yrs[last_idx][:, None] - yrs[None, :])
    slen = ps.groupby("season")["slen"].max().reindex(seasons).to_numpy(dtype=float)

    # Moyenne de buts marqués par match d'équipe et par saison (normalisation d'ère)
    sk = ps[ps["grp"] != "G"]
    tm_stats = sk.groupby(["season", "team"]).agg(g=("g", "sum"), gp=("gp", "max")).reset_index()
    tm_stats["gpg"] = tm_stats["g"] / tm_stats["gp"].clip(lower=1)
    lg_gpg_series = tm_stats.groupby("season")["gpg"].mean()
    lg_gpg = np.array([lg_gpg_series.get(s, 3.0) for s in seasons], dtype=float)

    return Panel(
        pids=pids, grp=last["grp"].to_numpy(), name=last["name"].to_numpy(), pos=last["pos"].to_numpy(),
        team=last["team"].to_numpy(), seasons=seasons, yrs=yrs, slen=slen, gp=gp, pool=pool, rate=rate,
        toi=mat("toi"), pp=mat("pp_toi"), gs=mat("gs"), w=mat("w"), age=age,
        g=g_mat, a=a_mat, lg_gpg=lg_gpg,
        corsi_pct=mat("corsi_pct"), corsi_rel=mat("corsi_rel"),
        oz_start_pct=mat("oz_start_pct"), sh_pct_5v5=mat("sh_pct_5v5"),
        on_ice_sh_pct=mat("on_ice_sh_pct"), pdo=mat("pdo"),
        shots_per_gp=mat("shots_per_gp"), a1_per_gp=mat("a1_per_gp"),
        ipp5v5_per60=mat("ipp5v5_per60"), g_per60_5v5=mat("g_per60_5v5"),
        toi_5v5=mat("toi_5v5"), hits_per60=mat("hits_per60"),
        take_per60=mat("take_per60"), total_shot_att=mat("total_shot_att"),
        first_goals=mat("first_goals"), net_pen_per60=mat("net_pen_per60"),
    )


def _complete_flags(panel: Panel, cur: int) -> np.ndarray:
    """Saison exploitable comme cible/origine : assez longue et (si courante) quasi terminée."""
    comp = panel.slen >= 0.6 * GAMES
    for j, s in enumerate(panel.seasons):
        if s == cur and panel.slen[j] < 0.9 * GAMES:
            comp[j] = False
    return comp


# ──────────────────────────────────────────────────────────────────────────────
# COURBES D'ÂGE ET DE RÉTENTION (estimées sur les données, mélangées à un prior)
# ──────────────────────────────────────────────────────────────────────────────
def prior_delta(grp: str, a: int | float) -> float:
    """Évolution a priori de la production selon l'âge et la position."""
    if grp == "G":
        return 1.06 if a < 23 else 1.03 if a < 26 else 1.01 if a < 30 else 0.99 if a < 33 else 0.97 if a < 36 else 0.94
    if grp == "D":
        # Les défenseurs ont une courbe de développement jusqu'à 25-26 ans, plateau doré de 26 à 30 ans, puis déclin graduel
        return (1.12 if a < 21 else 1.08 if a < 24 else 1.04 if a < 26 else 1.01 if a < 30
                else 0.985 if a < 32 else 0.96 if a < 34 else 0.93)
    # Les attaquants progressent fortement vers leur sommet, avec un plateau doré entre 24 et 29.5 ans
    return (1.14 if a < 21 else 1.09 if a < 23 else 1.04 if a < 25 else 1.01 if a < 29
            else 0.985 if a < 31 else 0.96 if a < 33 else 0.93)


def prior_ret(grp: str, a: int | float) -> float:
    """Probabilité a priori de rester un joueur LNH utile la saison suivante."""
    if grp == "G":
        return 0.97 if a < 33 else 0.93 if a < 36 else 0.85 if a < 38 else 0.70
    # Dans la fleur de l'âge (24-30 ans), le risque de sortie de ligue pour un joueur régulier est minime
    return 0.985 if a < 29 else 0.965 if a < 32 else 0.92 if a < 34 else 0.83 if a < 36 else 0.65


def _blend(emp: dict, n: dict, prior_fn: Any, lo: float, hi: float, strength: int = 40) -> np.ndarray:
    vals = []
    for a in AGES:
        pr = prior_fn(a)
        nn, e = n.get(a, 0), emp.get(a, np.nan)
        if nn > 0 and np.isfinite(e):
            al = nn / (nn + strength)
            vals.append(al * e + (1 - al) * pr)
        else:
            vals.append(pr)
    return pd.Series(vals).rolling(3, center=True, min_periods=1).mean().clip(lo, hi).values


def build_curves(ps: pd.DataFrame, cur: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Courbes d'âge (variation du taux de production) et de rétention, par groupe F/D/G."""
    slen_cur = ps.loc[ps["season"] == cur, "slen"].max()
    last_ok = cur if (np.isfinite(slen_cur) and slen_cur >= 70) else season_shift(cur, -1)
    u = ps[ps["season"] <= last_ok]
    nx = u[["pid", "season", "rate", "gp"]].copy()
    nx["season"] = nx["season"].map(lambda s: season_shift(s, -1))
    nx = nx.rename(columns={"rate": "rate_n", "gp": "gp_n"})
    curves, rets = {}, {}
    for grp in GROUPS:
        mg = MIN_GP[grp]
        d = u[(u["grp"] == grp) & (u["gp"] >= mg) & u["age"].notna() & (u["season"] < last_ok)]
        m = d.merge(nx, on=["pid", "season"], how="left")
        pr = m[m["gp_n"] >= mg].copy()
        pr["a"] = pr["age"].round().astype(int)
        pr["w"] = np.minimum(pr["gp"], pr["gp_n"])
        pr["num"] = pr["w"] * pr["rate_n"]
        pr["den"] = pr["w"] * pr["rate"]
        agg = pr.groupby("a").agg(num=("num", "sum"), den=("den", "sum"), n=("pid", "size"))
        emp = {a: r.num / r.den for a, r in agg.iterrows() if r.den > 0}
        delta = _blend(emp, agg["n"].to_dict(), lambda a, g=grp: prior_delta(g, a), 0.85, 1.18, strength=50)
        curves[grp] = np.concatenate([[1.0], np.cumprod(delta[:-1])])
        thr = 5 if grp == "G" else 10
        m["a"] = m["age"].round().astype(int)
        m["kept"] = (m["gp_n"] >= thr).astype(float)
        ag = m.groupby("a")["kept"].agg(["mean", "size"])
        rets[grp] = _blend(ag["mean"].to_dict(), ag["size"].to_dict(),
                           lambda a, g=grp: prior_ret(g, a), 0.3, 0.99)
    return curves, rets


# ──────────────────────────────────────────────────────────────────────────────
# PRIOR D'UTILISATION ET MODÈLE DE GRADIENT BOOSTING (GBDT)
# ──────────────────────────────────────────────────────────────────────────────
def _feat_matrix(toi: np.ndarray, pp: np.ndarray, cols: list[str]) -> np.ndarray:
    toi = np.clip(toi, 8, 28)
    table = {"toi": toi, "toi2": toi ** 2, "pp": pp}
    return np.column_stack([table[c] for c in cols])


def _extract_skater_features(
    panel: Panel,
    p_idx: np.ndarray,
    s: int,
    toi_override: np.ndarray | None = None,
    pp_override: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extrait les variables explicatives pour le modèle de Gradient Boosting (GBDT).
    Capture temps de glace (global et PP), âge, cadence de production de base,
    buts/passes par match, dynamiques de progression, ainsi que les métriques avancées
    (volume de tirs, possession Corsi, zone starts, PDO/chance, efficacité 5c5).
    """
    toi = np.clip(panel.toi[p_idx, s] if toi_override is None else toi_override, 8.0, 28.0)
    pp = np.nan_to_num(panel.pp[p_idx, s] if pp_override is None else pp_override)
    age = panel.age[p_idx, s]
    gp = panel.gp[p_idx, s]
    rate = panel.rate[p_idx, s]
    g_rate = np.where(gp > 0, panel.g[p_idx, s] / np.maximum(gp, 1.0), 0.0)
    a_rate = np.where(gp > 0, panel.a[p_idx, s] / np.maximum(gp, 1.0), 0.0)
    if s >= 1:
        toi_prev = panel.toi[p_idx, s - 1]
        rate_prev = panel.rate[p_idx, s - 1]
        t_toi = np.where(np.isfinite(toi_prev) & np.isfinite(toi), toi - toi_prev, 0.0)
        t_rate = np.where(np.isfinite(rate_prev) & np.isfinite(rate), rate - rate_prev, 0.0)
    else:
        t_toi = np.zeros(len(p_idx))
        t_rate = np.zeros(len(p_idx))

    # Métriques avancées / NHL Edge (avec valeurs par défaut neutres si non disponibles)
    shots_pg = np.nan_to_num(panel.shots_per_gp[p_idx, s]) if panel.shots_per_gp is not None else np.zeros(len(p_idx))
    corsi = np.nan_to_num(panel.corsi_pct[p_idx, s], nan=0.50) if panel.corsi_pct is not None else np.full(len(p_idx), 0.50)
    oz_start = np.nan_to_num(panel.oz_start_pct[p_idx, s], nan=0.50) if panel.oz_start_pct is not None else np.full(len(p_idx), 0.50)
    sh_pct = np.nan_to_num(panel.sh_pct_5v5[p_idx, s], nan=0.09) if panel.sh_pct_5v5 is not None else np.full(len(p_idx), 0.09)
    on_ice_sh = np.nan_to_num(panel.on_ice_sh_pct[p_idx, s], nan=0.09) if panel.on_ice_sh_pct is not None else np.full(len(p_idx), 0.09)
    pdo_val = np.nan_to_num(panel.pdo[p_idx, s], nan=1.0) if panel.pdo is not None else np.full(len(p_idx), 1.0)
    ipp_per60 = np.nan_to_num(panel.ipp5v5_per60[p_idx, s]) if panel.ipp5v5_per60 is not None else np.zeros(len(p_idx))
    g_per60 = np.nan_to_num(panel.g_per60_5v5[p_idx, s]) if panel.g_per60_5v5 is not None else np.zeros(len(p_idx))
    a1_pg = np.nan_to_num(panel.a1_per_gp[p_idx, s]) if panel.a1_per_gp is not None else np.zeros(len(p_idx))
    toi5v5 = np.nan_to_num(panel.toi_5v5[p_idx, s]) if panel.toi_5v5 is not None else np.zeros(len(p_idx))
    take = np.nan_to_num(panel.take_per60[p_idx, s]) if panel.take_per60 is not None else np.zeros(len(p_idx))
    net_pen = np.nan_to_num(panel.net_pen_per60[p_idx, s]) if panel.net_pen_per60 is not None else np.zeros(len(p_idx))

    # Signaux dérivés : chance de tir (écart Sh% vs On-Ice Sh%) et ratio de passes primaires
    sh_luck = sh_pct - on_ice_sh
    a1_ratio = np.where(rate > 0, a1_pg / np.maximum(rate, 0.01), 0.5)

    return np.column_stack([
        toi, pp, age, gp, rate, g_rate, a_rate, t_toi, t_rate,
        shots_pg, corsi, oz_start, sh_pct, pdo_val, ipp_per60,
        g_per60, a1_pg, toi5v5, take, net_pen, sh_luck, a1_ratio
    ])


def fit_priors_panel(panel: Panel, t: int, feats: str, comp: np.ndarray) -> dict[str, dict]:
    """
    Taux de la saison s+1 prédit par les données de la saison s (paires jusqu'à t incluse).
    Supporte :
      - 'boost' : Modèle de Gradient Boosting (HistGradientBoostingRegressor) pour patineurs.
      - 'toi' / 'toi_pp' : Régression ridge linéaire pondérée.
    Les saisons cibles incomplètes sont exclues.
    """
    cols = FEATS.get(feats, [])
    out = {}
    for g in GROUPS:
        mk = panel.grp == g
        thr = MIN_GP["G"] if g == "G" else 25
        ys, ws, toi_l, pp_l, Xs = [], [], [], [], []
        for s in range(0, t):
            if not comp[s + 1]:
                continue
            ok = (panel.gp[mk, s] >= thr) & (panel.gp[mk, s + 1] >= 5)
            p_idx = np.where(mk)[0][ok]
            if len(p_idx) == 0:
                continue
            ys.append(panel.rate[p_idx, s + 1])
            ws.append(panel.gp[p_idx, s + 1])
            toi_l.append(panel.toi[p_idx, s])
            pp_l.append(panel.pp[p_idx, s])
            if feats == "boost" and g != "G":
                Xs.append(_extract_skater_features(panel, p_idx, s))
        y = np.concatenate(ys) if ys else np.array([])
        w = np.concatenate(ws) if ws else np.array([])
        res = {
            "mean": float(np.average(y, weights=w)) if len(y) else (2.5 if g == "G" else 0.4),
            "cols": cols,
            "beta": None,
            "model": None,
            "feats": feats,
        }
        if feats == "boost" and g != "G" and len(y) >= 60 and Xs:
            X_all = np.vstack(Xs)
            if _HAS_LIGHTGBM:
                model = LGBMRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.08,
                    reg_lambda=3.0,
                    min_child_samples=15,
                    verbosity=-1,
                    random_state=42,
                )
                model.fit(X_all, y, sample_weight=w)
                res["model"] = model
                imp = model.feature_importances_.astype(float)
                res["importances"] = dict(zip(BOOST_FEATURES, (imp / max(imp.sum(), 1e-9) * 100).round(1)))
            else:
                hgb = HistGradientBoostingRegressor(
                    max_iter=80,
                    max_depth=3,
                    min_samples_leaf=15,
                    learning_rate=0.08,
                    l2_regularization=3.0,
                    random_state=42,
                )
                hgb.fit(X_all, y, sample_weight=w)
                res["model"] = hgb
        elif cols and g != "G" and len(y) >= 60:
            toi, pp = np.concatenate(toi_l), np.concatenate(pp_l)
            keep = np.isfinite(toi)
            if keep.sum() >= 60:
                X = _feat_matrix(toi[keep], pp[keep], cols)
                mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
                sd[sd == 0] = 1.0
                Xs = np.nan_to_num((X - mu) / sd)
                A = np.column_stack([np.ones(len(Xs)), Xs])
                W = w[keep]
                reg = np.eye(A.shape[1]) * 1.0
                reg[0, 0] = 0.0
                beta = np.linalg.solve(A.T @ (A * W[:, None]) + reg, A.T @ (W * y[keep]))
                res.update(mu=mu, sd=sd, beta=beta)
        out[g] = res
    return out


def predict_prior(
    pr: dict,
    toi: np.ndarray,
    pp: np.ndarray,
    X_boost: np.ndarray | None = None,
) -> np.ndarray:
    if pr.get("model") is not None and X_boost is not None:
        preds = pr["model"].predict(X_boost)
        return np.maximum(preds, 0.0)
    if pr.get("beta") is None:
        return np.full(len(toi), pr["mean"])
    X = _feat_matrix(toi, pp, pr["cols"])
    Xs = np.nan_to_num((X - pr["mu"]) / pr["sd"])
    return np.maximum(pr["beta"][0] + Xs @ pr["beta"][1:], 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# PROJECTION VECTORISÉE
# ──────────────────────────────────────────────────────────────────────────────
def _nwavg(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = np.isfinite(x) & (w > 0)
    num = np.where(m, np.nan_to_num(x) * w, 0.0).sum(1)
    den = np.where(m, w, 0.0).sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def forecast(panel: Panel, t: int, par: ModelParams, curves: dict, rets: dict, priors: dict,
             kmax: int = MAX_YEARS, pre_nhl: dict[int, dict[str, Any]] | None = None) -> dict[str, np.ndarray]:
    """
    Projette les points de pool pour les saisons t+k (k = 0..kmax) avec les seules données ≤ t.
    Méthode : moyenne pondérée des taux historiques (ajustés à l'âge actuel, décroissance temporelle)
    régressée vers un prior d'utilisation (TOI), × disponibilité × courbe d'âge future.
    Intègre les statistiques pré-LNH (junior, AHL, Europe) pour calibrer l'évidence des jeunes joueurs.
    Retourne aussi la probabilité de survie S_k (rétention LNH).
    """
    sl = slice(0, t + 1)
    n = panel.gp.shape[0]
    gp, rate = panel.gp[:, sl], panel.rate[:, sl]
    yrs, slen = panel.yrs[sl], panel.slen[sl]
    wdec = par.decay ** (yrs[t] - yrs)
    w = gp * wdec[None, :]
    has = gp > 0
    first = np.where(has.any(1), has.argmax(1), t + 1)
    after = np.arange(t + 1)[None, :] >= first[:, None]
    nhl_seasons_count = has.sum(axis=1)
    tot_nhl_gp_all = gp.sum(axis=1)
    age_o = panel.age[:, t]

    # Ratio des matchs joués / longueur de saison
    gp_ratio = np.minimum(1.0, gp / np.maximum(slen[None, :], 1.0))
    # Neutralisation de la fausse fragilité chez les jeunes patineurs
    for i in range(n):
        if age_o[i] < 24.5 or tot_nhl_gp_all[i] < 120:
            for s_idx in range(t + 1):
                if gp[i, s_idx] < 25 or s_idx <= first[i] + 1:
                    gp_ratio[i, s_idx] = np.maximum(gp_ratio[i, s_idx], 0.96)

    ws = wdec[None, :] * np.minimum(1.0, slen / GAMES)[None, :] * after
    avn = (ws * gp_ratio).sum(1)
    avd = ws.sum(1)
    lo = max(0, t - 1)
    toi_w = _nwavg(panel.toi[:, lo:t + 1], w[:, lo:t + 1])
    pp_w = _nwavg(panel.pp[:, lo:t + 1], w[:, lo:t + 1])
    ks = np.arange(0, kmax + 1)
    ages = age_o[:, None] + ks[None, :]

    pre_rates = np.array([pre_nhl.get(int(p), {}).get("pre_rate", 0.0) if pre_nhl else 0.0 for p in panel.pids])
    pre_gps = np.array([pre_nhl.get(int(p), {}).get("pre_gp_eq", 0.0) if pre_nhl else 0.0 for p in panel.pids])
    pre_summaries = np.array([pre_nhl.get(int(p), {}).get("pre_summary", "") if pre_nhl else "" for p in panel.pids])

    rate0, den_all, prior_all, avail = (np.full(n, np.nan) for _ in range(4))
    P, S = np.zeros((n, kmax + 1)), np.zeros((n, kmax + 1))
    P_pace, A = np.zeros((n, kmax + 1)), np.zeros((n, kmax + 1))

    # Facteur d'ajustement selon l'ère offensive de la ligue
    lg_gpg = getattr(panel, "lg_gpg", None)
    if lg_gpg is not None and len(lg_gpg) > t and lg_gpg[t] > 0:
        era_adj = lg_gpg[t] / np.maximum(lg_gpg[:t + 1], 1e-6)
    else:
        era_adj = np.ones(t + 1)

    for g in GROUPS:
        mk = panel.grp == g
        if not mk.any():
            continue
        f = curves[g]
        f_now = np.interp(age_o[mk], AGES, f)
        f_s = np.interp(panel.age[mk][:, sl], AGES, f)
        era_scale = era_adj[None, :] if g != "G" else np.ones((1, t + 1))
        adj = (f_now[:, None] / f_s) * era_scale
        num = (w[mk] * rate[mk] * adj).sum(1)
        den = w[mk].sum(1)
        p_idx = np.where(mk)[0]
        X_boost = (
            _extract_skater_features(panel, p_idx, t, toi_w[mk], pp_w[mk])
            if (g != "G" and priors[g].get("model") is not None)
            else None
        )
        prior_toi = predict_prior(priors[g], toi_w[mk], pp_w[mk], X_boost)
        K = K_PRIOR[g] * par.k_mult

        p_gp = pre_gps[mk]
        p_rate = pre_rates[mk]
        has_pre = (p_gp > 0) & (g != "G")

        tot_gp_mk = tot_nhl_gp_all[mk]
        emp_rate = num / np.maximum(den, 1e-6)
        is_young = (age_o[mk] < 24.5) | (tot_gp_mk < 120)

        taper = np.maximum(0.0, 1.0 - tot_gp_mk / 50.0)
        eff_p_gp = p_gp * taper

        pedigree_prior = np.where(has_pre & (p_rate > prior_toi), p_rate, prior_toi)
        demonstrated_floor = np.where(is_young & (den >= 6) & (emp_rate > pedigree_prior),
                                      0.88 * emp_rate,
                                      pedigree_prior)

        is_prime_established = (age_o[mk] >= 24.0) & (age_o[mk] <= 31.5) & (den >= 20)
        prime_floor = np.where(is_prime_established & (emp_rate > prior_toi),
                               0.95 * emp_rate,
                               prior_toi)
        prior = np.where(is_young, np.maximum(pedigree_prior, demonstrated_floor),
                         np.where(is_prime_established, np.maximum(prior_toi, prime_floor), prior_toi))

        K_eff = np.where(is_young & has_pre,
                         np.maximum(4.0, K * (1.0 - 0.4 * taper)),
                         np.where(is_prime_established & (emp_rate > prior_toi),
                                  np.maximum(8.0, K * 0.50),
                                  K))

        use_pre_emp = has_pre & (eff_p_gp > 0) & ((den < 10) | (emp_rate < p_rate))
        den_eff = den + np.where(use_pre_emp, eff_p_gp, 0.0)
        num_eff = num + np.where(use_pre_emp, eff_p_gp * p_rate, 0.0)

        r0 = (num_eff + K_eff * prior) / (den_eff + K_eff)
        av_p, K_av = AV_PRIOR[g]

        is_young_or_prospect = is_young | has_pre
        K_av_eff = np.where(is_young_or_prospect, np.maximum(K_av, 3.5), K_av)
        av = (avn[mk] + K_av_eff * av_p) / (avd[mk] + K_av_eff)
        if g != "G":
            av = np.where(is_young_or_prospect, np.maximum(av, 0.95), av)
            av = np.where(is_prime_established, np.maximum(av, 0.93), av)
        av = np.minimum(av, AV_CAP[g])
        
        rate0[mk], den_all[mk], prior_all[mk], avail[mk] = r0, den_eff, prior, av
        rate_k = r0[:, None] * np.interp(ages[mk], AGES, f) / f_now[:, None]
        av_k = av[:, None] * (1 - 0.010 * np.maximum(0, ages[mk] - 31.5))
        
        # Séparation du rythme et de la disponibilité
        P_pace[mk] = rate_k * GAMES * par.cal[g]
        A[mk] = av_k
        P[mk] = P_pace[mk] * av_k
        
        S[mk, 0] = 1.0
        elite_mult = np.where(is_prime_established & (r0 >= 0.75), 1.04, 1.0)
        rets_eff = np.minimum(0.99, np.interp(ages[mk][:, :-1], AGES, rets[g]) * elite_mult[:, None])
        S[mk, 1:] = np.cumprod(rets_eff, axis=1)
        
    return dict(P=P, P_pace=P_pace, A=A, S=S, rate0=rate0, avail=avail, den=den_all, prior=prior_all,
                toi_w=toi_w, pp_w=pp_w, age_o=age_o,
                pre_rate=pre_rates, pre_gp_eq=pre_gps, pre_summary=pre_summaries)


def sd_matrix(par: ModelParams, grp: np.ndarray, den: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Écart-type prédictif (pts) = c(position, évidence) × √pts × croissance^(k-1)."""
    gi = np.array([GROUPS.index(g) for g in grp])
    bi = np.digitize(np.nan_to_num(den), INFO_BINS)
    tab = np.array([[par.sd_c[(g, b)] for b in range(3)] for g in GROUPS])
    c = tab[gi, bi]
    ks = np.arange(P.shape[1])
    return c[:, None] * np.sqrt(np.maximum(P, 5.0)) * (par.sd_growth ** (ks - 1.0))[None, :]


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST WALK-FORWARD, GRILLE ET CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────
def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return np.nan
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _build_folds(ps: pd.DataFrame, panel: Panel, comp: np.ndarray) -> list[dict]:
    T = len(panel.seasons)
    folds = []
    for t in range(1, T - 1):
        if not (comp[t] and comp[t + 1]):
            continue
        curves, rets = build_curves(ps[ps["season"] <= panel.seasons[t]], panel.seasons[t])
        pri = {fn: fit_priors_panel(panel, t, fn, comp) for fn in FEATS}
        folds.append(dict(t=t, curves=curves, rets=rets, priors=pri))
    return folds


def _evaluate(panel: Panel, folds: list[dict], par: ModelParams, comp: np.ndarray, ks=(1, 2)) -> pd.DataFrame:
    """Prédit chaque fold avec `par` et compare aux points réels (ramenés à 82 matchs)."""
    T = len(panel.seasons)
    pts82 = panel.pool / panel.slen[None, :] * GAMES
    min_gp = np.where(panel.grp == "G", MIN_GP["G"], MIN_GP["F"])
    played_thr = np.where(panel.grp == "G", 5, 10)
    out = []
    for fd in folds:
        t = fd["t"]
        fc = forecast(panel, t, par, fd["curves"], fd["rets"], fd["priors"][par.feats], kmax=2)
        elig = panel.gp[:, t] >= min_gp
        lo = max(0, t - 2)
        hasl = panel.gp[:, lo:t + 1] > 0
        naive2 = (pts82[:, lo:t + 1] * hasl).sum(1) / np.maximum(hasl.sum(1), 1)
        sd_all = sd_matrix(par, panel.grp, fc["den"], fc["P"])
        trend = np.where((panel.gp[:, t] >= 20) & (panel.gp[:, t - 1] >= 20),
                         panel.toi[:, t] - panel.toi[:, t - 1], np.nan)
        for k in ks:
            j = t + k
            if j >= T or not comp[j]:
                continue
            out.append(pd.DataFrame({
                "fold": t, "k": k, "grp": panel.grp, "den": fc["den"],
                "P": fc["P"][:, k], "E": fc["P"][:, k] * fc["S"][:, k], "sd": sd_all[:, k],
                "actual": pts82[:, j], "played": panel.gp[:, j] >= played_thr,
                "naive1": pts82[:, t], "naive2": naive2,
                "toi_trend": trend, "pp": panel.pp[:, t],
            })[elig])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in sorted(df["k"].unique()):
        dk = df[df["k"] == k]
        for g in list(GROUPS) + ["Tous"]:
            d = dk if g == "Tous" else dk[dk["grp"] == g]
            if len(d) < 10:
                continue
            pl = d[d["played"]]
            cover = float(((pl["actual"] - pl["P"]).abs() <= Z80 * pl["sd"]).mean()) if len(pl) else np.nan
            m_mod, m_n1, m_n2 = _mae(d["E"], d["actual"]), _mae(d["naive1"], d["actual"]), _mae(d["naive2"], d["actual"])
            rows.append({
                "Horizon (ans)": int(k), "Position": GRP_NAME.get(g, g), "N": len(d),
                "MAE modèle (pts)": round(m_mod, 2), "MAE naïf 1 an": round(m_n1, 2), "MAE naïf moy. 3 ans": round(m_n2, 2),
                "Gain vs meilleur naïf": f"{(1 - m_mod / min(m_n1, m_n2)) * 100:+.1f} %",
                "Corr. rangs modèle": round(_rank_corr(d["E"].values, d["actual"].values), 3),
                "Corr. rangs naïf 1 an": round(_rank_corr(d["naive1"].values, d["actual"].values), 3),
                "Biais (pts)": round(float((d["E"] - d["actual"]).mean()), 2),
                "Couverture IC 80 %": round(cover, 3) if np.isfinite(cover) else np.nan,
            })
    return pd.DataFrame(rows)


def _calibrate_impl(data: dict[str, Any]) -> dict[str, Any]:
    ps = prepare_long(data)
    panel = make_panel(ps)
    cur = data["cur"]
    comp = _complete_flags(panel, cur)
    folds = _build_folds(ps, panel, comp)
    empty = dict(params=ModelParams(), backtest=pd.DataFrame(), grid=pd.DataFrame(), signals=pd.DataFrame(),
                 n_folds=0)
    if not folds:
        return empty

    # 1) Grille : minimise le MAE de l'espérance E = S × P à l'horizon 1 (échantillon complet, absents inclus)
    res = []
    for decay, km, ft in product(GRID_DECAY, GRID_KMULT, FEATS):
        par = ModelParams(decay=decay, k_mult=km, feats=ft)
        ev = _evaluate(panel, folds, par, comp, ks=(1,))
        if len(ev):
            res.append(dict(decay=decay, k_mult=km, feats=ft, MAE=_mae(ev["E"], ev["actual"]), N=len(ev)))
    if not res:
        return empty
    grid = pd.DataFrame(res).sort_values("MAE").reset_index(drop=True)
    b = grid.iloc[0]
    par = ModelParams(decay=float(b["decay"]), k_mult=float(b["k_mult"]), feats=str(b["feats"]))

    # 2) Facteur de biais par position (Σ réel / Σ espéré)
    ev = _evaluate(panel, folds, par, comp, ks=(1,))
    for g in GROUPS:
        d = ev[ev["grp"] == g]
        if len(d) >= 30 and d["E"].sum() > 0:
            par.cal[g] = float(np.clip(d["actual"].sum() / d["E"].sum(), 0.95, 1.05))

    # 3) Écarts-types prédictifs calibrés (par position × quantité d'information)
    ev = _evaluate(panel, folds, par, comp, ks=(1, 2))

    def c_of(d: pd.DataFrame) -> float:
        d = d[d["played"]]
        return float(np.sqrt(((d["actual"] - d["P"]) ** 2).sum() / np.maximum(d["P"], 5.0).sum())) if len(d) else np.nan

    e1 = ev[ev["k"] == 1].copy()
    e1["bin"] = np.digitize(np.nan_to_num(e1["den"]), INFO_BINS)
    for g in GROUPS:
        dg = e1[e1["grp"] == g]
        c_g = c_of(dg) if (dg["played"].sum() >= MIN_PAIRS_C) else np.nan
        for bi in range(3):
            db = dg[dg["bin"] == bi]
            c = c_of(db) if db["played"].sum() >= MIN_PAIRS_C else c_g
            par.sd_c[(g, bi)] = float(np.clip(c, 0.8, 4.0)) if np.isfinite(c) else DEFAULT_C[(g, bi)]
    e2 = ev[ev["k"] == 2]
    if len(e1) and len(e2):
        c1, c2 = c_of(e1), c_of(e2)
        if np.isfinite(c1) and np.isfinite(c2) and c1 > 0:
            par.sd_growth = float(np.clip(c2 / c1, 1.0, 1.4))
    par.calibrated = True

    # 4) Rapport final (échantillon de calibration : légèrement optimiste, à lire comme un ordre de grandeur)
    ev = _evaluate(panel, folds, par, comp, ks=(1, 2))
    summary = _summarize(ev)

    # 5) Les signaux « opportunité » ajoutent-ils quelque chose au-delà du modèle ?
    sig = []
    d = ev[(ev["k"] == 1) & ev["played"] & (ev["grp"] != "G")].copy()
    d["resid"] = d["actual"] - d["E"]
    for label, col in (("Tendance du TOI (an t vs t-1)", "toi_trend"), ("TOI en avantage numérique", "pp")):
        dd = d[np.isfinite(d[col])]
        rho = _rank_corr(dd["resid"].values, dd[col].values)
        sig.append({"Variable": label, "N": len(dd), "Corr. avec l'erreur du modèle": round(rho, 3) if np.isfinite(rho) else np.nan,
                    "Lecture": "signal résiduel exploitable" if np.isfinite(rho) and abs(rho) >= 0.10
                    else "pas de valeur ajoutée démontrée"})
    importances = {}
    last_pri = folds[-1]["priors"].get("boost", {}) if folds else {}
    for g in ("F", "D"):
        if "importances" in last_pri.get(g, {}):
            importances[GRP_NAME[g]] = last_pri[g]["importances"]

    return dict(
        params=par,
        backtest=summary,
        grid=grid.head(8),
        signals=pd.DataFrame(sig),
        importances=importances,
        n_folds=len(folds),
    )


@st.cache_data(show_spinner=False)
def run_calibration(data: dict[str, Any]) -> dict[str, Any]:
    """Calibration par backtest (mise en cache : ne dépend que des données, pas des curseurs d'interface)."""
    return _calibrate_impl(data)


# ──────────────────────────────────────────────────────────────────────────────
# VALEUR : VORP ESPÉRÉ, REMPLACEMENT SIMULÉ, DISTRIBUTION DE LA VALEUR KEEPER
# ──────────────────────────────────────────────────────────────────────────────
_erf = np.vectorize(math.erf, otypes=[float])


def _cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def exp_excess(mu: np.ndarray, sd: np.ndarray, r: np.ndarray) -> np.ndarray:
    """E[max(0, X − r)] pour X ~ N(mu, sd) (valeur d'option : un joueur incertain vaut plus que sa moyenne près du remplacement)."""
    sd = np.maximum(sd, 1e-6)
    d = (mu - r) / sd
    return sd * _pdf(d) + (mu - r) * _cdf(d)


def simulated_replacement(mu: np.ndarray, sd: np.ndarray, rank: int, rng: np.random.Generator, nsim: int = 300) -> float:
    """Niveau de remplacement = rang `rank` des RÉSULTATS simulés (la dispersion réelle est plus grande que celle des projections)."""
    n = len(mu)
    if n == 0:
        return 0.0
    draws = np.maximum(0.0, mu[None, :] + sd[None, :] * rng.standard_normal((nsim, n)))
    kth = min(rank, n)
    return float((-np.partition(-draws, kth - 1, axis=1)[:, kth - 1]).mean())


def kv_distribution(P_pace: np.ndarray, A: np.ndarray, S: np.ndarray, SD_pace: np.ndarray, r_vec: np.ndarray, h: int, disc: np.ndarray,
                    rng: np.random.Generator, nsim: int = 200) -> np.ndarray:
    """Simulation de la valeur keeper avec VORP sur le rythme (pace) pondéré par la disponibilité."""
    n = P_pace.shape[0]
    z0 = rng.standard_normal((nsim, n, 1))
    zk = rng.standard_normal((nsim, n, h))
    
    # Dispersion du talent autour du rythme 82 matchs
    X_pace = P_pace[None, :, 1:h + 1] + SD_pace[None, :, 1:h + 1] * (RHO_TALENT * z0 + math.sqrt(1 - RHO_TALENT ** 2) * zk)
    
    # Le VORP est l'excédent de rythme multiplié par les matchs réellement joués (A)
    V = np.maximum(0.0, X_pace - r_vec[None, :, None]) * A[None, :, 1:h + 1]
    return (disc[None, None, :] * S[None, :, 1:h + 1] * V).sum(axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS D'OPPORTUNITÉ (informatifs ; non utilisés dans le Score)
# ──────────────────────────────────────────────────────────────────────────────
def add_team_context(df: pd.DataFrame, ps: pd.DataFrame, cur: int) -> pd.DataFrame:
    """Concurrence interne, contexte offensif d'équipe et progression des buts."""
    df = df.copy()
    df["role"] = df["pos"].map({"C": "C", "L": "W", "R": "W", "D": "D"})
    active = df[(df["grp"] != "G") & (df["gp_cur"] > 0)]
    groups = {k: v for k, v in active.groupby(["team", "role"])}
    n_ahead, age_ahead, tm_age, gap = [], [], [], []
    for r in df.itertuples():
        if r.grp == "G":
            n_ahead.append(np.nan); age_ahead.append(np.nan); tm_age.append(np.nan); gap.append(np.nan)
            continue
        gdf = groups.get((r.team, r.role))
        others = gdf[gdf["pid"] != r.pid] if gdf is not None else pd.DataFrame(columns=df.columns)
        tm_age.append(others["age_cur"].mean() if len(others) else np.nan)
        gap.append(r.toi_w - others["toi_w"].mean() if len(others) and np.isfinite(r.toi_w) else np.nan)
        ahead = others[others["toi_w"] > r.toi_w] if np.isfinite(r.toi_w) else others.iloc[0:0]
        n_ahead.append(len(ahead))
        age_ahead.append(ahead["age_cur"].mean() if len(ahead) else np.nan)
    df["tm_age"], df["tm_gap"], df["n_ahead"], df["age_ahead"] = tm_age, gap, n_ahead, age_ahead

    # Contexte offensif d'équipe (moyenne de buts marqués et progression d'une saison à l'autre)
    prev_s = season_shift(cur, -1)
    sk_ps = ps[ps["grp"] != "G"]
    tm_cur = sk_ps[sk_ps["season"] == cur].groupby("team").agg(g=("g", "sum"), gp=("gp", "max"))
    tm_prev = sk_ps[sk_ps["season"] == prev_s].groupby("team").agg(g=("g", "sum"), gp=("gp", "max"))

    gpg_cur_dict = (tm_cur["g"] / tm_cur["gp"].clip(lower=1)).to_dict()
    gpg_prev_dict = (tm_prev["g"] / tm_prev["gp"].clip(lower=1)).to_dict()
    gp_cur_dict = tm_cur["gp"].to_dict()

    # Statistiques individuelles de buts (saison courante et précédente)
    sk_c = sk_ps[sk_ps["season"] == cur].set_index("pid")
    sk_p = sk_ps[sk_ps["season"] == prev_s].set_index("pid")

    g_c_map = sk_c["g"].to_dict()
    gp_c_map = sk_c["gp"].to_dict()
    g_p_map = sk_p["g"].to_dict()
    gp_p_map = sk_p["gp"].to_dict()

    tm_gpg_l, tm_trend_l, tm_pace_goals_l, tm_ceil_l = [], [], [], []
    g_pace82_l, g_trend_l, g_rate_cur_l = [], [], []

    for r in df.itertuples():
        tm = str(r.team)
        c_gpg = gpg_cur_dict.get(tm, 3.1)
        p_gpg = gpg_prev_dict.get(tm, c_gpg)
        tm_trend = c_gpg - p_gpg

        # Stabilisation de l'échantillon précoce (w_eff)
        tm_gp = gp_cur_dict.get(tm, 15)
        w_eff = min(1.0, tm_gp / 40.0)
        eff_gpg = w_eff * c_gpg + (1.0 - w_eff) * p_gpg
        tm_pace_goals = eff_gpg * GAMES

        # Plafond réaliste d'implication offensive
        is_f = r.grp == "F"
        tm_ceil = tm_pace_goals * (0.52 if is_f else 0.40)

        tm_gpg_l.append(c_gpg)
        tm_trend_l.append(tm_trend)
        tm_pace_goals_l.append(tm_pace_goals)
        tm_ceil_l.append(tm_ceil if r.grp != "G" else np.nan)

        # Buts individuels (skaters)
        if r.grp != "G":
            gc = g_c_map.get(r.pid, 0.0)
            gpc = max(gp_c_map.get(r.pid, 0.0), 1.0) if r.pid in gp_c_map else 0.0
            gr_c = gc / gpc if gpc > 0 else 0.0

            gp = g_p_map.get(r.pid, 0.0)
            gpp = max(gp_p_map.get(r.pid, 0.0), 1.0) if r.pid in gp_p_map else 0.0
            gr_p = gp / gpp if gpp > 0 else gr_c

            diff_g = gr_c - gr_p if gpc >= 5 and gpp >= 10 else 0.0
            g_pace = gr_c * GAMES

            g_rate_cur_l.append(gr_c)
            g_trend_l.append(diff_g)
            g_pace82_l.append(g_pace)
        else:
            g_rate_cur_l.append(np.nan)
            g_trend_l.append(np.nan)
            g_pace82_l.append(np.nan)

    df["tm_gpg"] = tm_gpg_l
    df["tm_trend"] = tm_trend_l
    df["tm_pace_goals"] = tm_pace_goals_l
    df["tm_ceil"] = tm_ceil_l
    df["g_rate_cur"] = g_rate_cur_l
    df["g_trend"] = g_trend_l
    df["g_pace82"] = g_pace82_l

    return df


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Rang centile (0-100) au sein de la position."""
    return (df.groupby("grp")[col].rank(pct=True) * 100).fillna(50.0)


# ──────────────────────────────────────────────────────────────────────────────
# MODÈLE COMPLET
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_model(
    data: dict[str, Any],
    n_teams: int,
    bench: dict[str, int],
    horizon: int,
    discount: float,
    decay: float,
    auto_calibrate: bool = True,
    method: str = "boost",
) -> dict[str, Any]:
    """
    Projections, VORP espéré sur le rythme, incertitude et rareté.
    """
    cur = data["cur"]
    ps = prepare_long(data)
    panel = make_panel(ps)
    T = len(panel.seasons)
    t = T - 1
    comp = _complete_flags(panel, cur)

    if auto_calibrate:
        calib = run_calibration(data)
    else:
        calib = dict(params=ModelParams(decay=decay, feats="boost"), backtest=pd.DataFrame(), grid=pd.DataFrame(),
                     signals=pd.DataFrame(), n_folds=0)
    par: ModelParams = calib["params"]
    if method in ("boost", "toi", "toi_pp", "none"):
        par = replace(par, feats=method)

    curves, rets = build_curves(ps, cur)
    priors = fit_priors_panel(panel, t, par.feats, comp)
    fc = forecast(panel, t, par, curves, rets, priors, kmax=MAX_YEARS, pre_nhl=data.get("pre_nhl"))

    prev_ok = (panel.gp[:, t - 1] > 0) if t >= 1 else np.zeros(len(panel.pids), bool)
    idx = np.where((panel.gp[:, t] > 0) | prev_ok)[0]
    g_arr = panel.grp[idx]
    
    # Extraction des matrices avec le Pace
    P, P_pace, A, S, den = fc["P"][idx], fc["P_pace"][idx], fc["A"][idx], fc["S"][idx], fc["den"][idx]
    
    # L'écart-type prédictif doit mesurer la variance du RYTHME, mais on garde aussi SD total 
    # pour affichage / constance avec l'interface.
    SD = sd_matrix(par, g_arr, den, P)
    SD_pace = sd_matrix(par, g_arr, den, P_pace)

    # ── Rareté et remplacement (calibrés sur le PACE 82 matchs) ────────────────────────
    rng = np.random.default_rng(2026)
    repl, ginfo = {}, []
    for g in GROUPS:
        m = g_arr == g
        vals = np.sort(P_pace[m, 0])[::-1]
        R = n_teams * (SLOTS[g] + bench[g])
        repl[g] = simulated_replacement(P_pace[m, 0], SD_pace[m, 0], R, rng)
        starters = vals[:n_teams * SLOTS[g]]
        ginfo.append({
            "Position": GRP_NAME[g],
            "Postes de départ (ligue)": n_teams * SLOTS[g],
            "Rang de remplacement": R,
            "Pts remplacement (Pace)": round(repl[g], 1),
            "Moyenne partants (Pace)": round(float(starters.mean()), 1) if len(starters) else np.nan,
            "Écart-type partants": round(float(starters.std()), 1) if len(starters) else np.nan,
            "Moy. top 10 (Pace)": round(float(vals[:10].mean()), 1) if len(vals) else np.nan,
            "Écart élite – remplacement": round(float(vals[:10].mean() - repl[g]), 1) if len(vals) else np.nan,
        })
    ginfo = pd.DataFrame(ginfo)
    r_vec = np.array([repl[g] for g in g_arr])

    # ── Valeur espérée au-dessus du remplacement (VRAI VORP) ─────────────────────────────
    V_pace = exp_excess(P_pace, SD_pace, r_vec[:, None])
    V = V_pace * A 
    
    h = int(horizon)
    disc = discount ** np.arange(0, h)
    kv = (disc[None, :] * S[:, 1:h + 1] * V[:, 1:h + 1]).sum(axis=1)
    
    p_above = 1.0 - _cdf((r_vec[:, None] - P_pace) / np.maximum(SD_pace, 1e-6))
    useful = (S[:, 1:] * p_above[:, 1:]).sum(axis=1)
    kvs = kv_distribution(P_pace, A, S, SD_pace, r_vec, h, disc, rng)

    # ── Table joueurs ─────────────────────────────────────────────────────────
    lo = max(0, t - 1)
    is_sk = g_arr != "G"
    pool_lo = panel.pool[idx][:, lo:]
    gp_lo = panel.gp[idx][:, lo:]
    toi_lo = panel.toi[idx][:, lo:]
    denom = (gp_lo * np.nan_to_num(toi_lo)).sum(1)
    p60 = np.where(is_sk & (denom > 0), pool_lo.sum(1) / np.maximum(denom, 1e-9) * 60, np.nan)
    slen_lo = np.where(gp_lo > 0, panel.slen[lo:][None, :], 0.0).sum(1)
    gs_share = np.where(~is_sk, np.nan_to_num(panel.gs[idx][:, lo:]).sum(1) / np.maximum(slen_lo, 1), np.nan)
    gp_sum = gp_lo.sum(1)
    win_rate = np.where(~is_sk & (gp_sum > 0), np.nan_to_num(panel.w[idx][:, lo:]).sum(1) / np.maximum(gp_sum, 1), np.nan)
    toi_cur = np.where(panel.gp[idx, t] >= 8, panel.toi[idx, t], np.nan)
    if t >= 1:
        toi_prev = np.where(panel.gp[idx, t - 1] >= 20, panel.toi[idx, t - 1], np.nan)
    else:
        toi_prev = np.full(len(idx), np.nan)
    slen_cur = panel.slen[t]
    rem = max(0.0, GAMES - slen_cur) if slen_cur else float(GAMES)

    df = pd.DataFrame({
        "pid": panel.pids[idx], "name": panel.name[idx], "pos": panel.pos[idx], "grp": g_arr, "team": panel.team[idx],
        "age_cur": fc["age_o"][idx], "gp_cur": panel.gp[idx, t], "pool_cur": panel.pool[idx, t],
        "rate0": fc["rate0"][idx], "avail": fc["avail"][idx], "ros": P[:, 0] * rem / GAMES, "evid": den,
        "toi_w": fc["toi_w"][idx], "pp_w": fc["pp_w"][idx], "p60": p60, "toi_trend": toi_cur - toi_prev,
        "gs_share": gs_share, "win_rate": win_rate, "prior": fc["prior"][idx],
        "pre_rate": fc["pre_rate"][idx], "pre_gp_eq": fc["pre_gp_eq"][idx], "pre_nhl": fc["pre_summary"][idx],
    })
    df["pts0"], df["sd0"] = P[:, 0], SD[:, 0]
    df["pace82"] = fc["rate0"][idx] * GAMES
    df["vorp0"] = V[:, 0]
    df["peak_pts"] = P.max(axis=1)
    df["peak_age"] = fc["age_o"][idx] + P.argmax(axis=1)
    df["kv"] = kv
    df["kv_sd"] = kvs.std(axis=0)
    df["kv_p10"], df["kv_p90"] = np.percentile(kvs, 10, axis=0), np.percentile(kvs, 90, axis=0)
    df["useful"] = useful
    df["repl"] = r_vec
    df["disc_sum"] = float(disc.sum())
    df["conf"] = np.select([den >= INFO_BINS[1], den >= INFO_BINS[0]], ["Élevée", "Moyenne"], default="Faible")
    
    gi = ginfo.set_index("Position")
    st_mean = {g: gi.loc[GRP_NAME[g], "Moyenne partants (Pace)"] for g in GROUPS}
    st_std = {g: gi.loc[GRP_NAME[g], "Écart-type partants"] for g in GROUPS}

    # ── Diagnostics d'opportunité et contexte d'équipe ───────────────────────
    df = add_team_context(df, ps, cur)

    # ── Balisage réaliste par le réservoir offensif de l'équipe ──
    for i in range(len(df)):
        if df.iloc[i]["grp"] != "G":
            ceil_val = df.iloc[i]["tm_ceil"]
            if np.isfinite(ceil_val) and ceil_val > 0:
                max_allowed = 1.05 * ceil_val
                if P[i, 0] > max_allowed:
                    P[i, 0] = max_allowed
                if P_pace[i, 0] > max_allowed:
                    P_pace[i, 0] = max_allowed
                if df.at[df.index[i], "pace82"] > max_allowed:
                    df.at[df.index[i], "pace82"] = max_allowed

    # Application du balisage au points totaux et au VORP recalculé
    df["pts0"] = P[:, 0]
    df["ros"] = P[:, 0] * rem / GAMES
    df["vorp0"] = (exp_excess(P_pace[:, 0:1], SD_pace[:, 0:1], r_vec[:, None]) * A[:, 0:1]).squeeze()
    df["peak_pts"] = P.max(axis=1)
    df["peak_age"] = fc["age_o"][idx] + P.argmax(axis=1)
    df["tm_share"] = np.where(df["grp"] != "G", df["pts0"] / np.maximum(df["tm_pace_goals"], 1.0), np.nan)
    df["rank_pos"] = df.groupby("grp")["pts0"].rank(ascending=False, method="first").astype(int)
    df["z_pos"] = (df["pts0"] - df["grp"].map(st_mean)) / df["grp"].map(st_std).replace(0, np.nan)

    # ── Données avancées / NHL Edge (saison courante) ───────────────────────────
    df["corsi_pct"] = panel.corsi_pct[idx, t] if panel.corsi_pct is not None else np.nan
    df["corsi_rel"] = panel.corsi_rel[idx, t] if panel.corsi_rel is not None else np.nan
    df["pdo"] = panel.pdo[idx, t] if panel.pdo is not None else np.nan
    df["oz_start_pct"] = panel.oz_start_pct[idx, t] if panel.oz_start_pct is not None else np.nan
    df["shots_per_gp"] = panel.shots_per_gp[idx, t] if panel.shots_per_gp is not None else np.nan
    df["sh_pct_5v5"] = panel.sh_pct_5v5[idx, t] if panel.sh_pct_5v5 is not None else np.nan
    df["on_ice_sh_pct"] = panel.on_ice_sh_pct[idx, t] if panel.on_ice_sh_pct is not None else np.nan
    df["ipp5v5_per60"] = panel.ipp5v5_per60[idx, t] if panel.ipp5v5_per60 is not None else np.nan
    df["g_per60_5v5"] = panel.g_per60_5v5[idx, t] if panel.g_per60_5v5 is not None else np.nan
    df["a1_per_gp"] = panel.a1_per_gp[idx, t] if panel.a1_per_gp is not None else np.nan
    df["take_per60"] = panel.take_per60[idx, t] if panel.take_per60 is not None else np.nan
    df["net_pen_per60"] = panel.net_pen_per60[idx, t] if panel.net_pen_per60 is not None else np.nan

    # ── Données de repêchage (Draft Capital) ──────────────────────────────────
    if "draft_overall" in ps.columns:
        draft_ovr_map = ps.dropna(subset=["draft_overall"]).groupby("pid")["draft_overall"].last()
        df["draft_overall"] = df["pid"].map(draft_ovr_map)
    else:
        df["draft_overall"] = np.nan

    if "draft_round" in ps.columns:
        draft_rnd_map = ps.dropna(subset=["draft_round"]).groupby("pid")["draft_round"].last()
        df["draft_round"] = df["pid"].map(draft_rnd_map)
    else:
        df["draft_round"] = np.nan

    if "draft_year" in ps.columns:
        draft_yr_map = ps.dropna(subset=["draft_year"]).groupby("pid")["draft_year"].last()
        df["draft_year"] = df["pid"].map(draft_yr_map)
    else:
        df["draft_year"] = np.nan

    # ── Bonus de capital de repêchage (Draft Capital Bias) ────────────────────
    # Plus un jeune joueur a été repêché haut, plus la franchise lui offre une laisse longue
    # et une patience organisationnelle accrue (top trios, PP). Ce biais est maximal
    # à 18-20 ans et s'estompe progressivement pour disparaître à 24 ans.
    pick_val = pd.to_numeric(df["draft_overall"], errors="coerce")
    is_drafted = pick_val.notna() & (pick_val > 0)
    draft_score = np.where(
        is_drafted,
        np.clip(100.0 * (1.0 - (np.log2(np.clip(pick_val, 1.0, 224.0)) / np.log2(224.0))), 0.0, 100.0),
        0.0,
    )
    # Décroissance avec l'âge : 100% à <=20 ans, puis baisse linéaire jusqu'à 0% à 24 ans
    age_weight = np.clip((24.0 - df["age_cur"]) / 4.0, 0.0, 1.0)
    draft_bonus = draft_score * age_weight
    df["draft_score"] = draft_score.round(1)
    df["draft_bonus"] = draft_bonus.round(1)

    sk = df["grp"] != "G"
    toi_pct, pp_pct = _pct(df, "toi_w"), _pct(df, "pp_w")
    lead_pct, trend_pct = _pct(df, "tm_gap"), _pct(df, "toi_trend")
    opening = np.where(df["age_ahead"].notna(), np.clip((df["age_ahead"] - 26) / 9, 0, 1) * 100, 60.0)
    oz_pct = _pct(df, "oz_start_pct").fillna(50.0)
    shot_vol_pct = _pct(df, "shots_per_gp").fillna(50.0)
    opp_sk = (
        0.26 * toi_pct
        + 0.16 * pp_pct
        + 0.10 * lead_pct
        + 0.10 * opening
        + 0.08 * trend_pct
        + 0.10 * oz_pct
        + 0.08 * shot_vol_pct
        + 0.12 * draft_bonus
    )
    opp_g = 0.60 * _pct(df, "gs_share") + 0.25 * _pct(df, "win_rate") + 0.15 * draft_bonus
    df["c_opp"] = np.where(sk, opp_sk, opp_g)
    df["underuse"] = np.where(sk, _pct(df, "p60") - toi_pct, np.nan)
    # Signal de régression / chance : un PDO sous 1.000 indique un rebond attendu
    luck_rebound = np.where(df["pdo"].notna(), (1.00 - df["pdo"]).clip(-0.06, 0.06) * 500.0, 0.0)
    upside_base = df["kv_p90"] - df["kv"] + luck_rebound
    df["c_upside"] = _pct(df.assign(_u=upside_base), "_u")
    df["c_durab"] = 0.5 * _pct(df, "avail") + 0.5 * _pct(df, "useful")
    df["key"] = df["name"].map(norm)
    df["actif"] = df["gp_cur"] > 0
    df["P"] = list(P)
    df["S"] = list(S)
    df["SD"] = list(SD)
    return {
        "players": df, "ps": ps, "groups": ginfo, "repl": repl, "cur": cur,
        "curves": curves, "rets": rets, "params": par,
        "backtest": calib["backtest"], "grid": calib["grid"], "signals": calib["signals"],
        "importances": calib.get("importances", {}),
        "n_folds": calib["n_folds"], "calibrated": par.calibrated,
        "pre_nhl": data.get("pre_nhl", {}),
    }


def apply_weights(df: pd.DataFrame, w: dict[str, float], n_teams: int, risk_aversion: float = 0.0) -> pd.DataFrame:
    """
    Score (0-100) = synthèse pondérée et normalisée des 5 piliers stratégiques :
      - w['keeper'] (défaut 45%) : Valeur keeper future actualisée (rythme, trajectoire et âge)
      - w['current'] (défaut 20%) : Valeur de la saison courante (VORP au-dessus du remplacement)
      - w['opp'] (défaut 15%)     : Opportunité / déploiement (TOI, PP, OZ Start%, tirs/m, club)
      - w['durab'] (défaut 10%)   : Durabilité et disponibilité (historique de blessures, années utiles)
      - w['upside'] (défaut 10%)  : Plafond offensif (P90) + potentiel de rebond statistique (PDO bas)
    """
    df = df.copy()
    wk = float(w.get("keeper", 45))
    wc = float(w.get("current", 20))
    wo = float(w.get("opp", 15))
    wd = float(w.get("durab", 10))
    wu = float(w.get("upside", 10))
    tot = (wk + wc + wo + wd + wu) or 1.0

    kv_adj = (df["kv"] - risk_aversion * df["kv_sd"]).clip(lower=0)
    kv_ann = kv_adj / df["disc_sum"]
    df["ev_annuel"] = kv_ann

    # Normalisation 0-100 de chaque pilier
    c_k = 100.0 * kv_ann / max(float(kv_ann.max()), 1e-9)
    c_c = 100.0 * df["vorp0"].clip(lower=0) / max(float(df["vorp0"].max()), 1e-9)
    c_o = df["c_opp"].fillna(50.0) if "c_opp" in df.columns else pd.Series(50.0, index=df.index)
    c_d = df["c_durab"].fillna(50.0) if "c_durab" in df.columns else pd.Series(50.0, index=df.index)
    c_u = df["c_upside"].fillna(50.0) if "c_upside" in df.columns else pd.Series(50.0, index=df.index)

    df["c_keeper"] = c_k
    df["c_current"] = c_c

    score_raw = (wk * c_k + wc * c_c + wo * c_o + wd * c_d + wu * c_u) / tot
    df["Score"] = 100.0 * score_raw / max(float(score_raw.max()), 1e-9)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rang"] = np.arange(1, len(df) + 1)
    n = n_teams
    bins = [0, 2 * n, 6 * n, 12 * n, 24 * n, 10 ** 9]
    labels = ["S – Franchise", "A – Noyau", "B – Protégeable", "C – Repêchable", "D – Ballottage"]
    df["Tier"] = pd.cut(df["Rang"], bins=bins, labels=labels)
    from .utils import get_edge_signal
    df["Verdict Edge"] = df.apply(get_edge_signal, axis=1)
    return df

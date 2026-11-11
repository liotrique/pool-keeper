# -*- coding: utf-8 -*-
"""
Utilitaires et fonctions de transformation.
==========================================
Normalisation de chaînes, calculs de saisons, recherche de noms et vues tabulaires.
"""
from __future__ import annotations

import unicodedata
import numpy as np
import pandas as pd


def norm(s: str | object) -> str:
    """Normalise une chaîne de caractères (retire les accents, minuscules, sans espaces superflus)."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def season_shift(season: int, n: int) -> int:
    """Décale une saison LNH (ex: 20232024 avec n=1 -> 20242025)."""
    y = int(season) // 10000 + n
    return y * 10000 + y + 1


def season_label(s: int) -> str:
    """Formate une saison en libellé lisible (ex: 20232024 -> '2023-24')."""
    return f"{int(s) // 10000}-{str(int(s) % 10000)[2:]}"


def parse_names(text: str) -> list[str]:
    """Extrait et normalise la liste de noms d'un bloc de texte ligne par ligne."""
    return [norm(x) for x in text.splitlines() if x.strip()]


def match_names(df: pd.DataFrame, names: list[str]) -> tuple[list, list[str]]:
    """Associe des noms de joueurs normalisés aux lignes du DataFrame."""
    found, missing = [], []
    for n in names:
        ex = df[df["key"] == n]
        if ex.empty:
            ex = df[df["key"].str.contains(n, regex=False)]
        if ex.empty:
            missing.append(n)
        else:
            found.append(ex.sort_values("Rang").index[0])
    return found, missing


def to_view(df: pd.DataFrame) -> pd.DataFrame:
    """Formate le DataFrame des joueurs pour l'affichage utilisateur en mettant la valeur clé en évidence."""
    res = {
        "Rang": df["Rang"],
        "Joueur": df["name"],
        "Score": df["Score"].round(1),
        "Tier": df["Tier"].astype(str),
        "Pos": df["pos"],
        "Rang pos": df["rank_pos"],
        "Éq.": df["team"],
        "Off. Éq.": df["tm_gpg"].round(2) if "tm_gpg" in df else np.nan,
        "Âge": df["age_cur"].round(1),
        "Valeur keeper": df["kv"].round(1),
        "Rythme 82m": df["pace82"].round(1) if "pace82" in df else (df["rate0"] * 82).round(1),
        "Proj. saison": df["pts0"].round(1),
        "Peak": df["peak_pts"].round(1) if "peak_pts" in df else df["pts0"].round(1),
        "Âge peak": df["peak_age"].round(0) if "peak_age" in df else df["age_cur"].round(0),
        "Proj. restant": df["ros"].round(1),
        "VORP": df["vorp0"].round(1),
        "Pts pool": df["pool_cur"].round(1),
        "GP": df["gp_cur"].astype(int),
        "Années utiles": df["useful"].round(1),
        "Dispo %": (df["avail"] * 100).round(0),
        "TOI": df["toi_w"].round(1),
        "TOI PP": df["pp_w"].round(1),
        "Pts/60": df["p60"].round(2),
        "Opport.": df["c_opp"].round(0),
        "Durab.": df["c_durab"].round(0),
        "Upside": df["c_upside"].round(0),
    }
    if "corsi_pct" in df:
        res["Corsi CF%"] = (df["corsi_pct"] * 100).round(1)
    if "pdo" in df:
        res["PDO"] = df["pdo"].round(3)
    if "oz_start_pct" in df:
        res["OZ Start%"] = (df["oz_start_pct"] * 100).round(1)
    if "shots_per_gp" in df:
        res["Tirs/m"] = df["shots_per_gp"].round(1)
    if "sh_pct_5v5" in df:
        res["Sh% 5c5"] = (df["sh_pct_5v5"] * 100).round(1)
    if "on_ice_sh_pct" in df:
        res["On-Ice Sh%"] = (df["on_ice_sh_pct"] * 100).round(1)
    if "pre_nhl" in df:
        res["Pré-NHL"] = df["pre_nhl"].fillna("")
    return pd.DataFrame(res, index=df.index)


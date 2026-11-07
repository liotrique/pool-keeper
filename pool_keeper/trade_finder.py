# -*- coding: utf-8 -*-
"""
Trade Finder : choisis un joueur cible, l'outil propose des combinaisons de
joueurs de TON équipe à offrir pour un échange raisonnable (selon le Score)
qui te laisse quand même un gain de valeur.

À placer à côté de trade_simulator.py (réutilise compare_trade).
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import pandas as pd

try:
    from pool_keeper.trade import compare_trade
except ImportError:
    from trade import compare_trade

NAME_CANDIDATES = ("Joueur", "Nom", "Name", "Player", "player", "name")


def _detect_name_col(df: pd.DataFrame, name_col: str | None) -> str | None:
    if name_col and name_col in df.columns:
        return name_col
    for c in NAME_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _names(df: pd.DataFrame, name_col: str | None) -> list[str]:
    if name_col:
        return df[name_col].astype(str).tolist()
    return [str(i) for i in df.index]


def suggest_offers(
    target: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    name_col: str | None = None,
    max_give: int = 3,
    min_ratio: float = 0.92,
    max_ratio: float = 1.15,
    protected: Iterable[str] = (),
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Propose des offres pour obtenir `target` (1 joueur, ou plus) avec des joueurs de `roster`.

    Paramètres
    ----------
    target     : DataFrame (1+ lignes) du/des joueur(s) visé(s) (colonnes de summarize_roster).
    roster     : DataFrame de TON équipe (même colonnes). La cible ne doit pas y figurer.
    min_ratio  : Score offert / Score cible minimum. 0.92 = tu peux payer 8 % de moins
                 en valeur Score que ce que tu reçois (ton gain), sans trop insulter l'autre GM.
                 Monte vers 1.0 pour des offres plus "acceptables" pour l'autre équipe.
    max_ratio  : plafond (au-delà tu sur-paies : pas intéressant pour toi).
    protected  : noms (ou index) de joueurs intouchables.
    max_give   : nombre max de joueurs offerts (1, 2 ou 3).

    Retourne un DataFrame trié du plus avantageux pour toi (delta Score) au moins avantageux,
    parmi les offres qui restent dans la fenêtre « raisonnable ».
    """
    if target.empty or roster.empty:
        return pd.DataFrame()

    name_col = _detect_name_col(roster, name_col)
    target_score = float(target["Score"].sum())
    if target_score <= 0:
        return pd.DataFrame()

    protected_set = {str(p) for p in protected}
    r_names = _names(roster, name_col)
    target_names = set(_names(target, name_col))
    target_indices = set(target.index)

    pool_idx = [
        i for i, (idx, n) in enumerate(zip(roster.index, r_names))
        if n not in protected_set
        and str(idx) not in protected_set
        and idx not in protected_set
        and n not in target_names
        and idx not in target_indices
    ]
    scores = roster["Score"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for k in range(1, max_give + 1):
        for combo in combinations(pool_idx, k):
            give_score = float(scores[list(combo)].sum())
            ratio = give_score / target_score
            if not (min_ratio <= ratio <= max_ratio):
                continue
            # Évite les offres où un joueur est "du remplissage" quasi sans valeur
            if k > 1 and min(scores[list(combo)]) < 0.05 * give_score:
                continue
            rows.append({"combo": combo, "ratio": ratio, "give_score": give_score})

    if not rows:
        return pd.DataFrame()

    # Pré-tri par gain Score (côté toi), puis analyse détaillée sur les meilleurs seulement
    rows.sort(key=lambda r: r["ratio"])          # ratio bas = plus de gain pour toi
    rows = rows[: max(top_n * 5, 50)]

    out: list[dict[str, Any]] = []
    for r in rows:
        give_df = roster.iloc[list(r["combo"])]
        cmp = compare_trade(give_df, target)
        out.append({
            "Tu offres": " + ".join(_names(give_df, name_col)),
            "Nb": len(give_df),
            "Score offert": round(r["give_score"], 1),
            "Score cible": round(target_score, 1),
            "Ratio": round(r["ratio"], 2),
            "Δ Score": round(cmp["delta_score"], 1),
            "Δ Keeper": round(cmp["delta_kv"], 1),
            "Δ ROS": round(cmp["delta_ros"], 1),
            "Δ VORP": round(cmp["delta_vorp"], 1),
            "Δ Âge": round(cmp["delta_age"], 1),
            "Profil": cmp["profile_type"],
            "Notes": " | ".join(cmp["notes"]),
            "_give_ids": give_df.index.tolist(),
        })

    res = pd.DataFrame(out)
    # Tri : plus gros gain de Score, puis gain keeper, puis gain ROS
    res = res.sort_values(["Δ Score", "Δ Keeper", "Δ ROS"], ascending=False)
    return res.head(top_n).reset_index(drop=True)


def find_target(pool: pd.DataFrame, query: str, name_col: str | None = None) -> pd.DataFrame:
    """Retrouve un joueur par nom (recherche partielle, insensible à la casse)."""
    name_col = _detect_name_col(pool, name_col)
    if not name_col:
        raise ValueError("Aucune colonne de nom détectée : passe name_col=...")
    mask = pool[name_col].astype(str).str.contains(query, case=False, na=False, regex=False)
    return pool[mask]


# ----------------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------------
# cible   = find_target(df_autres_equipes, "McDavid")          # 1 ligne attendue
# offres  = suggest_offers(cible, df_mon_equipe,
#                          max_give=2, min_ratio=0.92,
#                          protected=["Mon joueur intouchable"])
# print(offres[["Tu offres", "Ratio", "Δ Score", "Δ Keeper", "Δ ROS", "Notes"]])
#
# Streamlit : st.dataframe(offres)

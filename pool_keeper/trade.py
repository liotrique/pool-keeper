# -*- coding: utf-8 -*-
"""
Module de simulation et d'évaluation d'échanges (Trade Simulator).
=================================================================
Calcul des différentiels de valeur keeper, points restants, VORP,
âge moyen, et diagnostics contextuels (Win-Now vs Rebuild).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_roster(df: pd.DataFrame) -> dict[str, Any]:
    """Calcule les statistiques agrégées d'un ensemble de joueurs dans une transaction."""
    if df.empty:
        return {
            "count": 0,
            "score_sum": 0.0,
            "score_mean": 0.0,
            "kv_sum": 0.0,
            "kv_mean": 0.0,
            "vorp_sum": 0.0,
            "vorp_mean": 0.0,
            "ros_sum": 0.0,
            "pts0_sum": 0.0,
            "age_mean": 0.0,
            "useful_sum": 0.0,
            "n_f": 0,
            "n_d": 0,
            "n_g": 0,
        }

    return {
        "count": len(df),
        "score_sum": float(df["Score"].sum()),
        "score_mean": float(df["Score"].mean()),
        "kv_sum": float(df["kv"].sum()),
        "kv_mean": float(df["kv"].mean()),
        "vorp_sum": float(df["vorp0"].sum()),
        "vorp_mean": float(df["vorp0"].mean()),
        "ros_sum": float(df["ros"].sum()),
        "pts0_sum": float(df["pts0"].sum()),
        "age_mean": float(df["age_cur"].mean()),
        "useful_sum": float(df["useful"].sum()),
        "n_f": int((df["grp"] == "F").sum()),
        "n_d": int((df["grp"] == "D").sum()),
        "n_g": int((df["grp"] == "G").sum()),
    }


def compare_trade(side_a: pd.DataFrame, side_b: pd.DataFrame) -> dict[str, Any]:
    """
    Compare deux côtés d'un échange du point de vue de l'Équipe A
    (qui donne side_a et reçoit side_b).
    """
    stats_a = summarize_roster(side_a)
    stats_b = summarize_roster(side_b)

    delta_score = stats_b["score_sum"] - stats_a["score_sum"]
    delta_kv = stats_b["kv_sum"] - stats_a["kv_sum"]
    delta_vorp = stats_b["vorp_sum"] - stats_a["vorp_sum"]
    delta_ros = stats_b["ros_sum"] - stats_a["ros_sum"]
    delta_pts0 = stats_b["pts0_sum"] - stats_a["pts0_sum"]
    delta_age = stats_b["age_mean"] - stats_a["age_mean"] if stats_a["count"] and stats_b["count"] else 0.0
    delta_useful = stats_b["useful_sum"] - stats_a["useful_sum"]

    # Diagnostic textuel intelligent
    verdict_short: str
    if delta_ros > 8.0:
        verdict_short = f"🟢 **Gain immédiat net** : +{delta_ros:.1f} pts de pool projetés d'ici la fin de saison."
    elif delta_ros < -8.0:
        verdict_short = f"🔴 **Perte à court terme** : {delta_ros:.1f} pts de pool projetés pour le reste de la saison."
    else:
        verdict_short = "🟡 **Impact immédiat équilibré** : écart marginal sur le reste de la saison courante."

    verdict_long: str
    if delta_kv > 15.0:
        verdict_long = f"🟢 **Victoire nette sur la valeur keeper** (+{delta_kv:.1f} pts VORP actualisés)."
    elif delta_kv < -15.0:
        verdict_long = f"🔴 **Déficit d'actifs futurs** ({delta_kv:.1f} pts VORP actualisés sur l'horizon keeper)."
    else:
        verdict_long = "🟡 **Valeur keeper équilibrée** sur les prochaines saisons."

    # Profil de la transaction
    profile_type: str
    if delta_ros > 6.0 and delta_kv < -8.0:
        profile_type = "🏆 **Transaction de type 'Win-Now'** : Vous sacrifiez de la pérennité future pour maximiser vos chances de titre immédiatement."
    elif delta_ros < -6.0 and delta_kv > 8.0:
        profile_type = "🌱 **Transaction de type 'Reconstruction'** : Vous cédez de la production présente pour rajeunir et bonifier durablement votre noyau de protégés."
    elif delta_kv >= 0 and delta_ros >= 0:
        profile_type = "✨ **Gain multi-facettes** : Transaction favorable à la fois pour le court terme et le futur !"
    elif delta_kv < 0 and delta_ros < 0:
        profile_type = "⚠️ **Attention** : L'échange accuse un déficit à la fois immédiat et futur selon les projections actuelles."
    else:
        profile_type = "⚖️ **Transaction équilibrée** : Échange poste pour poste ou échange tactique."

    notes: list[str] = []
    # Roster spots
    if stats_a["count"] > stats_b["count"]:
        diff = stats_a["count"] - stats_b["count"]
        notes.append(
            f"📦 **Alignement (2 pour 1)** : Vous libérez {diff} place(s) d'alignement. "
            f"N'oubliez pas d'inclure la valeur d'un joueur libre sur le ballottage dans votre réflexion."
        )
    elif stats_b["count"] > stats_a["count"]:
        diff = stats_b["count"] - stats_a["count"]
        notes.append(
            f"📦 **Alignement (1 pour 2)** : Vous devrez libérer {diff} joueur(s) de votre club vers le ballottage."
        )

    # Position alerts
    if stats_a["n_g"] > stats_b["n_g"]:
        notes.append("🥅 **Gardiens** : Vous donnez plus de gardiens que vous n'en recevez. Vérifiez votre respect du quota de partants (2 G).")
    elif stats_b["n_g"] > stats_a["n_g"]:
        notes.append("🥅 **Gardiens** : Vous ajoutez un gardien. Rappel : la limite de protégés autorise un maximum de 3 gardiens.")

    if stats_b["age_mean"] < stats_a["age_mean"] - 1.5 and stats_a["count"] > 0 and stats_b["count"] > 0:
        notes.append(f"👶 **Rajeunissement** : L'âge moyen des joueurs acquis est plus jeune de {abs(delta_age):.1f} ans.")
    elif stats_b["age_mean"] > stats_a["age_mean"] + 1.5 and stats_a["count"] > 0 and stats_b["count"] > 0:
        notes.append(f"⏳ **Vieillissement** : L'âge moyen des joueurs acquis est supérieur de {delta_age:.1f} ans.")

    return {
        "side_a": stats_a,
        "side_b": stats_b,
        "delta_score": delta_score,
        "delta_kv": delta_kv,
        "delta_vorp": delta_vorp,
        "delta_ros": delta_ros,
        "delta_pts0": delta_pts0,
        "delta_age": delta_age,
        "delta_useful": delta_useful,
        "verdict_short": verdict_short,
        "verdict_long": verdict_long,
        "profile_type": profile_type,
        "notes": notes,
    }


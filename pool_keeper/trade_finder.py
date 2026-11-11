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


def scan_rival_targets(
    candidates_pool: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    name_col: str | None = None,
    max_give: int = 2,
    min_ratio: float = 0.88,
    max_ratio: float = 1.12,
    protected: Iterable[str] = (),
    target_category: str = "all",
    pos_filter: str = "ALL",
    min_target_score: float = 35.0,
    sort_by: str = "attractiveness",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Scanne l'ensemble des joueurs d'autres équipes (ou du pool général) et identifie
    automatiquement les meilleures cibles à acquérir, en calculant pour chacune la combinaison
    optimale de joueurs de votre équipe à offrir pour une transaction équilibrée et rentable.

    Paramètres
    ----------
    candidates_pool  : DataFrame des joueurs des autres équipes / du pool.
    roster           : DataFrame de votre propre alignement.
    max_give         : Nombre max de joueurs à offrir (1 ou 2).
    min_ratio        : Ratio Score offert / Score cible minimum (ex: 0.88).
    max_ratio        : Ratio Score offert / Score cible maximum (ex: 1.12).
    protected        : Joueurs intouchables de votre club à exclure.
    target_category  : Filtre stratégique ('all', 'buy_low', 'keeper', 'win_now', 'defense', 'goalie', 'sell_high_give').
    pos_filter       : Filtre de position ('ALL', 'F', 'D', 'G').
    min_target_score : Seuil de score minimal pour la cible (ignore le ballottage marginal).
    sort_by          : Critère de tri ('attractiveness', 'delta_score', 'delta_kv', 'delta_ros', 'score_target').
    top_n            : Nombre maximal de cibles retournées.
    """
    if candidates_pool.empty or roster.empty:
        return pd.DataFrame()

    name_col = _detect_name_col(roster, name_col) or _detect_name_col(candidates_pool, name_col)

    protected_set = {str(p) for p in protected}
    r_names = _names(roster, name_col)
    roster_indices = set(roster.index)

    # Pré-filtrage des joueurs disponibles de notre club
    avail_indices = [
        i for i, (idx, n) in enumerate(zip(roster.index, r_names))
        if n not in protected_set and str(idx) not in protected_set and idx not in protected_set
    ]
    if not avail_indices:
        return pd.DataFrame()

    scores_roster = roster["Score"].to_numpy(dtype=float)
    r_idx_list = list(roster.index)

    # Pré-génération de toutes les combinaisons possibles de notre club (1 ou 2 joueurs)
    combos: list[tuple[tuple[int, ...], float, list[Any]]] = []
    for k in range(1, max_give + 1):
        for c in combinations(avail_indices, k):
            give_s = float(scores_roster[list(c)].sum())
            if give_s <= 0:
                continue
            # Éviter les combos avec un joueur bouche-trou négligeable (< 10% du score offert)
            if k > 1 and min(scores_roster[list(c)]) < 0.10 * give_s:
                continue
            combos.append((c, give_s, [r_idx_list[i] for i in c]))

    if not combos:
        return pd.DataFrame()

    # Filtrage initial des candidats adverses (exclure nos propres joueurs)
    cands = candidates_pool[~candidates_pool.index.isin(roster_indices)]
    if name_col and name_col in cands.columns:
        cands = cands[~cands[name_col].astype(str).isin(set(r_names))]

    if min_target_score > 0 and "Score" in cands.columns:
        cands = cands[cands["Score"] >= min_target_score]

    if pos_filter != "ALL":
        if pos_filter == "F":
            cands = cands[cands["grp"] == "F"] if "grp" in cands.columns else cands[cands["pos"].isin(["C", "L", "R", "LW", "RW"])]
        elif pos_filter == "D":
            cands = cands[cands["grp"] == "D"] if "grp" in cands.columns else cands[cands["pos"].isin(["D", "LD", "RD"])]
        elif pos_filter == "G":
            cands = cands[cands["grp"] == "G"] if "grp" in cands.columns else cands[cands["pos"] == "G"]

    if cands.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []

    for target_idx, target_row in cands.iterrows():
        t_score = float(target_row.get("Score", 0.0))
        if t_score <= 0:
            continue

        target_name = str(target_row[name_col]) if name_col and name_col in target_row else str(target_idx)
        target_edge = str(target_row.get("Verdict Edge", "—"))
        target_pos = str(target_row.get("pos", target_row.get("grp", "—")))
        target_team = str(target_row.get("team", "—"))
        age = float(target_row.get("age_cur", 26.0))
        is_young = age <= 24.5
        is_buy_low = any(b in target_edge for b in ["Buy-Low", "Malchanceux"])

        target_df_single = pd.DataFrame([target_row])

        # Trouver la meilleure offre pour cette cible
        best_candidate: dict[str, Any] | None = None
        best_attr = -999.0
        valid_offers_count = 0

        for c, give_s, give_ids in combos:
            ratio = give_s / t_score
            if not (min_ratio <= ratio <= max_ratio):
                continue

            valid_offers_count += 1
            give_df = roster.loc[give_ids]
            cmp = compare_trade(give_df, target_df_single)

            give_edges = [str(e) for e in give_df["Verdict Edge"]] if "Verdict Edge" in give_df.columns else []
            is_sell_high_give = any("Sell-High" in e or "Chanceux" in e for e in give_edges)

            # Score composite d'attractivité de la cible
            attr = (
                cmp["delta_score"] * 3.0
                + cmp["delta_kv"] * 0.4
                + cmp["delta_ros"] * 0.5
                + (18.0 if is_buy_low else 0.0)
                + (12.0 if is_sell_high_give else 0.0)
                + (8.0 if (is_young and cmp["delta_kv"] > 0) else 0.0)
            )

            if attr > best_attr:
                best_attr = attr
                best_candidate = {
                    "give_df": give_df,
                    "give_ids": give_ids,
                    "give_score": give_s,
                    "ratio": ratio,
                    "cmp": cmp,
                    "is_sell_high_give": is_sell_high_give,
                }

        if not best_candidate:
            continue

        give_df = best_candidate["give_df"]
        cmp = best_candidate["cmp"]
        is_sell_high_give = best_candidate["is_sell_high_give"]

        # Filtre stratégique post-évaluation
        if target_category == "buy_low" and not is_buy_low:
            continue
        if target_category == "keeper" and not (cmp["delta_kv"] >= 10.0 or (is_young and cmp["delta_kv"] > 0)):
            continue
        if target_category == "win_now" and not (cmp["delta_ros"] >= 4.0):
            continue
        if target_category == "defense" and not (target_pos in ("D", "LD", "RD") or target_row.get("grp") == "D"):
            continue
        if target_category == "goalie" and not (target_pos == "G" or target_row.get("grp") == "G"):
            continue
        if target_category == "sell_high_give" and not is_sell_high_give:
            continue

        # Qualification du profil & argumentaire stratégique
        give_names_str = " + ".join(_names(give_df, name_col))
        if is_buy_low and is_sell_high_give:
            profile_tag = "💎 Arbitrage Parfait"
            pitch = f"Coup double : Vendez au sommet {give_names_str} (Sell-High) pour acheter {target_name} (Buy-Low) à rabais !"
        elif is_buy_low:
            profile_tag = "🟢 Cible Buy-Low"
            pitch = f"{target_name} traverse un creux statistique passager ({target_edge}). Moment idéal pour l'arracher à son DG."
        elif is_sell_high_give:
            profile_tag = "🔴 Vente au Sommet"
            pitch = f"Monnayez {give_names_str} pendant que sa valeur est gonflée par la chance pour obtenir un joueur supérieur."
        elif cmp["delta_ros"] >= 5.0 and cmp["delta_kv"] >= 5.0:
            profile_tag = "✨ Gain Multi-Facettes"
            pitch = f"Bonification globale : +{cmp['delta_ros']:.1f} pts restants cette saison et +{cmp['delta_kv']:.0f} pts Keeper !"
        elif cmp["delta_ros"] >= 4.5:
            profile_tag = "🏆 Renfort Win-Now"
            pitch = f"Boost immédiat (+{cmp['delta_ros']:.1f} pts ROS) idéal pour la course au championnat ou aux séries."
        elif cmp["delta_kv"] >= 18.0 or (is_young and cmp["delta_kv"] >= 8.0):
            profile_tag = "🌱 Pépite & Avenir"
            pitch = f"Noyau futur consolidé : +{cmp['delta_kv']:.0f} pts VORP Keeper avec un jeune talent prometteur ({age:.1f} ans)."
        elif target_pos in ("D", "LD", "RD") or target_row.get("grp") == "D":
            profile_tag = "🛡️ Pilier Défensif"
            pitch = f"Renfort majeur à la ligne bleue (+{cmp['delta_score']:+.1f} Score)."
        elif target_pos == "G" or target_row.get("grp") == "G":
            profile_tag = "🥅 Gardien Partant"
            pitch = f"Stabilité devant le filet (+{cmp['delta_score']:+.1f} Score)."
        else:
            profile_tag = "⚖️ Gain de Valeur"
            pitch = f"Gain net de +{cmp['delta_score']:+.1f} Score sans déséquilibrer votre masse d'actifs."

        results.append({
            "Cible": target_name,
            "Éq.": target_team,
            "Pos": target_pos,
            "Âge": round(age, 1),
            "Verdict Edge": target_edge,
            "Score Cible": round(t_score, 1),
            "Tu offres": give_names_str,
            "Nb": len(give_df),
            "Score offert": round(best_candidate["give_score"], 1),
            "Ratio": round(best_candidate["ratio"], 2),
            "Δ Score": round(cmp["delta_score"], 1),
            "Δ Keeper": round(cmp["delta_kv"], 1),
            "Δ ROS": round(cmp["delta_ros"], 1),
            "Δ VORP": round(cmp["delta_vorp"], 1),
            "Δ Âge": round(cmp["delta_age"], 1),
            "Attractivité": round(best_attr, 1),
            "Profil": profile_tag,
            "Pourquoi cibler": pitch,
            "Offres dispo": valid_offers_count,
            "_target_id": target_idx,
            "_give_ids": best_candidate["give_ids"],
        })

    if not results:
        return pd.DataFrame()

    out_df = pd.DataFrame(results)

    # Tri selon le critère choisi
    if sort_by == "attractiveness":
        out_df = out_df.sort_values(["Attractivité", "Δ Score", "Δ Keeper"], ascending=False)
    elif sort_by == "delta_score":
        out_df = out_df.sort_values(["Δ Score", "Δ Keeper", "Δ ROS"], ascending=False)
    elif sort_by == "delta_kv":
        out_df = out_df.sort_values(["Δ Keeper", "Δ Score", "Δ ROS"], ascending=False)
    elif sort_by == "delta_ros":
        out_df = out_df.sort_values(["Δ ROS", "Δ Score", "Δ Keeper"], ascending=False)
    elif sort_by == "score_target":
        out_df = out_df.sort_values(["Score Cible", "Attractivité"], ascending=False)
    else:
        out_df = out_df.sort_values(["Attractivité", "Δ Score"], ascending=False)

    return out_df.head(top_n).reset_index(drop=True)


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

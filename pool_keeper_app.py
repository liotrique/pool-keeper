# -*- coding: utf-8 -*-
"""
Pool GOOD YEAR Keeper – Moteur de scores LNH (Streamlit)
=========================================================
Lancer :
    pip install streamlit pandas numpy requests
    streamlit run pool_keeper_app.py

Données : API officielle de la LNH
    - https://api.nhle.com/stats/rest/en/...   (stats saison par saison, TOI, bios/date de naissance)
    - https://api-web.nhle.com/v1/...          (saison courante, game logs des gardiens)

Règlements intégrés (Constitution v2.1) :
    Attaquants : But 1 / Passe 1      Défenseurs : But 2 / Passe 1
    Gardiens   : Victoire 1 / Arrêt 0.1 / But accordé -1, plancher 0 par match
    Alignement : 12 A + 6 D + 2 G     Protégés : 12 joueurs/équipe     Banc illimité
"""
from __future__ import annotations

import streamlit as st

from pool_keeper.data import load_data
from pool_keeper.model import apply_weights, build_model
from pool_keeper.ui import (
    detect_mobile_mode,
    inject_responsive_css,
    render_sidebar,
    render_tab_club_waivers,
    render_tab_methodology,
    render_tab_player,
    render_tab_ranking,
    render_tab_scarcity,
    render_tab_trade_simulator,
)
from pool_keeper.utils import season_label


def main() -> None:
    st.set_page_config(page_title="Pool GOOD YEAR – Scores LNH", page_icon="🏒", layout="wide")
    inject_responsive_css()

    if "token" not in st.session_state:
        st.session_state["token"] = 0

    # ── Détection dynamique de l'écran mobile ────────────────────────────────
    is_mobile = detect_mobile_mode()

    st.title("🏒 Pool GOOD YEAR Keeper")

    # ── Barre latérale ────────────────────────────────────────────────────────
    n_seasons, use_logs, n_teams, bench, horizon, discount, decay, w, model_method = render_sidebar()

    # ── Chargement des données ────────────────────────────────────────────────
    try:
        data = load_data(n_seasons, use_logs, st.session_state["token"])
    except Exception as e:
        st.error(f"Impossible de charger les données : {e}")
        st.stop()

    # ── Modélisation et pondération ───────────────────────────────────────────
    model = build_model(data, n_teams, bench, horizon, discount, decay, method=model_method)
    players = apply_weights(model["players"], w, n_teams)
    cur = model["cur"]

    st.caption(
        f"Saison courante : **{season_label(cur)}** · {n_seasons} saisons chargées · "
        f"mise à jour : {data['fetched_at']} · {len(players)} joueurs scorés"
    )
    for msg in data["warnings"]:
        st.warning(msg)
    if use_logs and data["goalie_exact_share"] < 0.5:
        st.info("Les game logs des gardiens n'ont pas pu être tous récupérés : approximation du plancher utilisée pour certains.")

    is_mobile = st.session_state.get("is_mobile", False)

    # ── Onglets principaux ────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏆 Classement",
        "👤 Fiche joueur",
        "📊 Rareté par position",
        "🧢 Mon club & ballottage",
        "🔄 Simulateur d'échange",
        "📚 Méthodologie",
    ])

    with tabs[0]:
        render_tab_ranking(players, cur, is_mobile=is_mobile)

    with tabs[1]:
        render_tab_player(players, model, cur, is_mobile=is_mobile)

    with tabs[2]:
        render_tab_scarcity(players, model, is_mobile=is_mobile)

    with tabs[3]:
        render_tab_club_waivers(players, is_mobile=is_mobile)

    with tabs[4]:
        render_tab_trade_simulator(players, is_mobile=is_mobile)

    with tabs[5]:
        render_tab_methodology(model)


if __name__ == "__main__":
    main()

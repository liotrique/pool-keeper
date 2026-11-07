# -*- coding: utf-8 -*-
"""
Composants d'interface Streamlit du Pool Keeper.
================================================
Onglets d'affichage, graphiques, filtres, gestionnaire de club
et simulateur d'échanges multi-critères.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pool_keeper.config import (
    BENCH_DEFAULT,
    COLCFG,
    DEFAULT_N_SEASONS,
    DEFAULT_N_TEAMS,
    KEEPERS,
    MAX_YEARS,
    METHODO,
    SLOTS,
)
from pool_keeper.data import cache_purge_nonfinal
from pool_keeper.trade import compare_trade
from pool_keeper.trade_finder import find_target, suggest_offers
from pool_keeper.utils import match_names, norm, parse_names, season_label, season_shift, to_view


def inject_responsive_css() -> None:
    """Injecte des règles CSS responsives pour téléphones mobiles et tablettes."""
    st.markdown(
        """
        <style>
        /* ── Optimisations tactiles et mobiles globales ── */
        @media (max-width: 768px) {
            /* Marges et rembourrage de page réduits sur smartphone */
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                max-width: 100% !important;
            }

            /* Typographie adaptée pour petit écran */
            h1 {
                font-size: 1.45rem !important;
                line-height: 1.25 !important;
                margin-bottom: 0.3rem !important;
            }
            h2 {
                font-size: 1.2rem !important;
                margin-top: 0.8rem !important;
                margin-bottom: 0.4rem !important;
            }
            h3 {
                font-size: 1.05rem !important;
                margin-top: 0.6rem !important;
            }

            /* Onglets défilables horizontalement au toucher */
            div[data-baseweb="tab-list"] {
                overflow-x: auto !important;
                overflow-y: hidden !important;
                flex-wrap: nowrap !important;
                -webkit-overflow-scrolling: touch !important;
                padding-bottom: 6px !important;
                gap: 4px !important;
                border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
            }
            button[data-baseweb="tab"] {
                font-size: 0.82rem !important;
                padding: 6px 10px !important;
                white-space: nowrap !important;
                flex-shrink: 0 !important;
            }

            /* Cartes de métriques proportionnées */
            div[data-testid="stMetric"] {
                background: rgba(128, 128, 128, 0.08) !important;
                border-radius: 8px !important;
                padding: 8px 10px !important;
                margin-bottom: 6px !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
                font-weight: 700 !important;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.78rem !important;
            }

            /* Boutons agrandis pour le pouce */
            div[data-testid="stButton"] > button {
                min-height: 42px !important;
                font-size: 0.92rem !important;
                border-radius: 8px !important;
            }

            /* Tableaux défilables avec inertie tactile */
            div[data-testid="stDataFrame"] {
                border-radius: 8px !important;
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch !important;
            }
        }

        /* Style des fiches joueurs en mode carte */
        .player-mobile-card {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }
        .player-mobile-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .player-mobile-name {
            font-size: 1.05rem;
            font-weight: 700;
        }
        .player-mobile-badges {
            display: flex;
            gap: 6px;
        }
        .player-badge {
            font-size: 0.75rem;
            padding: 2px 7px;
            border-radius: 6px;
            font-weight: 600;
            background: rgba(128, 128, 128, 0.2);
        }
        .player-badge-score {
            background: #1e88e5;
            color: white;
        }
        .player-mobile-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            text-align: center;
            font-size: 0.8rem;
        }
        .player-stat-box {
            background: rgba(128, 128, 128, 0.06);
            border-radius: 6px;
            padding: 5px 3px;
        }
        .player-stat-val {
            font-weight: 700;
            font-size: 0.95rem;
        }
        .player-stat-lbl {
            font-size: 0.7rem;
            opacity: 0.8;
        }

        /* Masquage discret de l'iframe de détection dynamique d'écran */
        iframe[title="st.iframe"] {
            position: fixed !important;
            top: -9999px !important;
            left: -9999px !important;
            width: 0 !important;
            height: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def detect_mobile_mode() -> bool:
    """
    Détecte dynamiquement si l'utilisateur consulte l'application sur un écran mobile :
    1. Lit le paramètre d'URL 'mobile' s'il est déjà positionné (1 ou 0).
    2. Sinon, inspecte les en-têtes HTTP (User-Agent) pour un premier rendu immédiat.
    3. Injecte un script JavaScript invisible côté navigateur pour mesurer en continu
       la largeur réelle de l'écran (breakpoint <= 768px) et réagir aux redimensionnements
       ou rotations d'écran sans action manuelle requise.
    """
    # 1. Vérification dans l'URL
    url_mobile = st.query_params.get("mobile")
    if url_mobile in ("1", "true", "True"):
        is_mobile = True
    elif url_mobile in ("0", "false", "False"):
        is_mobile = False
    else:
        # 2. Détection initiale par User-Agent
        ua = ""
        try:
            if hasattr(st, "context") and hasattr(st.context, "headers"):
                ua = str(st.context.headers.get("user-agent", "") or "").lower()
        except Exception:
            ua = ""
        is_mobile = any(k in ua for k in ("mobi", "android", "iphone", "ipad", "ipod", "touch"))

    st.session_state["is_mobile"] = is_mobile

    # 3. Écouteur JS client de la taille réelle du viewport (<= 768px)
    py_mobile_str = "true" if is_mobile else "false"
    js_code = f"""
    <script>
    (function() {{
        let lastWidth = null;
        function checkViewport() {{
            try {{
                const targetWin = window.parent;
                const href = targetWin.location.href;
                const width = targetWin.innerWidth;
                if (lastWidth === width) return;
                lastWidth = width;

                const isMobileScreen = width <= 768;
                const currentPyMobile = {py_mobile_str};

                if (isMobileScreen !== currentPyMobile) {{
                    const url = new URL(href);
                    const targetVal = isMobileScreen ? "1" : "0";
                    if (url.searchParams.get("mobile") !== targetVal) {{
                        url.searchParams.set("mobile", targetVal);
                        targetWin.location.replace(url.href);
                    }}
                }}
            }} catch(e) {{
                // Sécurisé en contexte cross-origin : ne jamais recharger l'iframe !
            }}
        }}

        // Exécution au chargement
        checkViewport();

        // Réaction fluide au redimensionnement ou rotation d'appareil
        let timer = null;
        try {{
            window.parent.addEventListener("resize", function() {{
                clearTimeout(timer);
                timer = setTimeout(checkViewport, 300);
            }});
            window.parent.addEventListener("orientationchange", function() {{
                setTimeout(checkViewport, 200);
            }});
        }} catch(e) {{}}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

    return is_mobile


def render_sidebar() -> tuple[int, bool, int, dict[str, int], int, float, float, dict[str, int], str]:
    """Affiche la barre latérale et retourne les paramètres choisis par l'utilisateur."""
    sb = st.sidebar
    if sb.button("🔄 Rafraîchir les données LNH", type="primary", use_container_width=True):
        cache_purge_nonfinal()
        st.cache_data.clear()
        st.session_state["token"] += 1
        st.rerun()
    sb.caption(
        "Les saisons terminées sont gardées en cache disque ; "
        "le rafraîchissement ne retélécharge que la saison en cours."
    )

    sb.header("Données")
    n_seasons = sb.slider("Saisons d'historique", 3, 10, DEFAULT_N_SEASONS)
    use_logs = sb.checkbox(
        "Plancher exact des gardiens (game logs)",
        value=True,
        help="Plus précis, mais plus long au premier chargement (mis en cache ensuite).",
    )

    sb.header("Ligue")
    n_teams = sb.number_input(
        "Nombre d'équipes dans le pool",
        4,
        32,
        DEFAULT_N_TEAMS,
        step=1,
        help="Non précisé dans la Constitution : ajustez.",
    )
    with sb.expander("Banc par position (repêchage initial : 4 joueurs)"):
        bf = st.number_input("Banc A", 0, 6, BENCH_DEFAULT["F"])
        bd = st.number_input("Banc D", 0, 4, BENCH_DEFAULT["D"])
        bg = st.number_input("Banc G", 0, 3, BENCH_DEFAULT["G"])
    bench = {"F": int(bf), "D": int(bd), "G": int(bg)}

    sb.header("Modèle")
    method_options = {
        "boost": "🤖 IA Hybride (Gradient Boosting + Courbes)",
        "auto": "📊 Auto-calibré par Backtest",
        "toi_pp": "🎯 TOI + Avantage numérique (PP)",
        "toi": "📈 Temps de glace (TOI seul)",
    }
    model_method = sb.selectbox(
        "Moteur de prédiction du talent",
        list(method_options.keys()),
        index=0,
        format_func=lambda k: method_options[k],
        help="L'IA Hybride utilise le Gradient Boosting (LightGBM/HistGBDT) pour apprendre les non-linéarités complexes (rôle, PP, âge, cadence de buts) afin de projeter le rythme 82m, combiné aux courbes de vieillissement pour la trajectoire keeper.",
    )
    horizon = sb.slider("Horizon keeper (années)", 2, MAX_YEARS, 5)
    discount = sb.slider("Facteur d'actualisation / an", 0.60, 1.00, 0.85, 0.01)
    decay = sb.slider(
        "Poids des saisons passées (décroissance)",
        0.25,
        0.90,
        0.50,
        0.05,
        help="0.50 = chaque saison passée compte pour moitié moins. Les 1-3 dernières années dominent la projection individuelle du joueur tout en utilisant les 10 ans pour les courbes d'âge, l'opportunité et la survie.",
    )

    sb.header("Pondération du score")
    w = {
        "keeper": sb.slider("Valeur keeper", 0, 100, 45),
        "current": sb.slider("Valeur saison actuelle", 0, 100, 20),
        "opp": sb.slider("Opportunité", 0, 100, 15),
        "durab": sb.slider("Durabilité / longévité", 0, 100, 10),
        "upside": sb.slider("Upside", 0, 100, 10),
    }

    return n_seasons, use_logs, int(n_teams), bench, int(horizon), float(discount), float(decay), w, model_method


def render_tab_ranking(players: pd.DataFrame, cur: int, is_mobile: bool = False) -> None:
    """Affiche l'onglet Classement des joueurs avec filtres, tri explicite, sélection multiple, presets mobiles et export CSV."""
    if not is_mobile:
        st.info(
            "💡 **Ordre du classement de base :** Les joueurs sont classés par leur **Score (0-100)**, "
            "qui pondère la Valeur Keeper (45%), la Saison en cours (20%), l'Opportunité (15%), la Durabilité (10%) et l'Upside (10%). "
            "Survolez les en-têtes de colonnes **(?)** pour afficher l'explication détaillée de chaque statistique."
        )

    # ── Callbacks de gestion de la sélection (exécutés AVANT le rendu des widgets) ──
    def _clear_ranking_players() -> None:
        st.session_state["select_ranking_players"] = []

    def _apply_table_selection(indices: list[int]) -> None:
        st.session_state["select_ranking_players"] = list(indices)

    if "select_ranking_players" not in st.session_state:
        st.session_state["select_ranking_players"] = []

    all_indices = players.index.tolist()
    labels = {
        i: f"{players.loc[i, 'name']} ({players.loc[i, 'pos']}, {players.loc[i, 'team']}) – Rang #{int(players.loc[i, 'Rang'])} (Score: {players.loc[i, 'Score']:.1f})"
        for i in all_indices
    }

    sort_options = {
        "Score (Ordre de base par défaut)": ("Score", False),
        "Valeur keeper (Long terme)": ("kv", False),
        "Peak projeté (Sommet de carrière)": ("peak_pts", False),
        "Rythme 82m (Cadence sans blessure)": ("pace82", False),
        "Proj. restant (ROS - Court terme)": ("ros", False),
        "Proj. saison (Avec blessures prévues)": ("pts0", False),
        "Offensive d'équipe (Buts/m du club)": ("tm_gpg", False),
        "VORP (Rareté cette saison)": ("vorp0", False),
        "Pts pool réels accumulés": ("pool_cur", False),
        "Âge (Plus jeunes en premier)": ("age_cur", True),
    }

    a_min, a_max = float(players["age_cur"].min()), float(players["age_cur"].max())

    if is_mobile:
        # ── Disposition optimisée pour smartphone ─────────────────────────────
        player_sel = st.multiselect(
            "Sélectionner des joueurs :",
            options=all_indices,
            format_func=lambda i: labels.get(i, str(i)),
            placeholder="Rechercher des joueurs...",
            key="select_ranking_players",
        )
        search = st.text_input(
            "Recherche par nom(s) :",
            placeholder="ex: McDavid, Makar...",
            help="Recherche par nom complet ou partiel (séparer par des virgules pour plusieurs).",
        )
        sort_choice = st.selectbox("Trier par :", list(sort_options.keys()), index=0)

        with st.expander("🔍 Filtres supplémentaires (Position, Équipe, Tier, Âge...)", expanded=False):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                pos_sel = st.multiselect("Position", ["C", "L", "R", "D", "G"])
            with col_m2:
                tier_sel = st.multiselect("Tier", sorted(players["Tier"].astype(str).unique()))
            team_sel = st.multiselect("Équipe", sorted(players["team"].dropna().unique()))
            age_rng = st.slider(
                "Âge",
                float(int(a_min)),
                float(int(a_max) + 1),
                (float(int(a_min)), float(int(a_max) + 1)),
            )
            only_active = st.checkbox("Seulement ceux qui ont joué cette saison", value=False)
    else:
        # ── Disposition classique sur grand écran (Desktop) ──────────────────
        c1, c2, c3 = st.columns([2, 3, 2.5])
        pos_sel = c1.multiselect("Position", ["C", "L", "R", "D", "G"])
        team_sel = c2.multiselect("Équipe", sorted(players["team"].dropna().unique()))
        tier_sel = c3.multiselect("Tier", sorted(players["Tier"].astype(str).unique()))

        cp1, cp2 = st.columns([3.5, 2.5])
        player_sel = cp1.multiselect(
            "Sélectionner des joueurs (filtrage direct & comparaison multiple) :",
            options=all_indices,
            format_func=lambda i: labels.get(i, str(i)),
            placeholder="Tapez le nom d'un ou plusieurs joueurs...",
            key="select_ranking_players",
        )
        search = cp2.text_input(
            "Recherche libre (nom(s) séparés par des virgules) :",
            placeholder="ex: McDavid, Makar, Celebrini...",
            help="Recherche par nom complet ou partiel. Vous pouvez inscrire plusieurs joueurs séparés par des virgules ou des retours à la ligne.",
        )

        c_sort, c_age, c_act = st.columns([3, 2.5, 2.5])
        sort_choice = c_sort.selectbox("Trier par la valeur clé :", list(sort_options.keys()), index=0)
        age_rng = c_age.slider(
            "Âge",
            float(int(a_min)),
            float(int(a_max) + 1),
            (float(int(a_min)), float(int(a_max) + 1)),
        )
        only_active = c_act.checkbox("Seulement ceux qui ont joué cette saison", value=False)

    # ── Notification et bouton pour réinitialiser la sélection de joueurs ─────
    if player_sel:
        c_pinfo, c_pbtn = st.columns([4, 1])
        sel_names = [players.loc[i, "name"] for i in player_sel if i in players.index]
        c_pinfo.success(f"🎯 **{len(player_sel)} joueur(s) sélectionné(s) :** {', '.join(sel_names)}")
        c_pbtn.button(
            "✕ Tout afficher",
            key="btn_clear_player_sel",
            on_click=_clear_ranking_players,
            use_container_width=True,
        )

    # ── Application des filtres ───────────────────────────────────────────────
    f = players
    if player_sel:
        f = f[f.index.isin(player_sel)]
    if pos_sel:
        f = f[f["pos"].isin(pos_sel)]
    if team_sel:
        f = f[f["team"].isin(team_sel)]
    if tier_sel:
        f = f[f["Tier"].astype(str).isin(tier_sel)]
    if search:
        search_terms = [
            norm(term.strip())
            for term in (
                search.replace("\n", ",")
                .replace(";", ",")
                .replace("/", ",")
                .replace("|", ",")
                .split(",")
            )
            if term.strip()
        ]
        if search_terms:
            pattern = "|".join([re.escape(t) for t in search_terms])
            f = f[f["key"].str.contains(pattern, regex=True)]
    f = f[(f["age_cur"] >= age_rng[0]) & (f["age_cur"] <= age_rng[1])]
    if only_active:
        f = f[f["actif"]]

    if player_sel and f.empty:
        st.warning(
            "⚠️ Aucun des joueurs sélectionnés n'apparaît car d'autres filtres "
            "(Position, Équipe, Tier, Âge ou Actif) les excluent. "
            "Ajustez ou réinitialisez les autres filtres pour les afficher."
        )

    sort_col, sort_asc = sort_options[sort_choice]
    f = f.sort_values(sort_col, ascending=sort_asc)

    view = to_view(f)

    # ── Presets de colonnes & Choix du mode d'affichage ───────────────────────
    col_presets = {
        "📱 Essentiel (Mobile)": [
            "Rang", "Joueur", "Pos", "Éq.", "Score", "Tier", "Valeur keeper", "Proj. saison", "Peak"
        ],
        "⚡ Offensif (Points & Rythme)": [
            "Rang", "Joueur", "Pos", "Éq.", "Score", "Rythme 82m", "Proj. saison", "Proj. restant", "Pts pool", "GP"
        ],
        "🔮 Keeper (Long terme)": [
            "Rang", "Joueur", "Pos", "Éq.", "Score", "Âge", "Valeur keeper", "Peak", "Âge peak", "Années utiles"
        ],
        "💻 Complet (Toutes les 26 colonnes)": None,
    }

    col_p1, col_p2 = st.columns([2.5, 1.5] if not is_mobile else [2, 1])
    preset_choice = col_p1.selectbox(
        "Colonnes du tableau :",
        list(col_presets.keys()),
        index=0 if is_mobile else 3,
        key="ranking_col_preset",
    )
    view_format = col_p2.radio(
        "Format :",
        ["📋 Tableau", "🃏 Cartes"],
        horizontal=True,
        key="ranking_view_format",
    )

    selected_cols = col_presets[preset_choice]
    if selected_cols:
        view_cols_to_use = [c for c in selected_cols if c in view.columns]
        view_to_show = view[view_cols_to_use]
    else:
        view_to_show = view

    table_selected_indices = []

    if view_format == "🃏 Cartes":
        max_cards = 30
        top_f = f.head(max_cards)
        st.caption(f"Affichage de **{len(top_f)}** cartes joueur(s) sur **{len(f)}** trouvés :")
        for _, r in top_f.iterrows():
            pace_val = r.get("pace82", r["rate0"] * 82)
            card_html = f"""
            <div class="player-mobile-card">
              <div class="player-mobile-card-header">
                <div>
                  <span style="font-weight:800; opacity:0.6; margin-right:6px;">#{int(r['Rang'])}</span>
                  <span class="player-mobile-name">{r['name']}</span>
                </div>
                <div class="player-mobile-badges">
                  <span class="player-badge">{r['pos']} · {r['team']} · {r['age_cur']:.0f} ans</span>
                  <span class="player-badge player-badge-score">Score {r['Score']:.1f}</span>
                </div>
              </div>
              <div class="player-mobile-stats">
                <div class="player-stat-box"><div class="player-stat-val">{r['Score']:.1f}</div><div class="player-stat-lbl">Score (0-100)</div></div>
                <div class="player-stat-box"><div class="player-stat-val">{pace_val:.0f}</div><div class="player-stat-lbl">Rythme 82m</div></div>
                <div class="player-stat-box"><div class="player-stat-val">{r['pts0']:.0f}</div><div class="player-stat-lbl">Proj. Saison</div></div>
                <div class="player-stat-box"><div class="player-stat-val">{r['kv']:.0f}</div><div class="player-stat-lbl">Keeper</div></div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        if len(f) > max_cards:
            st.info(f"💡 {len(f) - max_cards} autres joueurs disponibles. Utilisez la recherche ci-dessus ou basculez sur l'affichage 'Tableau' pour voir l'ensemble.")
    else:
        # ── Affichage interactif avec sélection multi-lignes ──────────────────
        selection_event = st.dataframe(
            view_to_show,
            use_container_width=True,
            hide_index=True,
            column_config=COLCFG,
            height=520 if is_mobile else 640,
            on_select="rerun",
            selection_mode="multi-row",
            key="ranking_table_grid",
        )

        table_selected_rows: list[int] = []
        if hasattr(selection_event, "selection") and selection_event.selection:
            sel_obj = selection_event.selection
            if isinstance(sel_obj, dict):
                table_selected_rows = sel_obj.get("rows", [])
            else:
                table_selected_rows = getattr(sel_obj, "rows", [])

        table_selected_indices = [view.index[r] for r in table_selected_rows if r < len(view)]

        if table_selected_indices and not player_sel:
            c_ts1, c_ts2 = st.columns([3.5, 1.5] if not is_mobile else [2.5, 1.5])
            tbl_names = [players.loc[i, "name"] for i in table_selected_indices if i in players.index]
            c_ts1.info(f"📌 **{len(table_selected_indices)} joueur(s) coché(s) :** {', '.join(tbl_names)}")
            c_ts2.button(
                "🔍 Filtrer sur cette sélection",
                key="btn_apply_table_sel",
                type="primary",
                on_click=_apply_table_selection,
                args=(table_selected_indices,),
                use_container_width=True,
            )

    # ── Comparaison graphique interactive si plusieurs joueurs filtrés ────────
    if 2 <= len(f) <= 30:
        with st.expander(
            f"📊 Comparaison graphique ({len(f)} joueurs)",
            expanded=bool(player_sel or table_selected_indices),
        ):
            chart_names = [f"{r['name']} ({r['pos']}, {r['team']})" for _, r in f.iterrows()]
            metrics_dict = {
                "Score (0-100)": [round(float(v), 1) for v in f["Score"]],
                "Rythme 82m": [
                    round(float(v), 1)
                    for v in (f["pace82"] if "pace82" in f else f["rate0"] * 82)
                ],
                "Peak projeté": [
                    round(float(v), 1)
                    for v in (f["peak_pts"] if "peak_pts" in f else f["pts0"])
                ],
                "Proj. restant (ROS)": [round(float(v), 1) for v in f["ros"]],
                "Proj. saison": [round(float(v), 1) for v in f["pts0"]],
                "Valeur Keeper": [round(float(v), 1) for v in f["kv"]],
                "VORP": [round(float(v), 1) for v in f["vorp0"]],
            }

            col_opt1, col_opt2 = st.columns([2.5, 1.5])
            with col_opt1:
                sel_metrics = st.multiselect(
                    "Statistiques à comparer :",
                    options=list(metrics_dict.keys()),
                    default=["Score (0-100)", "Rythme 82m", "Peak projeté", "Proj. saison"],
                    key="ranking_chart_metrics_sel",
                )
            with col_opt2:
                chart_mode = st.radio(
                    "Mode d'affichage :",
                    ["📊 Barres groupées", "🎯 Une seule statistique"],
                    horizontal=True,
                    key="ranking_chart_mode",
                )

            if chart_mode == "🎯 Une seule statistique":
                single_metric = st.selectbox(
                    "Statistique à comparer :",
                    options=list(metrics_dict.keys()),
                    index=0,
                    key="ranking_chart_single_metric",
                )
                chart_df = pd.DataFrame({single_metric: metrics_dict[single_metric]}, index=chart_names)
                st.bar_chart(chart_df)
            else:
                if sel_metrics:
                    chart_df = pd.DataFrame({m: metrics_dict[m] for m in sel_metrics}, index=chart_names)
                    st.bar_chart(chart_df)
                else:
                    st.info("Sélectionnez au moins une statistique ci-dessus pour afficher le graphique.")

    # ── Boutons d'export CSV ──────────────────────────────────────────────────
    c_exp1, c_exp2 = st.columns([2, 3])
    with c_exp1:
        st.download_button(
            "⬇️ Exporter le tableau affiché en CSV",
            view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"scores_pool_{cur}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if table_selected_indices:
        view_sel = view.loc[table_selected_indices]
        with c_exp2:
            st.download_button(
                f"⬇️ Exporter uniquement les {len(table_selected_indices)} joueur(s) coché(s) en CSV",
                view_sel.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"selection_joueurs_pool_{cur}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption("Tiers : S = top 2×équipes · A = top 6× · B = top 12× (zone de protection) · C = top 24× · D = reste.")


def render_tab_player(players: pd.DataFrame, model: dict[str, Any], cur: int, is_mobile: bool = False) -> None:
    """Affiche la fiche détaillée d'un joueur, sa trajectoire complète (passé + futur), son historique et ses coéquipiers."""
    labels = {i: f"{r['name']} ({r['pos']}, {r['team']})" for i, r in players.iterrows()}
    idx = st.selectbox("Joueur", list(labels), format_func=lambda i: labels[i])
    r = players.loc[idx]

    P = np.array(r["P"])
    S = np.array(r["S"])
    peak_val = float(P.max())
    peak_k = int(P.argmax())
    peak_age = r["age_cur"] + peak_k
    peak_season = season_shift(cur, peak_k)
    pace82 = r.get("pace82", r["rate0"] * 82)

    if is_mobile:
        m1, m2 = st.columns(2)
        m1.metric("Score", f"{r['Score']:.1f}", f"Rang #{int(r['Rang'])}")
        m2.metric("Tier", str(r["Tier"]))
        m3, m4 = st.columns(2)
        m3.metric("Rythme (82m)", f"{pace82:.0f} pts", f"{r['pts0']:.0f} pts espérés")
        m4.metric("Peak projeté", f"{peak_val:.0f} pts", f"à {peak_age:.0f} ans")
        m5, m6 = st.columns(2)
        m5.metric("Valeur keeper", f"{r['kv']:.0f}")
        m6.metric("Années utiles", f"{r['useful']:.1f}")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Score", f"{r['Score']:.1f}", f"Rang {int(r['Rang'])}")
        m2.metric("Tier", str(r["Tier"]))
        m3.metric("Rythme (82m)", f"{pace82:.0f} pts", f"{r['pts0']:.0f} pts espérés ({r['avail']*100:.0f}% dispo)")
        m4.metric("Peak projeté", f"{peak_val:.0f} pts", f"à {peak_age:.0f} ans")
        m5.metric("Valeur keeper", f"{r['kv']:.0f}")
        m6.metric("Années utiles", f"{r['useful']:.1f}")

    h_all = model["ps"][model["ps"]["pid"] == r["pid"]].sort_values("season", ascending=True)
    h_past = h_all[h_all["gp"] >= 10]
    hist_peak_msg = ""
    if not h_past.empty:
        idx_max = h_past["pool"].idxmax()
        best_s = h_past.loc[idx_max]
        best_pace = (best_s["pool"] / max(best_s["gp"], 1)) * 82
        hist_peak_msg = (
            f" · 🏆 **Record LNH établi :** **{best_s['pool']:.1f} pts** "
            f"({best_pace:.1f} sur 82m) en **{season_label(best_s['season'])}**"
        )

    if peak_k > 0:
        peak_banner = (
            f"🏔️ **Production au sommet (Peak) :** **{peak_val:.1f} pts** de pool projetés par saison "
            f"à l'âge de **{peak_age:.0f} ans** (saison **{season_label(peak_season)}**, dans **{peak_k} an{'s' if peak_k > 1 else ''}**)"
            f"{hist_peak_msg}."
        )
    else:
        peak_banner = (
            f"🏔️ **Production au sommet (Peak) :** **{peak_val:.1f} pts** de pool projetés "
            f"(actuellement au sommet / en phase de plateau offensif)"
            f"{hist_peak_msg}."
        )
    st.success(peak_banner)

    if pd.notna(r.get("pre_nhl")) and r["pre_nhl"]:
        st.info(
            f"🏒 **Parcours pré-LNH :** {r['pre_nhl']} · Taux traduit en LNH : **{r['pre_rate']:.2f}** pts/match "
            f"(équivalence de **{r['pre_gp_eq']:.1f}** matchs LNH prise en compte dans la projection)."
        )

    # ── Trajectoire complète unifiée (Passé Réel + Saison Courante + Projections Futures) ──
    first_nhl_season = h_all["season"].min() if not h_all.empty else cur
    pre_info = model.get("pre_nhl", {}).get(int(r["pid"]), {})
    pre_rows = pre_info.get("rows", [])

    timeline: list[dict[str, Any]] = []

    # 1. Éventuelles saisons pré-LNH antérieures (pour les recrues/sophomores ayant peu de recul LNH)
    if len(h_all) <= 2 and pre_rows:
        pre_by_season: dict[int, dict[str, Any]] = {}
        for pr in pre_rows:
            s_id = pr["season"]
            if s_id < first_nhl_season:
                # Retenir la ligue principale (plus grand nombre de matchs) pour chaque saison
                if s_id not in pre_by_season or pr["gp"] > pre_by_season[s_id]["gp"]:
                    pre_by_season[s_id] = pr
        for s_id in sorted(pre_by_season.keys()):
            pr = pre_by_season[s_id]
            s_lbl = f"{season_label(pr['season'])} [{pr['league']}]"
            timeline.append({
                "Saison": s_lbl,
                "Rythme pré-LNH (traduit 82m)": round(pr["rate_nhl"] * 82, 1),
                "Rythme réel LNH (82m)": np.nan,
                "Points réels LNH": np.nan,
                "Projection modèle (82m)": np.nan,
                "Projection × Survie": np.nan,
                "Niveau de remplacement": round(r["repl"], 1),
            })

    # 2. Saisons LNH passées (< cur)
    for _, row in h_all[h_all["season"] < cur].iterrows():
        s_lbl = f"{season_label(row['season'])} (âge {row['age']:.0f})"
        pace82 = (row["pool"] / max(row["gp"], 1)) * 82
        timeline.append({
            "Saison": s_lbl,
            "Rythme pré-LNH (traduit 82m)": np.nan,
            "Rythme réel LNH (82m)": round(pace82, 1),
            "Points réels LNH": round(row["pool"], 1),
            "Projection modèle (82m)": np.nan,
            "Projection × Survie": np.nan,
            "Niveau de remplacement": round(r["repl"], 1),
        })

    # 3. Saison courante (cur) : point de jonction entre réel actuel et projection fin de saison
    cur_pace = (r["pool_cur"] / max(r["gp_cur"], 1)) * 82 if r["gp_cur"] > 0 else np.nan
    cur_lbl = f"{season_label(cur)} (âge {r['age_cur']:.0f})"
    timeline.append({
        "Saison": cur_lbl,
        "Rythme pré-LNH (traduit 82m)": np.nan,
        "Rythme réel LNH (82m)": round(cur_pace, 1) if np.isfinite(cur_pace) else np.nan,
        "Points réels LNH": round(r["pool_cur"], 1),
        "Projection modèle (82m)": round(P[0], 1),
        "Projection × Survie": round(P[0] * S[0], 1),
        "Niveau de remplacement": round(r["repl"], 1),
    })

    # 4. Saisons futures projetées (k = 1 .. 10)
    for k in range(1, len(P)):
        s_id = season_shift(cur, k)
        age_k = r["age_cur"] + k
        s_lbl = f"{season_label(s_id)} (âge {age_k:.0f})"
        timeline.append({
            "Saison": s_lbl,
            "Rythme pré-LNH (traduit 82m)": np.nan,
            "Rythme réel LNH (82m)": np.nan,
            "Points réels LNH": np.nan,
            "Projection modèle (82m)": round(P[k], 1),
            "Projection × Survie": round(P[k] * S[k], 1),
            "Niveau de remplacement": round(r["repl"], 1),
        })

    df_chart = pd.DataFrame(timeline).set_index("Saison")

    st.subheader("Trajectoire historique et projetée")
    st.line_chart(df_chart, height=300 if is_mobile else 440)
    st.caption(
        "💡 **Guide de lecture du graphique :** "
        "• **Rythme réel LNH (82m)** : Cadence offensive réelle des saisons passées et actuelle. "
        "• **Points réels LNH** : Total brut de points marqués. "
        "• **Projection modèle (82m)** : Trajectoire projetée sans attrition pour un calendrier complet. "
        "• **Projection × Survie** : Espérance de points pondérée par la probabilité de maintien dans la ligue (déclin d'âge). "
        "• **Niveau de remplacement** : Seuil sous lequel un joueur disponible sur le marché ferait aussi bien (VORP)."
    )

    if is_mobile:
        tab_hist, tab_ctx = st.tabs(["📜 Historique LNH", "🏙️ Contexte d'équipe"])
        cL = tab_hist
        cR = tab_ctx
    else:
        cL, cR = st.columns(2)
    with cL:
        st.subheader("Historique LNH")
        h = h_all.sort_values("season", ascending=False)
        hv = pd.DataFrame({
            "Saison": h["season"].map(season_label),
            "GP": h["gp"].astype(int),
            "Pts pool": h["pool"].round(1),
            "Pts/m": h["rate"].round(2),
            "Rythme 82m": ((h["pool"] / np.maximum(h["gp"], 1)) * 82).round(1),
            "TOI": h["toi"].round(1),
            "TOI PP": h["pp_toi"].round(1),
            "Âge": h["age"].round(1),
        })
        st.dataframe(hv, hide_index=True, use_container_width=True)

        if pre_rows:
            with st.expander("📚 Parcours pré-LNH détaillé", expanded=(len(h_all) <= 2)):
                pr_df = pd.DataFrame([
                    {
                        "Saison": season_label(pr["season"]),
                        "Ligue": pr["league"],
                        "GP": pr["gp"],
                        "B": pr.get("g", 0),
                        "P": pr.get("a", 0),
                        "Pts pool": pr["pts"],
                        "Pts/m": pr.get("rate_lg", round(pr["pts"] / max(pr["gp"], 1), 2)),
                        "Traduit LNH (pts/m)": pr["rate_nhl"],
                        "Rythme LNH (82m)": round(pr["rate_nhl"] * 82, 1),
                    }
                    for pr in sorted(pre_rows, key=lambda x: x["season"], reverse=True)
                ])
                st.dataframe(pr_df, hide_index=True, use_container_width=True)

    with cR:
        st.subheader("Contexte d'équipe & Offensive")
        tm_gpg = r.get("tm_gpg", np.nan)
        tm_trend = r.get("tm_trend", 0.0)
        tm_pace_goals = r.get("tm_pace_goals", np.nan)
        tm_ceil = r.get("tm_ceil", np.nan)
        g_pace82 = r.get("g_pace82", np.nan)
        g_trend = r.get("g_trend", 0.0)
        tm_share = r.get("tm_share", np.nan)

        if r["grp"] == "G":
            st.write(
                f"🏒 **Soutien offensif du club ({r['team']}) :** **{tm_gpg:.2f}** buts/m "
                f"({tm_trend:+.2f} vs an dernier) · Réservoir sur 82m : **{tm_pace_goals:.0f}** buts."
            )
            st.write(
                f"Part de départs : **{r['gs_share'] * 100:.0f} %** · "
                f"Taux de victoires : **{r['win_rate'] * 100:.0f} %**"
            )
        else:
            st.write(
                f"🏒 **Offensive du club ({r['team']}) :** **{tm_gpg:.2f}** buts/m "
                f"({tm_trend:+.2f} vs an dernier) · Réservoir sur 82m : **{tm_pace_goals:.0f}** buts."
            )
            cap_label = "52% (IPP max A)" if r["grp"] == "F" else "40% (IPP max D, buts 2 pts)"
            st.write(
                f"🎯 **Plafond offensif estimé ({cap_label}) :** **{tm_ceil:.0f} pts** · "
                f"Part accaparée : **{tm_share * 100:.0f} %** des buts du club."
            )
            st.write(
                f"⚡ **Cadence de buts individuelle :** **{g_pace82:.0f}** buts/82m "
                f"({g_trend:+.2f} b/m vs saison précédente)."
            )

            if pd.notna(tm_share) and (tm_share > 0.40 or (r["grp"] == "D" and tm_share > 0.30)):
                st.warning(
                    f"⚠️ **Balisage offensif élevé :** Ce joueur concentre déjà **{tm_share * 100:.0f} %** "
                    f"de tous les buts de son club (plafond de **{tm_ceil:.0f} pts**). "
                    f"La projection immédiate est balisée par l'attaque du club. Pour atteindre son Peak projeté "
                    f"(**{peak_val:.0f} pts**), une progression offensive de l'équipe sera nécessaire."
                )
            elif pd.notna(tm_trend) and tm_trend >= 0.40:
                st.info(
                    f"🚀 **Vent de dos offensif :** L'attaque de **{r['team']}** progresse fortement "
                    f"(**{tm_trend:+.2f} buts/match** vs l'an dernier), augmentant le réservoir offensif disponible."
                )
            elif pd.notna(tm_trend) and tm_trend <= -0.40:
                st.caption(
                    f"📉 **Ralentissement de l'équipe :** L'attaque de **{r['team']}** est en baisse "
                    f"(**{tm_trend:+.2f} buts/match** vs l'an dernier), ce qui bride temporairement les plafonds immédiats."
                )

            mates = players[(players["team"] == r["team"]) & (players["role"] == r["role"]) & players["actif"]]
            mv = pd.DataFrame({
                "Joueur": mates["name"],
                "Âge": mates["age_cur"].round(1),
                "TOI": mates["toi_w"].round(1),
                "TOI PP": mates["pp_w"].round(1),
                "Pts/60": mates["p60"].round(2),
                "Score": mates["Score"].round(1),
            }).sort_values("TOI", ascending=False)
            st.dataframe(mv, hide_index=True, use_container_width=True)
            if pd.notna(r["age_ahead"]):
                st.write(
                    f"Opportunité **{r['c_opp']:.0f}/100** · "
                    f"Coéquipiers devant lui : **{int(r['n_ahead']) if pd.notna(r['n_ahead']) else 0}** "
                    f"(âge moyen {r['age_ahead']:.1f})"
                )
            else:
                st.write(f"Opportunité **{r['c_opp']:.0f}/100** · Aucun coéquipier de même poste devant lui.")


def render_tab_scarcity(players: pd.DataFrame, model: dict[str, Any], is_mobile: bool = False) -> None:
    """Affiche l'analyse de rareté par position, courbes de chute et VORP."""
    st.subheader("Rareté et niveau de remplacement")
    st.dataframe(model["groups"], hide_index=True, use_container_width=True)
    st.caption("« Écart élite – remplacement » grand = position où les vrais bons joueurs sont rares et font la différence.")

    lim = {"F": 220, "D": 110, "G": 50}
    curves = {}
    for g, nm in (("F", "Attaquants"), ("D", "Défenseurs"), ("G", "Gardiens")):
        v = np.sort(players.loc[players["grp"] == g, "pts0"].values)[::-1][: lim[g]]
        curves[nm] = pd.Series(v, index=np.arange(1, len(v) + 1))
    st.subheader("Chute de production par rang (pts pool projetés)")
    st.line_chart(pd.DataFrame(curves), height=280 if is_mobile else 380)

    st.subheader("Meilleure valeur (VORP) par position")
    pos_items = [("F", "Attaquants"), ("D", "Défenseurs"), ("G", "Gardiens")]
    cols_vorp = ["Joueur", "Éq.", "Âge", "Proj. saison", "Peak", "VORP", "Valeur keeper", "Score", "Rang pos"]

    if is_mobile:
        pos_tabs = st.tabs([nm for _, nm in pos_items])
        for tab_p, (g, nm) in zip(pos_tabs, pos_items):
            with tab_p:
                t = players[players["grp"] == g].sort_values("vorp0", ascending=False).head(15)
                st.dataframe(
                    to_view(t)[cols_vorp],
                    hide_index=True,
                    use_container_width=True,
                )
    else:
        for g, nm in pos_items:
            t = players[players["grp"] == g].sort_values("vorp0", ascending=False).head(15)
            st.markdown(f"**{nm}**")
            st.dataframe(
                to_view(t)[cols_vorp],
                hide_index=True,
                use_container_width=True,
            )


CLUB_STORAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_club.json"
)


def save_club_to_disk(players: pd.DataFrame, club_indices: list[int], path: str = CLUB_STORAGE_PATH) -> bool:
    """Sauvegarde la liste des joueurs du club sur le disque."""
    try:
        valid_indices = [i for i in club_indices if i in players.index]
        pids = [int(players.loc[i, "pid"]) for i in valid_indices if "pid" in players.columns]
        names = [str(players.loc[i, "name"]) for i in valid_indices if "name" in players.columns]
        payload = {
            "version": 1,
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pids": pids,
            "names": names,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_club_from_disk(players: pd.DataFrame, path: str = CLUB_STORAGE_PATH) -> tuple[list[int], str | None]:
    """Charge la liste des joueurs du club depuis le disque. Retourne (indices, updated_at)."""
    if not os.path.exists(path):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_pids = set(data.get("pids", []))
        updated_at = data.get("updated_at")

        indices: list[int] = []
        if saved_pids and "pid" in players.columns:
            indices = players[players["pid"].isin(saved_pids)].index.tolist()

        if len(indices) < len(data.get("names", [])):
            missing_names = [n for n in data.get("names", []) if n not in players.loc[indices, "name"].values]
            if missing_names:
                extra_ids, _ = match_names(players, missing_names)
                for eid in extra_ids:
                    if eid not in indices:
                        indices.append(eid)

        return indices, updated_at
    except Exception:
        return [], None


def ensure_club_loaded(players: pd.DataFrame) -> None:
    """S'assure que le club est chargé en mémoire au démarrage (URL ou fichier disque)."""
    if "my_club_ids" not in st.session_state:
        url_club = st.query_params.get("club")
        indices_from_url: list[int] = []
        if url_club:
            try:
                url_pids = [int(p.strip()) for p in url_club.split(",") if p.strip()]
                if "pid" in players.columns:
                    indices_from_url = players[players["pid"].isin(url_pids)].index.tolist()
            except Exception:
                indices_from_url = []

        if indices_from_url:
            st.session_state["my_club_ids"] = sorted(indices_from_url)
            st.session_state["my_club_last_saved"] = "Restauré depuis l'URL"
        else:
            saved_ids, updated_at = load_club_from_disk(players)
            st.session_state["my_club_ids"] = sorted(saved_ids)
            if updated_at:
                st.session_state["my_club_last_saved"] = updated_at


def sync_club_state(players: pd.DataFrame, new_ids: list[int]) -> None:
    """Met à jour l'état de session, enregistre sur disque et synchronise les query params d'URL."""
    valid_ids = sorted([i for i in new_ids if i in players.index])
    st.session_state["my_club_ids"] = valid_ids
    save_club_to_disk(players, valid_ids)
    st.session_state["my_club_last_saved"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if valid_ids and "pid" in players.columns:
        pids_str = ",".join(str(int(players.loc[i, "pid"])) for i in valid_ids)
        if st.query_params.get("club") != pids_str:
            st.query_params["club"] = pids_str
    elif "club" in st.query_params:
        del st.query_params["club"]


def render_tab_club_waivers(players: pd.DataFrame, is_mobile: bool = False) -> None:
    """Affiche le gestionnaire de club, sélection des joueurs, protégés recommandés et ballottage."""
    st.subheader("🧢 Gestion de mon club")
    st.write(
        "Sélectionnez vos joueurs pour calculer leurs scores respectifs, identifier vos **12 protégés optimaux** "
        "et trouver les meilleures **cibles de ballottage**."
    )

    ensure_club_loaded(players)

    # ── Barre d'état de persistance & Actions de sauvegarde ───────────────────
    last_saved = st.session_state.get("my_club_last_saved")
    n_players = len(st.session_state.get("my_club_ids", []))

    if is_mobile:
        if last_saved:
            st.success(f"💾 **Club sauvegardé** ({n_players} joueurs) · *{last_saved}*", icon="✅")
        else:
            st.info("ℹ️ Aucun club enregistré. Sélectionnez vos joueurs ci-dessous.", icon="🏒")

        c_save_btn, c_reset_btn = st.columns(2)
        with c_save_btn:
            if st.button("💾 Enregistrer", use_container_width=True, help="Force la sauvegarde immédiate sur le serveur."):
                sync_club_state(players, st.session_state["my_club_ids"])
                st.rerun()
        with c_reset_btn:
            if st.button("🗑️ Vider le club", use_container_width=True, help="Efface la sélection et supprime la sauvegarde persistante."):
                st.session_state["my_club_ids"] = []
                if os.path.exists(CLUB_STORAGE_PATH):
                    try:
                        os.remove(CLUB_STORAGE_PATH)
                    except Exception:
                        pass
                if "club" in st.query_params:
                    del st.query_params["club"]
                st.session_state["my_club_last_saved"] = None
                st.rerun()
    else:
        c_status, c_save_btn, c_reset_btn = st.columns([3, 1.2, 1])
        with c_status:
            if last_saved:
                st.success(f"💾 **Club sauvegardé de façon permanente** ({n_players} joueurs) · *Dernière mise à jour : {last_saved}*", icon="✅")
            else:
                st.info("ℹ️ Aucun club enregistré pour le moment. Sélectionnez vos joueurs ci-dessous.", icon="🏒")

        with c_save_btn:
            if st.button("💾 Enregistrer", use_container_width=True, help="Force la sauvegarde immédiate sur le serveur."):
                sync_club_state(players, st.session_state["my_club_ids"])
                st.rerun()

        with c_reset_btn:
            if st.button("🗑️ Vider le club", use_container_width=True, help="Efface la sélection et supprime la sauvegarde persistante."):
                st.session_state["my_club_ids"] = []
                if os.path.exists(CLUB_STORAGE_PATH):
                    try:
                        os.remove(CLUB_STORAGE_PATH)
                    except Exception:
                        pass
                if "club" in st.query_params:
                    del st.query_params["club"]
                st.session_state["my_club_last_saved"] = None
                st.rerun()

    with st.expander("📁 Exporter / Importer une sauvegarde (Fichier JSON externe)", expanded=False):
        c_exp, c_imp = st.columns(2)
        with c_exp:
            st.markdown("**Télécharger une sauvegarde locale :**")
            valid_ids = [i for i in st.session_state["my_club_ids"] if i in players.index]
            export_payload = {
                "version": 1,
                "exported_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pids": [int(players.loc[i, "pid"]) for i in valid_ids if "pid" in players.columns],
                "names": [str(players.loc[i, "name"]) for i in valid_ids if "name" in players.columns],
            }
            st.download_button(
                "⬇️ Télécharger my_club.json",
                json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="my_club.json",
                mime="application/json",
                use_container_width=True,
                help="Conservez une copie sur votre ordinateur ou incluez ce fichier dans votre déploiement git.",
            )
        with c_imp:
            st.markdown("**Restaurer depuis un fichier JSON :**")
            uploaded_file = st.file_uploader(
                "Glissez un fichier JSON de sauvegarde",
                type=["json"],
                label_visibility="collapsed",
                key="club_file_uploader",
            )
            if uploaded_file is not None:
                file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("_last_club_upload_sig") != file_sig:
                    try:
                        upload_data = json.load(uploaded_file)
                        up_pids = set(upload_data.get("pids", []))
                        up_indices: list[int] = []
                        if up_pids and "pid" in players.columns:
                            up_indices = players[players["pid"].isin(up_pids)].index.tolist()
                        if len(up_indices) < len(upload_data.get("names", [])):
                            missing = [n for n in upload_data.get("names", []) if n not in players.loc[up_indices, "name"].values]
                            if missing:
                                extra_ids, _ = match_names(players, missing)
                                for eid in extra_ids:
                                    if eid not in up_indices:
                                        up_indices.append(eid)
                        if up_indices:
                            st.session_state["_last_club_upload_sig"] = file_sig
                            sync_club_state(players, up_indices)
                            st.success(f"✅ {len(up_indices)} joueurs restaurés avec succès !")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du fichier : {e}")
                else:
                    st.caption(f"📁 Fichier chargé : **{uploaded_file.name}** ({len(st.session_state.get('my_club_ids', []))} joueurs restaurés)")
            else:
                st.session_state["_last_club_upload_sig"] = None

    all_indices = players.index.tolist()
    labels = {
        i: f"{players.loc[i, 'name']} ({players.loc[i, 'pos']}, {players.loc[i, 'team']}) – Score: {players.loc[i, 'Score']:.1f} | Rang #{int(players.loc[i, 'Rang'])}"
        for i in all_indices
    }

    # Mode d'entrée
    mode = st.radio(
        "Mode d'entrée :",
        ["🔍 Sélection interactive (Recherche & clic)", "📋 Coller une liste de noms"],
        horizontal=True,
    )

    if mode == "🔍 Sélection interactive (Recherche & clic)":
        current_club_set = set(st.session_state.get("my_club_ids", []))
        selected = st.multiselect(
            "Joueurs de mon équipe :",
            options=all_indices,
            default=[i for i in st.session_state["my_club_ids"] if i in all_indices],
            format_func=lambda i: labels.get(i, str(i)),
            placeholder="Tapez le nom d'un joueur, sa position ou son équipe...",
        )
        if set(selected) != current_club_set:
            sync_club_state(players, selected)
            st.rerun()
    else:
        c_txt, c_btn = st.columns([4, 1])
        default_txt = (
            "\n".join(players.loc[st.session_state["my_club_ids"], "name"])
            if st.session_state["my_club_ids"]
            else ""
        )
        mine_txt = c_txt.text_area(
            "Collez votre alignement (un joueur par ligne)",
            value=default_txt,
            height=180,
            placeholder="Connor McDavid\nCale Makar\nIgor Shesterkin\n...",
        )
        if c_btn.button("📥 Importer la liste", use_container_width=True):
            if mine_txt.strip():
                ids, miss = match_names(players, parse_names(mine_txt))
                sync_club_state(players, ids)
                if miss:
                    st.warning("Joueurs non trouvés : " + ", ".join(miss))
                st.rerun()

    club_ids = [i for i in st.session_state["my_club_ids"] if i in players.index]

    if club_ids:
        mine = players.loc[club_ids].sort_values("Rang")

        # Résumé chiffré de l'effectif
        st.markdown("### 📊 Portrait de mon effectif")
        if is_mobile:
            k1, k2 = st.columns(2)
            k1.metric("Joueurs", f"{len(mine)}")
            k2.metric("Score moyen", f"{mine['Score'].mean():.1f}")
            k3, k4 = st.columns(2)
            k3.metric("Valeur Keeper", f"{mine['kv'].sum():.0f}")
            k4.metric("Pts restants proj.", f"{mine['ros'].sum():.0f}")
            k5, _ = st.columns(2)
            k5.metric("Âge moyen", f"{mine['age_cur'].mean():.1f} ans")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Joueurs", f"{len(mine)}")
            k2.metric("Score moyen", f"{mine['Score'].mean():.1f}")
            k3.metric("Valeur Keeper totale", f"{mine['kv'].sum():.0f}")
            k4.metric("Points restants proj.", f"{mine['ros'].sum():.0f}")
            k5.metric("Âge moyen", f"{mine['age_cur'].mean():.1f} ans")

        # Protégés recommandés
        keep, ng = [], 0
        for i, row in mine.iterrows():
            if len(keep) >= KEEPERS:
                break
            if row["grp"] == "G" and ng >= 3:
                continue
            keep.append(i)
            ng += row["grp"] == "G"

        if is_mobile:
            tab_keep, tab_opt = st.tabs([f"🔒 Protégés ({len(keep)}/{KEEPERS})", "🧊 Alignement partant"])
            col_left = tab_keep
            col_right = tab_opt
        else:
            col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader(f"🔒 Protégés recommandés ({len(keep)}/{KEEPERS}) – max 3 gardiens")
            st.dataframe(
                to_view(players.loc[keep])[["Joueur", "Pos", "Éq.", "Âge", "Score", "Tier", "Valeur keeper", "Proj. saison", "Peak", "VORP"]],
                hide_index=True,
                use_container_width=True,
                column_config=COLCFG,
            )

        with col_right:
            st.subheader("🧊 Alignement partant optimal (12 A · 6 D · 2 G)")
            lineup = []
            for g in ("F", "D", "G"):
                lineup += mine[mine["grp"] == g].sort_values("pts0", ascending=False).head(SLOTS[g]).index.tolist()
            st.dataframe(
                to_view(players.loc[lineup])[["Joueur", "Pos", "Éq.", "Proj. restant", "Pts pool", "Score"]],
                hide_index=True,
                use_container_width=True,
            )
            bench_ids = [i for i in mine.index if i not in lineup]
            if bench_ids:
                st.caption("Banc : " + ", ".join(players.loc[bench_ids, "name"]))

        # Tous les joueurs du club avec détail complet
        with st.expander("📋 Voir le tableau complet de mon club (tous les indicateurs)", expanded=False):
            st.dataframe(to_view(mine), hide_index=True, use_container_width=True, column_config=COLCFG)
    else:
        st.info("💡 Sélectionnez ou collez vos joueurs ci-dessus pour afficher l'analyse complète de votre club.")

    # Cibles de ballottage
    st.divider()
    st.subheader("🎯 Cibles de ballottage")
    st.write("Trouvez les meilleurs joueurs disponibles pour bonifier votre équipe (fenêtres : 15 novembre, 15 janvier, 15 mars).")
    taken_txt = st.text_area(
        "Joueurs déjà repêchés ou pris par les autres DG (optionnel, un par ligne pour les exclure du ballottage)",
        height=100,
        placeholder="Nathan MacKinnon\nAuston Matthews\n...",
    )
    used = set(club_ids)
    if taken_txt.strip():
        other_ids, _ = match_names(players, parse_names(taken_txt))
        used.update(other_ids)
    pool_free = players.drop(index=list(used)) if used else players
    wv = pool_free.sort_values("ros", ascending=False).head(30)
    st.dataframe(
        to_view(wv)[["Joueur", "Pos", "Éq.", "Âge", "Pts pool", "Proj. restant", "Pts/60", "TOI", "Opport.", "Score"]],
        hide_index=True,
        use_container_width=True,
        column_config=COLCFG,
    )


def _render_manual_trade_simulator(players: pd.DataFrame, is_mobile: bool = False) -> None:
    """Affiche le simulateur interactif d'échanges avec comparaison multi-critères."""
    st.subheader("🔄 Simulateur d'échange")
    st.write(
        "Simulez et évaluez des transactions entre deux équipes. Comparez en temps réel le gain de **points immédiats**, "
        "la **valeur keeper future**, le **score global**, et obtenez un diagnostic stratégique (Win-Now vs Reconstruction)."
    )

    ensure_club_loaded(players)

    if "trade_a" not in st.session_state:
        st.session_state["trade_a"] = []
    if "trade_b" not in st.session_state:
        st.session_state["trade_b"] = []

    # Barre d'actions rapides
    if is_mobile:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📥 Charger Mon club", use_container_width=True):
                if "my_club_ids" in st.session_state and st.session_state["my_club_ids"]:
                    st.session_state["trade_a"] = list(st.session_state["my_club_ids"])
                    st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
                    st.rerun()
                else:
                    st.info("Aucun joueur défini dans 'Mon club'.")
        with b2:
            if st.button("↔️ Inverser équipes", use_container_width=True):
                temp = list(st.session_state["trade_a"])
                st.session_state["trade_a"] = list(st.session_state["trade_b"])
                st.session_state["trade_b"] = temp
                st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
                st.rerun()
        if st.button("🗑️ Vider la transaction", use_container_width=True):
            st.session_state["trade_a"] = []
            st.session_state["trade_b"] = []
            st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
            st.rerun()
    else:
        b_col1, b_col2, b_col3, _ = st.columns([2.5, 2, 2, 3.5])
        with b_col1:
            if st.button("📥 Charger Mon club dans Équipe A", use_container_width=True):
                if "my_club_ids" in st.session_state and st.session_state["my_club_ids"]:
                    st.session_state["trade_a"] = list(st.session_state["my_club_ids"])
                    st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
                    st.rerun()
                else:
                    st.info("Aucun joueur défini dans 'Mon club' pour l'instant.")

        if b_col2.button("↔️ Inverser les équipes", use_container_width=True):
            temp = list(st.session_state["trade_a"])
            st.session_state["trade_a"] = list(st.session_state["trade_b"])
            st.session_state["trade_b"] = temp
            st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
            st.rerun()

        if b_col3.button("🗑️ Vider la transaction", use_container_width=True):
            st.session_state["trade_a"] = []
            st.session_state["trade_b"] = []
            st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
            st.rerun()

    all_indices = players.index.tolist()
    labels = {
        i: f"{players.loc[i, 'name']} ({players.loc[i, 'pos']}, {players.loc[i, 'team']}) – Score: {players.loc[i, 'Score']:.1f} | Rang #{int(players.loc[i, 'Rang'])}"
        for i in all_indices
    }

    # Sélecteurs côte à côte ou onglets sur mobile
    if is_mobile:
        tab_sel_a, tab_sel_b = st.tabs([
            f"📤 Équipe A ({len(st.session_state.get('trade_a', []))} cédé)",
            f"📥 Équipe B ({len(st.session_state.get('trade_b', []))} reçu)",
        ])
        col_a = tab_sel_a
        col_b = tab_sel_b
    else:
        col_a, col_b = st.columns(2)

    ver = st.session_state.get("trade_version", 0)
    with col_a:
        st.markdown("#### 📤 Équipe A (Ce que vous cédez)")
        selected_a = st.multiselect(
            "Joueurs cédés par l'Équipe A :",
            options=all_indices,
            default=[i for i in st.session_state["trade_a"] if i in all_indices],
            format_func=lambda i: labels.get(i, str(i)),
            key=f"trade_a_multiselect_{ver}",
            placeholder="Rechercher des joueurs à donner...",
        )
        st.session_state["trade_a"] = selected_a

    with col_b:
        st.markdown("#### 📥 Équipe B (Ce que vous recevez)")
        selected_b = st.multiselect(
            "Joueurs acquis de l'Équipe B :",
            options=all_indices,
            default=[i for i in st.session_state["trade_b"] if i in all_indices],
            format_func=lambda i: labels.get(i, str(i)),
            key=f"trade_b_multiselect_{ver}",
            placeholder="Rechercher des joueurs à recevoir...",
        )
        st.session_state["trade_b"] = selected_b

    # Avertissement doublon
    overlap = set(selected_a).intersection(set(selected_b))
    if overlap:
        st.warning(f"⚠️ Même joueur sélectionné des deux côtés : {', '.join(players.loc[list(overlap), 'name'])}")

    df_a = players.loc[selected_a] if selected_a else pd.DataFrame(columns=players.columns)
    df_b = players.loc[selected_b] if selected_b else pd.DataFrame(columns=players.columns)

    if not df_a.empty or not df_b.empty:
        comp = compare_trade(df_a, df_b)

        st.divider()
        st.subheader("⚖️ Bilan comparatif de l'échange")

        # Cartes métriques comparatives
        if is_mobile:
            m1, m2 = st.columns(2)
            m1.metric(
                label="Score global",
                value=f"{comp['side_b']['score_sum']:.1f}",
                delta=f"{comp['delta_score']:+.1f} vs Éq. A ({comp['side_a']['score_sum']:.1f})",
            )
            m2.metric(
                label="Valeur Keeper",
                value=f"{comp['side_b']['kv_sum']:.0f}",
                delta=f"{comp['delta_kv']:+.0f} vs Éq. A ({comp['side_a']['kv_sum']:.0f})",
            )
            m3, m4 = st.columns(2)
            m3.metric(
                label="Points restants",
                value=f"{comp['side_b']['ros_sum']:.1f}",
                delta=f"{comp['delta_ros']:+.1f} vs Éq. A ({comp['side_a']['ros_sum']:.1f})",
            )
            m4.metric(
                label="Âge moyen",
                value=f"{comp['side_b']['age_mean']:.1f} ans" if comp["side_b"]["count"] else "-",
                delta=(
                    f"{comp['delta_age']:+.1f} ans vs Éq. A ({comp['side_a']['age_mean']:.1f})"
                    if comp["side_a"]["count"] and comp["side_b"]["count"]
                    else None
                ),
                delta_color="inverse",
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                label="Score global cumulé",
                value=f"{comp['side_b']['score_sum']:.1f}",
                delta=f"{comp['delta_score']:+.1f} vs Équipe A ({comp['side_a']['score_sum']:.1f})",
            )
            m2.metric(
                label="Valeur Keeper (Long terme)",
                value=f"{comp['side_b']['kv_sum']:.0f}",
                delta=f"{comp['delta_kv']:+.0f} vs Équipe A ({comp['side_a']['kv_sum']:.0f})",
            )
            m3.metric(
                label="Points restants (Cette saison)",
                value=f"{comp['side_b']['ros_sum']:.1f}",
                delta=f"{comp['delta_ros']:+.1f} vs Équipe A ({comp['side_a']['ros_sum']:.1f})",
            )
            m4.metric(
                label="Âge moyen",
                value=f"{comp['side_b']['age_mean']:.1f} ans" if comp["side_b"]["count"] else "-",
                delta=(
                    f"{comp['delta_age']:+.1f} ans vs Équipe A ({comp['side_a']['age_mean']:.1f})"
                    if comp["side_a"]["count"] and comp["side_b"]["count"]
                    else None
                ),
                delta_color="inverse",
            )

        # Rapport et diagnostic stratégique
        st.markdown("#### 🧠 Diagnostic stratégique")
        st.info(comp["profile_type"])

        c_v1, c_v2 = st.columns(2)
        c_v1.markdown(f"**Court terme (Saison en cours)**\n\n{comp['verdict_short']}")
        c_v2.markdown(f"**Long terme (Horizon Keeper)**\n\n{comp['verdict_long']}")

        if comp["notes"]:
            for note in comp["notes"]:
                st.write(note)

        # Graphique comparatif
        st.markdown("#### 📊 Comparaison graphique")
        chart_data = pd.DataFrame(
            {
                "Équipe A (Cédé)": [
                    comp["side_a"]["score_sum"],
                    comp["side_a"]["kv_sum"],
                    comp["side_a"]["ros_sum"],
                    comp["side_a"]["vorp_sum"],
                ],
                "Équipe B (Reçu)": [
                    comp["side_b"]["score_sum"],
                    comp["side_b"]["kv_sum"],
                    comp["side_b"]["ros_sum"],
                    comp["side_b"]["vorp_sum"],
                ],
            },
            index=["Score total", "Valeur Keeper (kv)", "Pts restants (ROS)", "VORP saison"],
        )
        st.bar_chart(chart_data)

        # Détail des joueurs dans l'échange
        st.markdown("#### 👥 Détail des joueurs impliqués")
        if is_mobile:
            tab_dt_a, tab_dt_b = st.tabs([f"📤 Équipe A ({len(df_a)})", f"📥 Équipe B ({len(df_b)})"])
            col_t1 = tab_dt_a
            col_t2 = tab_dt_b
        else:
            col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown(f"**Équipe A : {len(df_a)} joueur(s)**")
            if not df_a.empty:
                st.dataframe(
                    to_view(df_a)[["Joueur", "Pos", "Éq.", "Âge", "Score", "Tier", "Valeur keeper", "Proj. restant", "Peak", "VORP"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config=COLCFG,
                )
        with col_t2:
            st.markdown(f"**Équipe B : {len(df_b)} joueur(s)**")
            if not df_b.empty:
                st.dataframe(
                    to_view(df_b)[["Joueur", "Pos", "Éq.", "Âge", "Score", "Tier", "Valeur keeper", "Proj. restant", "Peak", "VORP"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config=COLCFG,
                )
    else:
        st.info("💡 Ajoutez des joueurs dans l'Équipe A et l'Équipe B ci-dessus pour simuler et comparer la transaction.")


def render_tab_trade_finder(players: pd.DataFrame, is_mobile: bool = False) -> None:
    """Affiche le générateur automatique d'offres d'échange (Trade Finder)."""
    st.subheader("🎯 Chercheur d'offres (Trade Finder)")
    st.write(
        "Sélectionnez un ou plusieurs joueurs cibles que vous souhaitez acquérir. "
        "L'algorithme explore les combinaisons optimales de votre alignement pour vous proposer "
        "des offres équilibrées et rentables selon le Score, la valeur Keeper et la production restante (ROS)."
    )

    ensure_club_loaded(players)
    all_indices = players.index.tolist()
    labels = {
        i: f"{players.loc[i, 'name']} ({players.loc[i, 'pos']}, {players.loc[i, 'team']}) – Score: {players.loc[i, 'Score']:.1f} | Rang #{int(players.loc[i, 'Rang'])}"
        for i in all_indices
    }

    # 1. Alignement de l'utilisateur
    my_club_ids = st.session_state.get("my_club_ids", [])
    if not my_club_ids:
        st.info("💡 **Astuce** : Définissez votre équipe dans l'onglet **'🧢 Mon club & ballottage'** pour charger automatiquement vos joueurs ici.")

    exp_label = f"🧢 Votre équipe ({len(my_club_ids)} joueurs détectés)" if my_club_ids else "🧢 Sélectionner votre équipe"
    with st.expander(exp_label, expanded=not bool(my_club_ids)):
        roster_ids = st.multiselect(
            "Joueurs de votre équipe pouvant être offerts :",
            options=all_indices,
            default=[i for i in my_club_ids if i in all_indices],
            format_func=lambda i: labels.get(i, str(i)),
            key="finder_roster_select",
            placeholder="Sélectionnez vos joueurs...",
        )

    if not roster_ids:
        st.warning("⚠️ Veuillez sélectionner au moins un joueur dans votre alignement pour pouvoir générer des offres.")
        return

    # 2. Joueur(s) cible(s)
    st.markdown("#### 🎯 Cible convoitée")
    target_options = [i for i in all_indices if i not in roster_ids]
    if not target_options:
        st.warning("Aucun joueur externe disponible à cibler.")
        return

    target_ids = st.multiselect(
        "Joueur(s) convoité(s) de l'autre équipe :",
        options=target_options,
        default=[],
        format_func=lambda i: labels.get(i, str(i)),
        key="finder_target_select",
        placeholder="Tapez le nom d'un joueur convoité (ex: McDavid, Makar, Hughes)...",
    )

    if not target_ids:
        st.info("👈 Recherchez et sélectionnez un joueur cible ci-dessus pour lancer la recherche d'offres.")
        return

    target_df = players.loc[target_ids]
    target_score = float(target_df["Score"].sum())
    target_kv = float(target_df["kv"].sum())
    target_ros = float(target_df["ros"].sum())

    # Aperçu métrique de la cible
    if is_mobile:
        m1, m2 = st.columns(2)
        m1.metric("Score cible", f"{target_score:.1f}")
        m2.metric("Valeur Keeper cible", f"{target_kv:.0f}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score cible", f"{target_score:.1f}")
        m2.metric("Valeur Keeper cible", f"{target_kv:.0f}")
        m3.metric("Points restants (ROS)", f"{target_ros:.1f}")
        m4.metric("Joueurs ciblés", f"{len(target_df)}")

    # 3. Paramètres de filtrage
    st.markdown("#### ⚙️ Paramètres de recherche")
    if is_mobile:
        p1, p2 = st.columns(2)
        with p1:
            max_give = st.selectbox("Max joueurs cédés :", options=[1, 2, 3], index=1, key="finder_max_give")
        with p2:
            top_n = st.selectbox("Max offres affichées :", options=[5, 10, 15, 20], index=1, key="finder_top_n")

        ratio_range = st.slider(
            "Fourchette de valeur Score offerte vs cible (%) :",
            min_value=70,
            max_value=130,
            value=(92, 115),
            step=1,
            format="%d%%",
            key="finder_ratio_slider",
            help="Exemple : 92% à 115% signifie que vous offrez entre 92% et 115% du Score de la cible.",
        )
    else:
        p1, p2, p3 = st.columns([1.5, 2.5, 1.5])
        with p1:
            max_give = st.selectbox("Max joueurs cédés :", options=[1, 2, 3], index=1, key="finder_max_give")
        with p2:
            ratio_range = st.slider(
                "Fourchette de valeur Score offerte vs cible (%) :",
                min_value=70,
                max_value=130,
                value=(92, 115),
                step=1,
                format="%d%%",
                key="finder_ratio_slider",
                help="Exemple : 92% à 115% signifie que vous offrez entre 92% et 115% du Score de la cible.",
            )
        with p3:
            top_n = st.selectbox("Max offres affichées :", options=[5, 10, 15, 20, 30], index=1, key="finder_top_n")

    # Joueurs protégés
    protected_ids = st.multiselect(
        "🔒 Joueurs intouchables de votre club (exclus de toutes les offres) :",
        options=roster_ids,
        default=[],
        format_func=lambda i: players.loc[i, "name"],
        key="finder_protected_ids",
        placeholder="Sélectionnez vos intouchables...",
    )

    # 4. Exécution de suggest_offers
    roster_df = players.loc[roster_ids]
    min_ratio = float(ratio_range[0]) / 100.0
    max_ratio = float(ratio_range[1]) / 100.0

    offers = suggest_offers(
        target_df,
        roster_df,
        name_col="name",
        max_give=int(max_give),
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        protected=protected_ids,
        top_n=int(top_n),
    )

    st.divider()
    st.markdown("#### 📋 Combinaisons d'échanges suggérées")

    if offers.empty:
        st.warning(
            "Aucune offre trouvée dans cette fourchette de valeur.\n\n"
            "💡 **Suggestions** : Élargissez la fourchette de pourcentage (ex: 85% à 120%), "
            "autorisez 3 joueurs offerts, ou retirez certains joueurs de votre liste d'intouchables."
        )
        return

    st.success(f"🎯 **{len(offers)} offre(s) rentable(s) et équilibrée(s)** trouvée(s) !")

    display_cols = [
        "Tu offres",
        "Nb",
        "Score offert",
        "Score cible",
        "Ratio",
        "Δ Score",
        "Δ Keeper",
        "Δ ROS",
        "Δ VORP",
        "Δ Âge",
        "Profil",
        "Notes",
    ]
    table_df = offers[[c for c in display_cols if c in offers.columns]]

    finder_col_cfg = {
        "Tu offres": st.column_config.TextColumn("Joueurs à offrir", width="large"),
        "Nb": st.column_config.NumberColumn("Qté", width="small", format="%d"),
        "Score offert": st.column_config.NumberColumn("Score offert", format="%.1f"),
        "Score cible": st.column_config.NumberColumn("Score cible", format="%.1f"),
        "Ratio": st.column_config.NumberColumn("Ratio", format="%.2f", help="Score offert / Score cible"),
        "Δ Score": st.column_config.NumberColumn("Δ Score", format="%+.1f"),
        "Δ Keeper": st.column_config.NumberColumn("Δ Keeper", format="%+.1f"),
        "Δ ROS": st.column_config.NumberColumn("Δ ROS", format="%+.1f"),
        "Δ VORP": st.column_config.NumberColumn("Δ VORP", format="%+.1f"),
        "Δ Âge": st.column_config.NumberColumn("Δ Âge", format="%+.1f ans"),
        "Profil": st.column_config.TextColumn("Diagnostic", width="medium"),
        "Notes": st.column_config.TextColumn("Remarques", width="large"),
    }

    st.dataframe(
        table_df,
        column_config=finder_col_cfg,
        use_container_width=True,
        hide_index=True,
    )

    # 5. Passerelle vers le simulateur détaillé
    st.markdown("#### 🔬 Inspecter une offre dans le simulateur complet")
    c_off_sel, c_off_btn = st.columns([3.5, 1.5] if not is_mobile else [1, 1])
    with c_off_sel:
        offer_idx = st.selectbox(
            "Sélectionner l'offre à envoyer dans le simulateur :",
            options=range(len(offers)),
            format_func=lambda idx: (
                f"Offre #{idx+1} : {offers.loc[idx, 'Tu offres']} "
                f"(Δ Score: {offers.loc[idx, 'Δ Score']:+.1f} | "
                f"Δ Keeper: {offers.loc[idx, 'Δ Keeper']:+.1f} | "
                f"Δ ROS: {offers.loc[idx, 'Δ ROS']:+.1f})"
            ),
            key="finder_load_offer_idx",
        )

    with c_off_btn:
        if st.button("🚀 Charger dans le simulateur", use_container_width=True, type="primary"):
            chosen_give_ids = offers.loc[offer_idx, "_give_ids"]
            st.session_state["trade_a"] = list(chosen_give_ids)
            st.session_state["trade_b"] = list(target_ids)
            st.session_state["trade_version"] = st.session_state.get("trade_version", 0) + 1
            st.success("✅ Offre chargée ! Basculez sur l'onglet '🔄 Simulateur manuel' pour voir tous les graphiques et le diagnostic complet.")
            st.rerun()


def render_tab_trade_simulator(players: pd.DataFrame, is_mobile: bool = False) -> None:
    """Affiche le simulateur interactif d'échanges et le chercheur d'offres (Trade Finder)."""
    sub_tab_sim, sub_tab_finder = st.tabs([
        "🔄 Simulateur manuel",
        "🎯 Chercheur d'offres (Trade Finder)",
    ])
    with sub_tab_sim:
        _render_manual_trade_simulator(players, is_mobile=is_mobile)
    with sub_tab_finder:
        render_tab_trade_finder(players, is_mobile=is_mobile)


def render_tab_methodology(model: dict[str, Any] | None = None) -> None:
    """Affiche la documentation méthodologique complète et les résultats du backtest / IA."""
    if model and not model.get("backtest", pd.DataFrame()).empty:
        st.subheader("🔬 Modélisation IA Hybride (Gradient Boosting & Courbes d'âge) – Validation Backtest")
        st.markdown(
            "Le moteur utilise une **architecture hybride** : un modèle d'apprentissage machine par **Gradient Boosting (GBDT / LightGBM)** "
            "pour estimer le rythme offensif intrinsèque de base ($r_0$), combiné aux **courbes de vieillissement empiriques** pour projeter la trajectoire "
            "pluriannuelle sur 5 à 10 ans sans dérive.\n\n"
            "Chaque saison passée a été testée en **backtest walk-forward** (on prédit la saison $t+1$ sans aucune fuite d'information future) "
            "pour mesurer l'erreur absolue moyenne (MAE en points) face aux projections naïves."
        )

        c_bt, c_gr = st.columns([3, 2])
        with c_bt:
            st.markdown("##### 📈 Performance du Backtest Walk-Forward")
            st.dataframe(model["backtest"], hide_index=True, use_container_width=True)
            st.caption("Gain vs meilleur naïf : réduction de l'erreur absolue par rapport à la saison précédente ou la moyenne 3 ans.")

        with c_gr:
            st.markdown("##### ⚙️ Grille de Calibration (Top configurations)")
            st.dataframe(model["grid"], hide_index=True, use_container_width=True)
            par = model.get("params")
            if par:
                st.caption(f"Configuration optimale retenue : **Décroissance {par.decay}**, **Facteur prior K {par.k_mult}**, **Méthode : {par.feats.upper()}**")

        imps = model.get("importances", {})
        if imps:
            st.markdown("##### 🌲 Facteurs clés d'apprentissage du Gradient Boosting (Importance des variables)")
            st.caption("Pondération relative des prédicteurs identifiés par le modèle d'arbres décisionnels pour anticiper la production future.")
            cols_imp = st.columns(len(imps))
            for i, (pos_name, imp_dict) in enumerate(imps.items()):
                with cols_imp[i]:
                    st.write(f"**Importance des variables – {pos_name}**")
                    df_imp = pd.DataFrame(list(imp_dict.items()), columns=["Variable", "Importance (%)"]).sort_values("Importance (%)", ascending=False)
                    st.dataframe(df_imp, hide_index=True, use_container_width=True)

        st.divider()

    st.markdown(METHODO)

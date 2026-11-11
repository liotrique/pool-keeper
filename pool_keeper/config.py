# -*- coding: utf-8 -*-
"""
Configuration et constantes du Pool Keeper.
===========================================
Règles de la ligue, URL des API LNH, pondérations et texte méthodologique.
"""
from __future__ import annotations

import os
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# API ET RÉPERTOIRES
# ──────────────────────────────────────────────────────────────────────────────
STATS = "https://api.nhle.com/stats/rest/en"
WEB = "https://api-web.nhle.com/v1"
CACHE_DIR = ".nhl_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# RÈGLEMENTS DE LA LIGUE (Constitution v2.1)
# ──────────────────────────────────────────────────────────────────────────────
RULES = dict(
    F_GOAL=1.0,
    F_ASSIST=1.0,
    D_GOAL=2.0,
    D_ASSIST=1.0,
    G_WIN=1.0,
    G_SAVE=0.14,
    G_GA=-1.0,
)

SLOTS = {"F": 12, "D": 6, "G": 2}
BENCH_DEFAULT = {"F": 2, "D": 1, "G": 1}  # 4 joueurs de banc au repêchage initial
KEEPERS = 12
GAMES = 82
FLOOR_FACTOR = 1.06      # approximation du plancher 0/match si game logs indisponibles
MAX_YEARS = 10           # horizon max pour la longévité
DEFAULT_N_SEASONS = 10   # 10 saisons d'historique par défaut (toutes les saisons pour les modèles globaux)
DEFAULT_N_TEAMS = 12     # 12 équipes dans le pool par défaut
AGES = np.arange(17, 43)
MIN_GP = {"F": 30, "D": 30, "G": 15}
K_PRIOR = {"F": 25.0, "D": 30.0, "G": 15.0}  # poids (en matchs) de la régression vers la moyenne
INFO_BINS = np.array([60.0, 150.0])          # seuils de « matchs d'évidence » : faible (<60) / moyen (60-150) / élevé (>=150)

# ──────────────────────────────────────────────────────────────────────────────
# FACTEURS DE TRADUCTION PRÉ-NHL
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# FACTEURS DE TRADUCTION PRÉ-NHL (CALIBRÉS EMPIRIQUEMENT SUR 691 TRANSITIONS LNH)
# ──────────────────────────────────────────────────────────────────────────────
# Clé = leagueAbbrev de l'API NHL. Valeur = (facteur_taux, poids_évidence).
# Calibré sur l'historique réel 2015-2025 pour capturer la vraie valeur keeper (3 premières années LNH).
LEAGUE_TRANSLATION: dict[str, tuple[float, float]] = {
    # CHL (Junior majeur canadien) : médiane empirique ~0.40 pour F, ~0.34 pour D
    "WHL":   (0.40, 0.24),
    "OHL":   (0.40, 0.24),
    "QMJHL": (0.38, 0.24),
    "LHJMQ": (0.38, 0.24),
    # Pro mineure nord-américaine
    "AHL":   (0.56, 0.38),
    "ECHL":  (0.38, 0.20),
    # Ligues professionnelles européennes adultes élites (médianes réelles 0.68 à 0.82)
    "KHL":   (0.78, 0.35),
    "SHL":   (0.72, 0.35),
    "Liiga": (0.68, 0.32),
    "NL":    (0.58, 0.30),
    "NLA":   (0.58, 0.30),
    "SL":    (0.58, 0.30),
    "DEL":   (0.52, 0.26),
    "Czech": (0.55, 0.30),
    "Czech2": (0.38, 0.22),
    "EIHL":  (0.38, 0.20),
    "Extraliga": (0.55, 0.30),
    # Junior US / Universitaire (calibré empiriquement : NCAA médiane = 0.50)
    "NCAA":  (0.50, 0.30),
    "USHL":  (0.30, 0.18),
    "NAHL":  (0.24, 0.14),
    "BCHL":  (0.25, 0.15),
    "AJHL":  (0.22, 0.14),
    "CCHL":  (0.20, 0.12),
    "OJHL":  (0.20, 0.12),
    "SIJHL": (0.18, 0.10),
    # Programmes nationaux US (NTDP / USNTDP)
    "NTDP":  (0.35, 0.22),
    "USNTDP": (0.35, 0.22),
    # Juniors européens
    "MHL":   (0.28, 0.16),
    "SuperElit": (0.28, 0.16),
    "J20 Nationell": (0.26, 0.15),
    "Mestis": (0.42, 0.22),
    "Allsvenskan": (0.48, 0.25),
    "HockeyAllsvenskan": (0.48, 0.25),
}


def get_league_translation(league: str, rate: float = 1.0, pos: str = "F") -> tuple[float, float]:
    """
    Calcule le facteur de traduction NHLe dynamique selon la ligue, la position et le calibre du marqueur.
    Prend en compte l'élasticité pour les marqueurs juniors d'élite (>= 1.6 ou >= 2.0 PPG) qui obtiennent
    immédiatement un rôle de premier plan (top-6, PP1) en LNH.
    """
    base_trans, evid = LEAGUE_TRANSLATION.get(league, (0.35, 0.20))
    # Ajustement position : les défenseurs produisent légèrement moins en points bruts
    if pos == "D" and league in ("OHL", "WHL", "QMJHL", "LHJMQ", "AHL", "NCAA", "KHL", "SHL"):
        base_trans *= 0.88
    # Ajustement d'élasticité pour les marqueurs dominants juniors (CHL / USHL / BCHL / NTDP)
    if league in ("OHL", "WHL", "QMJHL", "LHJMQ", "USHL", "BCHL", "NTDP", "USNTDP"):
        if rate >= 2.0:
            base_trans *= 1.15  # ex: OHL 2.0 PPG -> facteur effectif 0.46 (75-80 pts en LNH)
        elif rate >= 1.6:
            base_trans *= 1.08  # ex: OHL 1.7 PPG -> facteur effectif 0.43
    return (round(base_trans, 3), evid)

# ──────────────────────────────────────────────────────────────────────────────
# DURÉE OFFICIELLE D'UNE SAISON RÉGULIÈRE PAR LIGUE
# ──────────────────────────────────────────────────────────────────────────────
# Utilisé pour calculer la complétude de la saison (ex: 36 matchs en NCAA = 100% de la saison).
LEAGUE_SCHEDULE: dict[str, int] = {
    "AHL": 72,
    "ECHL": 72,
    "WHL": 68,
    "OHL": 68,
    "QMJHL": 64,
    "LHJMQ": 64,
    "USHL": 62,
    "Liiga": 60,
    "KHL": 68,
    "SHL": 52,
    "NL": 52,
    "NLA": 52,
    "SL": 52,
    "DEL": 52,
    "Czech": 52,
    "Czech2": 52,
    "EIHL": 54,
    "Extraliga": 52,
    "Allsvenskan": 52,
    "HockeyAllsvenskan": 52,
    "NCAA": 36,
    "BCHL": 54,
    "AJHL": 60,
    "NAHL": 60,
    "CCHL": 55,
    "OJHL": 54,
    "SIJHL": 50,
    "NTDP": 55,
    "USNTDP": 55,
    "MHL": 50,
    "SuperElit": 45,
    "J20 Nationell": 45,
    "Mestis": 50,
}

PRE_NHL_MAX_GP_EQ = 40       # plafond d'équivalence en matchs NHL
PRE_NHL_MIN_GP = 5           # matchs minimum dans une ligue pour la compter
PRE_NHL_AGE_CUTOFF = 27      # ne chercher le pré-NHL que pour les joueurs < cet âge

# Configuration d'affichage des colonnes Streamlit avec infobulles détaillées
COLCFG = {
    "Rang": st.column_config.NumberColumn(
        "Rang",
        help="🏆 Rang global du joueur dans le pool, déterminé directement par son Score (ordre de base par défaut).",
        format="%d",
    ),
    "Score": st.column_config.ProgressColumn(
        "Score",
        help="⭐ Valeur clé du classement (0-100). Détermine l'ordre de base. Moyenne pondérée calculée selon vos réglages dans la barre latérale : Valeur Keeper (45%), Saison en cours (20%), Opportunité (15%), Durabilité (10%), Upside (10%).",
        min_value=0,
        max_value=100,
        format="%.1f",
    ),
    "Tier": st.column_config.TextColumn(
        "Tier",
        help="🏷️ Catégorie de statut dans le pool :\n• S : Joueurs de franchise intouchables\n• A : Noyau d'élite\n• B : Joueurs protégeables (zone des 12 protégés)\n• C : Joueurs de repêchage régulier\n• D : Joueurs de ballottage / profondeur.",
    ),
    "Verdict Edge": st.column_config.TextColumn(
        "Verdict Edge",
        help="⚡ Interprétation directe des métriques avancées NHL Edge :\n• 🟢 Buy-Low : Domine la possession ou les tirs mais malchanceux (PDO bas). À acquérir d'urgence avant l'explosion !\n• 🔴 Sell-High : Production dopée par la chance. À échanger au sommet.\n• 🔥 Moteur Élite : Domine la possession et le volume de tirs.\n• 🛡️ Rôle Défensif : Confiné aux missions ingrates.\n• ⚖️ Équilibré : Production conforme à son calibre.",
    ),
    "Joueur": st.column_config.TextColumn(
        "Joueur",
        help="👤 Nom officiel du joueur de la LNH.",
    ),
    "Pos": st.column_config.TextColumn(
        "Pos",
        help="🏒 Position officielle : C (Centre), L (Ailier gauche), R (Ailier droit), D (Défenseur - buts valent 2 pts), G (Gardien).",
    ),
    "Rang pos": st.column_config.NumberColumn(
        "Rang pos",
        help="🥇 Rang du joueur parmi tous les joueurs de sa propre position (ex: 5e défenseur ou 2e gardien de la ligue).",
        format="%d",
    ),
    "Éq.": st.column_config.TextColumn(
        "Éq.",
        help="🏙️ Équipe LNH actuelle du joueur.",
    ),
    "Âge": st.column_config.NumberColumn(
        "Âge",
        help="🎂 Âge exact du joueur au 1er février de la saison courante.",
        format="%.1f",
    ),
    "Valeur keeper": st.column_config.NumberColumn(
        "Valeur keeper",
        help="🔮 Somme actualisée des points au-dessus du remplacement (VORP) sur votre horizon (ex: 5 ans). Pondérée par la probabilité de survie dans la ligue et le facteur d'actualisation.",
        format="%.1f",
    ),
    "Rythme 82m": st.column_config.NumberColumn(
        "Rythme 82m",
        help="⚡ Cadence offensive projetée pour une saison complète de 82 matchs sans blessure (taux de points/match × 82).",
        format="%.1f",
    ),
    "Proj. saison": st.column_config.NumberColumn(
        "Proj. saison",
        help="📈 Espérance réelle de points pour la saison complète, ajustée selon l'historique de santé et la disponibilité prévue (matchs manqués anticipés).",
        format="%.1f",
    ),
    "Peak": st.column_config.NumberColumn(
        "Peak",
        help="🏔️ Production maximale projetée en points de pool au sommet de sa carrière (plafond offensif futur).",
        format="%.1f",
    ),
    "Âge peak": st.column_config.NumberColumn(
        "Âge peak",
        help="🎂 Âge estimé auquel le joueur atteindra son sommet de production.",
        format="%.0f",
    ),
    "Proj. restant": st.column_config.NumberColumn(
        "Proj. restant",
        help="⚡ Points de pool projetés d'ici la fin de la saison actuelle (idéal pour évaluer les échanges et les ballottages immédiats).",
        format="%.1f",
    ),
    "VORP": st.column_config.NumberColumn(
        "VORP",
        help="🛡️ Value Over Replacement Player : Points projetés cette saison au-dessus du joueur de remplacement disponible sur le marché à sa position.",
        format="%.1f",
    ),
    "Pts pool": st.column_config.NumberColumn(
        "Pts pool",
        help="🏒 Points de pool réels déjà accumulés cette saison selon la Constitution (Attaquant: B=1, P=1 | Défenseur: B=2, P=1 | Gardien: V=1, Arrêt=0.1, GA=-1, plancher 0).",
        format="%.1f",
    ),
    "GP": st.column_config.NumberColumn(
        "GP",
        help="📋 Nombre de matchs joués (Games Played) par le joueur durant la saison courante.",
        format="%d",
    ),
    "Années utiles": st.column_config.NumberColumn(
        "Années utiles",
        help="⏳ Nombre prévisionnel d'années futures où ce joueur dépassera le niveau de remplacement de la ligue avant son déclin ou sa retraite.",
        format="%.1f",
    ),
    "Dispo %": st.column_config.NumberColumn(
        "Dispo %",
        help="🏥 Pourcentage estimé de matchs joués par saison (durabilité et historique de santé). 100% = joueur de fer (82 matchs).",
        format="%d%%",
    ),
    "TOI": st.column_config.NumberColumn(
        "TOI",
        help="⏱️ Temps de glace moyen par match en minutes (Time On Ice). Meilleur prédicteur statistique de la production offensive.",
        format="%.1f",
    ),
    "TOI PP": st.column_config.NumberColumn(
        "TOI PP",
        help="🎯 Temps de glace moyen en avantage numérique par match (Power Play TOI). Détermine l'accès à la première vague de jeu de puissance (PP1).",
        format="%.1f",
    ),
    "Pts/60": st.column_config.NumberColumn(
        "Pts/60",
        help="🚀 Rythme de production : points de pool générés par tranche de 60 minutes de jeu.",
        format="%.2f",
    ),
    "Opport.": st.column_config.ProgressColumn(
        "Opport.",
        help="📈 Indice d'opportunité (0-100) : temps de glace, jeu de puissance, rôle dans le trio et âge des vétérans devant lui dans l'équipe.",
        min_value=0,
        max_value=100,
        format="%d",
    ),
    "Durab.": st.column_config.ProgressColumn(
        "Durab.",
        help="🛡️ Indice de durabilité (0-100) : régularité de présence au jeu et espérance d'années utiles restantes.",
        min_value=0,
        max_value=100,
        format="%d",
    ),
    "Upside": st.column_config.ProgressColumn(
        "Upside",
        help="✨ Potentiel de progression (0-100) : croissance projetée par l'âge combinée au potentiel de sous-utilisation (joueur qui produit beaucoup pour son temps de glace).",
        min_value=0,
        max_value=100,
        format="%d",
    ),
    "Pré-NHL": st.column_config.TextColumn(
        "Pré-NHL",
        help="🏒 Ligue pré-NHL principale et matchs joués. Production traduite en équivalent NHL via des facteurs de traduction par ligue pour bonifier la projection des jeunes joueurs.",
    ),
    "Repêchage": st.column_config.TextColumn(
        "Repêchage",
        help="🎯 Rang de sélection officiel au repêchage LNH (#1 au total, année). Les hauts choix bénéficient d'un capital d'opportunité et d'une patience organisationnelle accrue jusqu'à 24 ans.",
    ),
    "Off. Éq.": st.column_config.NumberColumn(
        "Off. Éq.",
        help="🏒 Moyenne de buts marqués par match par l'équipe du joueur cette saison. Mesure le calibre offensif de l'environnement immédiat.",
        format="%.2f",
    ),
    "Corsi CF%": st.column_config.NumberColumn(
        "Corsi CF%",
        help="🏒 Corsi For % (5c5) : pourcentage de tentatives de tir en faveur quand le joueur est sur la glace. >50% = l'équipe génère plus de tirs. Mesure dominante de la possession de rondelle.",
        format="%.1f%%",
    ),
    "PDO": st.column_config.NumberColumn(
        "PDO",
        help="🎰 PDO = % de tir de l'équipe + % d'arrêts de l'équipe quand le joueur est sur la glace. ~1.000 = neutre. >1.020 = chanceux (régression probable). <0.980 = malchanceux (rebond probable).",
        format="%.3f",
    ),
    "OZ Start%": st.column_config.NumberColumn(
        "OZ Start%",
        help="🎯 % de mises en jeu en zone offensive (5c5). >55% = déploiement offensif favorisé par l'entraîneur. <45% = rôle défensif. Impact direct sur la production.",
        format="%.1f%%",
    ),
    "Tirs/m": st.column_config.NumberColumn(
        "Tirs/m",
        help="🎯 Nombre de tirs au but par match. Proxy du volume offensif et de l'agressivité au tir. Les marqueurs d'élite tirent 3.5+/match.",
        format="%.1f",
    ),
    "Sh% 5c5": st.column_config.NumberColumn(
        "Sh% 5c5",
        help="🎯 Pourcentage de tirs convertis en buts à forces égales (5c5). Moyenne LNH ~9%. >12% = possiblement chanceux ou tireur d'élite. <7% = malchanceux ou mauvais finisseur.",
        format="%.1f%%",
    ),
    "On-Ice Sh%": st.column_config.NumberColumn(
        "On-Ice Sh%",
        help="🏒 % de tir de l'équipe quand le joueur est sur la glace à 5c5. Mesure la qualité de la finition collective. Compare à Sh% individuel pour détecter la chance.",
        format="%.1f%%",
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# TEXTE MÉTHODOLOGIQUE
# ──────────────────────────────────────────────────────────────────────────────
METHODO = """
## Ce que le score mesure
Le **Score (0-100)** est une moyenne pondérée (poids modifiables dans la barre latérale) de 5 composantes :

| Composante | Poids défaut | Ce que ça capture |
|---|---|---|
| **Valeur keeper** | 45 % | Somme actualisée des *points au-dessus du remplacement* (VORP) sur les N prochaines saisons, pondérée par la probabilité que le joueur soit encore utile. |
| **Valeur saison actuelle** | 20 % | VORP projeté de la saison en cours (pour les échanges et les ballottages). |
| **Opportunité** | 15 % | Temps de glace, avantage numérique, rôle dans l'équipe, ouverture future (âge des coéquipiers devant lui), tendance du TOI. |
| **Durabilité / longévité** | 10 % | Disponibilité (matchs joués) + nombre d'années utiles restantes. |
| **Upside** | 10 % | Croissance projetée + sous-utilisation (production par 60 min supérieure à ce que le TOI laisse croire). |

## 1. Modélisation hybride : Gradient Boosting (GBDT) + Courbes d'âge
- **Points de pool** = règlements de la Constitution : A/D/G ; **défenseur : but = 2 pts**, ce qui avantage les défenseurs offensifs.
- **Gardiens** : victoire + 0,1 × arrêts − buts accordés, **plancher à 0 par match**. Calculé match par match avec les game logs de l'API (option activée par défaut).
- **Étage 1 – Talent intrinsèque par Gradient Boosting (LightGBM / GBDT)** :
  - Un modèle d'apprentissage machine par arbres boostés capture les interactions non-linéaires complexes entre le temps de glace (TOI global et en avantage numérique PP), l'âge, la cadence de buts et de passes, et les accélérations de rôle ($\Delta \text{TOI}$, $\Delta \text{Pts}$).
  - Il fournit une estimation de haute précision du rythme offensif de base ($r_0$, sur 82 matchs) pour chaque patineur.
- **Étage 2 – Trajectoire pluriannuelle par courbes empiriques** :
  - Une fois le talent de base calibré, le modèle déroule la trajectoire pluriannuelle ($t+1 \dots t+10$) à l'aide des courbes de développement (18-24 ans), du plateau doré (25-30 ans) et de déclin (31+ ans) calculées sur plus de 15 000 transitions joueur-saison.
- **Probabilité de survie et rétention LNH ($S_k$)** :
  - L'espérance de chaque saison future est pondérée par la probabilité que le joueur conserve un poste régulier dans la LNH, éliminant la surévaluation des vétérans en fin de carrière.

## 2. Historique complet (10 saisons) vs Récence individuelle
- **Historique complet (10 saisons) pour l'apprentissage global :**
  - **Entraînement du Gradient Boosting (GBDT)** : apprentissage des relations entre volume de jeu, âge et production sur l'ensemble des transitions historiques.
  - **Courbes d'âge empiriques (build_curves)** : modélisation fidèle de l'ascension et du déclin.
  - **Courbes de rétention et survie dans la LNH** : basées sur 10 ans pour évaluer fidèlement la longévité de carrière.
  - **Normalisation d'ère de la ligue** : étalonne l'offensive globale (de 2.70 à 3.15+ buts/match) sur toute la période.
  - **Traductions pré-LNH** : conversion standardisée des ligues juniors, collégiales et européennes.
- **Récence concentrée pour la projection individuelle :**
  - La projection du talent actuel d'un joueur individuel ($r_0$) privilégie fortement ses saisons récentes (1 à 3 dernières années).
  - Les saisons de plus de 4 ans sont atténuées de manière exponentielle, de sorte qu'une saison de recrue d'il y a 6 ans n'entrave pas la projection d'une vedette de 28 ans aujourd'hui.

## 3. Jeunes joueurs, recrues et parcours pré-LNH
- **Intégration pré-LNH intelligente :** Les statistiques récoltées dans les ligues juniors (OHL, WHL, QMJHL), collégiales (NCAA) et européennes (KHL, SHL, Liiga) fournissent un pedigree offensif traduit en équivalent LNH.
- **Aucune dilution d'un jeune qui performe déjà :** L'évidence pré-LNH sert de plancher pour les recrues à très faible échantillon LNH. Dès qu'un joueur fait ses preuves dans la LNH, ses statistiques juniors s'estompent pour ne jamais pénaliser sa production réelle.
- **Protection contre le temps de jeu abrité (sheltered TOI) :** Un jeune de 19-21 ans qui joue 14-16 minutes n'est pas régressé au statut d'un vétéran de 4e trio ; son prior bayésien prend en compte son pedigree d'espoir et son efficacité offensive démontrée.
- **Neutralisation de la fausse fragilité :** Les saisons partielles de début de carrière (rappels de fin de saison, courts séjours) ne sont pas interprétées comme des blessures. La durabilité des jeunes en santé est protégée à 95-96%.
- **Courbes d'ascension vers le Peak :** Modélise la progression rapide entre 18 et 25 ans (hausse d'utilisation, intégration du jeu de puissance, maturité physique) pour projeter un sommet de carrière réaliste.

## 4. Moyenne de buts marqués, normalisation d'ère et balisage d'équipe
- **Normalisation par l'ère offensive de la ligue :** La moyenne de buts marqués par match d'équipe a varié de 2.72 b/m (2020-21) à plus de 3.15 b/m aujourd'hui (+15 % d'offensive globale). Les saisons historiques sont étalonnées selon le ratio de buts de l'ère moderne pour ne pas sous-estimer la production des vétérans et joueurs établis.
- **Réservoir offensif du club (Team Goal Reservoir) :** Un joueur individuel ne peut mathématiquement pas produire plus de points que son équipe ne marque de buts. Le modèle calcule le rythme de buts projeté de l'équipe sur 82 matchs ($GPG \times 82$) et établit un plafond d'implication maximale réaliste (IPP maximal historique : ~52 % des buts pour un attaquant d'élite, ~40 % en points de pool pour un défenseur offensif dont les buts valent 2 pts).
- **Progression offensive d'équipe :** Les équipes en forte progression offensive (+0.5 à +1.0 b/m, ex: Anaheim, Edmonton) créent un « vent de dos » qui élargit le potentiel de leurs joueurs. Les équipes en panne d'attaque (ex: Chicago, Philadelphie) balisent le rythme immédiat, rappelant que l'explosion vers le Peak nécessitera une progression collective.
- **Progression individuelle du taux de buts :** Suit la cadence de buts par match ($B/GP$) et sa progression annuelle ($g_t - g_{t-1}$). Primordial pour les défenseurs dont les buts valent double dans cette ligue (2 pts/but).

## 5. Rareté de la position (VORP)
- On calcule le **niveau de remplacement** par position : le joueur de rang `équipes × (partants + banc)` (ex. 10 équipes × (12 A + 2 banc) = 140ᵉ attaquant).
- **VORP = points projetés − niveau de remplacement** (plancher 0). C'est la mesure correcte pour comparer un D, un A et un G : on paie pour ce qu'on ne peut pas trouver facilement.
- Seulement **2 gardiens** partent par équipe (20 au total pour 10 équipes) mais les écarts entre bons et mauvais sont énormes : le VORP le reflète automatiquement.
- Onglet « Rareté » : courbes de chute par position, écart élite – remplacement, écart-type des partants.

## 6. Opportunités (le « pourquoi » derrière les points)
- **TOI/match** et **TOI en avantage numérique** (percentiles par position) : le facteur le plus prédictif des points futurs.
- **Rôle dans l'équipe** : écart de TOI vs. les coéquipiers de même position (C, ailier, D).
- **Ouverture future** : âge moyen des coéquipiers *devant lui* au même poste. Des vétérans devant = marche à franchir bientôt.
- **Tendance du TOI** (saison actuelle vs précédente) : détecte les hausses de rôle avant les points.
- **Production par 60 min vs. TOI** : écart positif = sous-utilisé (candidat à une promotion).
- **Gardiens** : part de départs de l'équipe et taux de victoires.

## 7. Longévité et fleur de l'âge (24 à 31 ans) dans un contexte de pool keeper
- **Plateau doré de la fleur de l'âge (24 à 30 ans) :** Les joueurs d'élite maintiennent leur production à un niveau stable et élevé (sommet étendu de 25 à 30 ans pour les attaquants et défenseurs). Le déclin réel ne s'amorce que vers 31-32 ans.
- **Protection de l'efficacité démontrée :** Les joueurs établis ayant prouvé leur calibre d'élite (ex: Makar, Matthews, Kaprizov, Hughes, Pastrnak, Bouchard) ne sont pas tirés vers le bas par un prior moyen de joueur de soutien : leur propre efficacité ancre leur projection de base.
- **Rétention réelle des joueurs d'élite :** Le risque de décrochage hors de la ligue en pleine maturité (25-30 ans) est minime (<1,5 %/an pour un régulier), préservant la Valeur Keeper (KV) sur tout l'horizon pluriannuel.
- **Disponibilité protégée à 93 % :** Des absences mineures lors d'une saison récente ne réduisent pas artificiellement l'espérance de matchs futurs d'un joueur en pleine possession de ses moyens.

## 8. Critères additionnels qui comptent
X 1. **Contrat** (échéance, risque de changer d'équipe/rôle) – non fourni par l'API officielle.
2. **Composition des trios et de l'avantage numérique** (PP1 vs PP2) – le TOI PP en est un bon proxy.
✓ 3. **Qualité de l'équipe et contexte de buts** : Prise en compte du volume offensif de l'équipe, de la progression de saison en saison, et balisage par le plafond offensif du club.
✓ 4. **Statistiques avancées NHL Edge** intégrées : Corsi CF% (possession de rondelle), PDO (détection de chance et régression à la moyenne), zone starts (déploiement offensif), volume de tirs/match, efficacité 5c5, passes primaires et récupérations de rondelle.
X 5. **Statut de blessure**, calendrier, matchs en retard.
✓ 6. **Jeunes/recrues (Statistiques pré-LNH & 2 premières saisons)** : Intégration des ligues juniors, AHL, NCAA et d'Europe traduites en équivalents LNH via facteurs de conversion et calendriers officiels. Neutralisation du biais de fragilité artificielle et de temps de jeu abrité (laissés de côté, rappels du club-école) lors des deux premières saisons dans la LNH.
7. **Structure du pool** : 12 protégés sur ~24 postes → il faut maximiser le surplus *par rapport au choix de repêchage* ; le Tier B-C représente cette frontière.

## Limites
- Les projections sont des **estimations statistiques**, pas des garanties.
- Les courbes d'âge empiriques reposent sur la fenêtre d'historique choisie (plus d'années = plus stable).
- Le nombre d'équipes de la ligue n'est pas dans la Constitution : ajustez-le dans la barre latérale.
"""


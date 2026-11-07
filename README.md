# 🏒 Pool GOOD YEAR Keeper

Moteur d'analyse prédictive, d'évaluation des joueurs et d'optimisation de transactions pour pools de hockey de la LNH (format Keeper).

## 🚀 Fonctionnalités clés

- **🏆 Classement complet & pondéré** : Évaluation multicritère des joueurs combinant valeur keeper future, rythme offensif courant (ROS), opportunité (temps de glace, PP1) et durabilité.
- **👤 Fiches joueurs détaillées** : Profils statistiques complets, projections pluriannuelles sur 5 à 10 ans via courbes de vieillissement empiriques et modèles de Gradient Boosting (LightGBM).
- **📊 Rareté par position (VORP)** : Calcul de la valeur ajoutée au-dessus du joueur de remplacement (VORP) ajusté selon les règlements spécifiques de la ligue.
- **🧢 Gestion de club & Ballottage** : Identification automatique des 12 protégés optimaux et détection des meilleures opportunités disponibles sur le marché libre.
- **🔄 Simulateur d'échanges (Trade Simulator)** : Évaluation instantanée des transactions (impact court terme vs long terme, diagnostic stratégique Win-Now vs Reconstruction).
- **🎯 Chercheur d'offres (Trade Finder)** : Générateur algorithmique d'offres optimales et équilibrées pour acquérir un joueur cible convoié.
- **📱 Détection dynamique mobile** : Interface Streamlit responsive adaptée automatiquement selon la taille de l'écran (breakpoint 768px).

---

## ⚙️ Installation & Lancement

### 1. Prérequis
- Python 3.9+
- Environnement virtuel recommandé

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
# ou manuellement :
pip install streamlit pandas numpy requests scikit-learn lightgbm matplotlib
```

### 3. Lancement de l'application
```bash
streamlit run pool_keeper_app.py
```

---

## 📜 Règlements de la ligue (Constitution v2.1)

- **Attaquants (F)** : But = 1.0 pt, Passe = 1.0 pt
- **Défenseurs (D)** : But = 2.0 pts, Passe = 1.0 pt
- **Gardiens (G)** : Victoire = 1.0 pt, Arrêt = 0.1 pt, But accordé = -1.0 pt (plancher de 0 pt par match)
- **Alignement** : 12 Attaquants + 6 Défenseurs + 2 Gardiens (Protégés : 12 joueurs)


# Modélisation ML de l'Humidité du Sol
## Données Lisimétriques en Conditions Semi-Arides



> Pipeline complet de Machine Learning pour la prédiction journalière de l'humidité du sol à partir d'un lysimètre de précision et de données météorologiques locales — station semi-aride du Maroc (33.57°N, 603 m a.s.l.), Mai 2023 – Mars 2026.

---

## Résultats Principaux

| Modèle | RMSE [Vol%] | MAE [Vol%] | R² | NSE |
|---|---|---|---|---|
| **Ridge Regression** | **1.921** | **1.055** | **0.850** | **0.850** |
| Random Forest | 2.447 | 1.279 | 0.757 | 0.757 |
| XGBoost | 2.491 | 1.491 | 0.749 | 0.749 |
| LSTM | 3.260 | 2.267 | 0.590 | 0.590 |

*Validation croisée temporelle — 5 folds walk-forward, gap = 30 jours, n = 1 018 observations journalières*

**Feature SHAP dominante :** SM rolling 3 jours (SHAP moyen = 1.91 Vol%) — la mémoire récente du sol prime sur les forçages climatiques immédiats.

---

## Structure du Projet

```
.
├── mois1_EDA.py                   # Mois 1 : chargement, nettoyage, EDA (388 lignes)
├── mois2_feature_engineering.py   # Mois 2 : ET₀ FAO-56, lags, rolling, saisonnalité (454 lignes)
├── mois3_models_ML.py             # Mois 3 : Ridge, RF, XGBoost, LSTM + CV temporelle (665 lignes)
├── mois4_shap_incertitude.py      # Mois 4 : SHAP, décomposition saisonnière, incertitude (580 lignes)
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
│
├── data/                          # Données brutes (non incluses — voir section Données)
│   ├── -data-2023-14-05_to_10-06.csv
│   ├── -data-2023-10-06_to_10-09.csv
│   ├── ... (12 fichiers lysimètre)
│   └── Weather_data-_14-05-2023_to_14-05-2026.csv
│
└── outputs/                       # Figures et datasets générés (créés automatiquement)
    ├── dataset_15min_clean.csv    # Après Mois 1
    ├── dataset_daily_clean.csv    # Après Mois 1
    ├── dataset_ML_features.csv    # Après Mois 2 (85 features)
    ├── fig1_series_temporelles.png
    ├── fig2_correlation.png
    ├── ... (30+ figures)
    └── pub_fig1_overview.png      # Figures publication-ready 300 dpi
```

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/OUMAIMAKABIRI/SuLaMo
cd ml-soil-moisture-semiarid
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

Les scripts doivent être exécutés **dans l'ordre**. Chaque script recharge les données depuis les fichiers CSV bruts — aucun fichier intermédiaire n'est requis entre les scripts.

### Placer les données
Copiez tous les fichiers `-data-*.csv` dans le **même dossier** que les scripts Python (ou dans `data/` et adaptez le chemin `glob` en début de chaque script).

### Mois 1 — Exploration et nettoyage
```bash
python mois1_EDA.py
```
**Durée :** ~2 min  
**Sorties :** `dataset_15min_clean.csv`, `dataset_daily_clean.csv`, `fig1` à `fig6`

Ce script :
- Charge et fusionne les 12 fichiers CSV lysimètre
- Détecte la fréquence duale (~2 min et ~13 min) et rééchantillonne sur grille 15 min
- Nettoie les anomalies (poids négatifs, capteur UMP02 défaillant)
- Agrège en journalier et produit 6 figures d'analyse exploratoire

### Mois 2 — Ingénierie des features
```bash
python mois2_feature_engineering.py
```
**Durée :** ~3 min  
**Sorties :** `dataset_ML_features.csv` (85 features), `m2_fig1` à `m2_fig4`

Ce script :
- Calcule l'ET₀ Penman-Monteith FAO-56 à l'échelle horaire (équation 53, Allen et al. 1998)
- Crée 42 lag features (lags 1, 2, 3, 5, 7, 14 jours sur 7 variables)
- Crée 20 rolling features (fenêtres 3, 7, 14, 30 jours)
- Encode la saisonnalité en sin/cos (continuité jan/déc préservée)
- Calcule les pluies cumulées (3, 7, 14, 30 jours) et le bilan hydrique

### Mois 3 — Modèles ML et validation croisée
```bash
python mois3_models_ML.py
```
**Durée :** ~15–30 min (LSTM ~10 min supplémentaires)  
**Sorties :** `m3_fig1` à `m3_fig5`, `m3_results_summary.csv`

Ce script :
- Entraîne 4 modèles (Ridge, Random Forest, XGBoost, LSTM)
- Validation croisée temporelle stricte : 5 folds walk-forward, gap = 30 jours
- Calcule RMSE, MAE, R², NSE (Nash-Sutcliffe), PBIAS pour chaque modèle
- Produit les graphiques de comparaison et d'importance des variables

### Mois 4 — SHAP, saisonnalité, incertitude
```bash
python mois4_shap_incertitude.py
```
**Durée :** ~20 min (SHAP sur 500 arbres × 1018 obs + bootstrap 50 réplicas)  
**Sorties :** `m4_fig1` à `m4_fig7`

Ce script :
- Calcule les valeurs SHAP exactes (TreeExplainer) pour RF et XGBoost
- Produit 3 types de visualisation SHAP (bar, beeswarm, dependence plots)
- Décompose les erreurs par saison, température et intensité de pluie
- Quantifie l'incertitude via les arbres RF individuels et le bootstrap
- Produit le diagramme de fiabilité (reliability diagram) et le QQ-plot des résidus

---

## Description des Données

### Lysimètre (données principales)
- **Source :** Station lysimétrique de précision (METER Group), Maroc
- **Période :** Mai 2023 – Mars 2026 (~197 000 enregistrements bruts)
- **Fréquence :** ~15 min (après harmonisation depuis protocole dual 2+13 min)
- **Capteurs clés :**
  - `UMP 01 water content [Vol%]` — **variable cible principale** (10 cm)
  - `UMP 03 water content [Vol%]` — variable cible secondaire (30 cm)
  - `FRT 01–03 tension [kPa]` — tensiomètres matriciels
  - `FRT E-01 à E-12` — réseau tensiomètre étendu
  - `air temperature [degC]`, `air humidity [%]`, `rain sum [mm]` — météo embarquée
  - `global radiation [W/m2]`, `wind speed [m/s]`, `air pressure [hPa]`

> **Note :** UMP 02 est exclu de toutes les analyses (capteur défaillant — lecture constante = 0 Vol%).

### Météo externe
- `Weather_data-_14-05-2023_to_14-05-2026.csv` — station météo externe, séparateur `;`
- Variables identiques à la météo embarquée (utilisée comme validation croisée)

---

## Paramètres du Site

| Paramètre | Valeur |
|---|---|
| Latitude | 33.57°N |
| Altitude | 603 m a.s.l. |
| Climat | Semi-aride méditerranéen (Köppen BSk) |
| ET₀ annuelle | ~2 044 mm/an |
| Précipitations annuelles | ~250–350 mm/an |
| Déficit hydrique | ~1 700 mm/an |

---

## Méthodes

### ET₀ Penman-Monteith FAO-56 (Allen et al., 1998)
Équation horaire (FAO-56, Eq. 53) :

```
ET₀ = [0.408·Δ·(Rn−G) + γ·(Cₙ/(T+273))·u₂·vpd] / [Δ + γ·(1+Cᵈ·u₂)]
```

Paramètres horaires : Cₙ = 37, Cᵈ = 0.24. Flux de chaleur du sol G = 0.1·Rn (jour) / 0.5·Rn (nuit).

### Validation Croisée Temporelle
Schéma walk-forward strict pour éviter le data leakage :
```
Fold 1 : [==TRAIN (328j)==] [gap 30j] [=TEST (164j)=]
Fold 2 : [====TRAIN (492j)====] [gap 30j] [=TEST (164j)=]
Fold 3 : [======TRAIN (656j)======] [gap 30j] [=TEST (164j)=]
Fold 4 : [========TRAIN (820j)========] [gap 30j] [=TEST (164j)=]
Fold 5 : [==========TRAIN (984j)==========] [gap] [TEST]
```

### Métriques d'évaluation
- **RMSE** [Vol%] : Root Mean Square Error
- **MAE** [Vol%] : Mean Absolute Error
- **R²** : coefficient de détermination
- **NSE** [-] : Nash-Sutcliffe Efficiency (NSE > 0.60 = satisfaisant, Moriasi et al. 2007)
- **PBIAS** [%] : Percent Bias (|PBIAS| < 10% = satisfaisant)

---

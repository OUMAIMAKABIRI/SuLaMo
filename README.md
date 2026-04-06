#  Soil Moisture Prediction using Lysimeter Data (ML Pipeline)

## Projet

Prédiction du **contenu en eau du sol (UMP Water Content [%]) et du poids des lysimètres** dans une zone semi-aride au Maroc à l’aide de **Machine Learning** (Ridge, Random Forest, XGBoost, LSTM).

Données : 12 lysimètres, 3 ans (~mai 2023 → mars 2026), capteurs FRT, mesures météo haute résolution.

---

---

## Pipeline ML

1. **Pré-traitement et nettoyage**

   * Fusion des fichiers lysimètres
   * Rééchantillonnage 15 min, gestion NaN (~1.2%)

2. **Feature Engineering**

   * Lags : 1,3,5,7,14 jours
   * Rolling windows : 3,7,14,30 jours
   * ET₀ Penman-Monteith, VPD, température, pluie cumulée
   * Encodage saisonnalité

3. **Modélisation**

   * Modèles : Ridge, Random Forest, XGBoost, LSTM
   * Validation : CV temporelle 5 folds, gap 30 jours
   * Métriques : RMSE, MAE, R², NSE

4. **Analyse explicative**

   * Importance features (RF/XGB)
   * SHAP pour Ridge et modèles complexes
   * Décomposition saisonnière des erreurs

---

## Résultats principaux

| Modèle           | RMSE | MAE  | R²   | NSE  |
| ---------------- | ---- | ---- | ---- | ---- |
| Ridge (baseline) | 1.87 | 1.09 | 0.86 | 0.86 |
| Random Forest    | 2.48 | 1.39 | 0.76 | 0.76 |
| XGBoost          | 2.53 | 1.59 | 0.74 | 0.74 |
| LSTM             | 3.26 | 2.27 | 0.59 | 0.59 |

✅ Observations : Ridge surpasse les modèles complexes grâce aux lags qui capturent déjà la dynamique non-linéaire du sol.
Variables les plus importantes : tension FRT-1, température du sol, VPD cumulé, SM lag-3j, pluie cumulée 7j.

---

## Installation & Exécution

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap tensorflow
```

Exemple :

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_models.py
python src/shap_analysis.py
```

---

## Projet 

 Projet PFE / Recherche appliquée en hydrologie semi-aride et ML

---



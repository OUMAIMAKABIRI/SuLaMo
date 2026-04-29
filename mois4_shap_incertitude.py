"""
════════════════════════════════════════════════════════════════════════════════
MOIS 4 — Validation, Explicabilité SHAP & Quantification de l'Incertitude
════════════════════════════════════════════════════════════════════════════════

Ce script doit être exécuté APRÈS mois3_models_ML.py.
Il reprend le même pipeline de données (Mois 1 + 2) et ré-entraîne les modèles
finaux sur l'ensemble des données pour l'analyse SHAP.

Entrée  : fichiers -data-*.csv (même dossier)
Sorties : m4_fig1_shap_global.png     — importance SHAP globale RF + XGB
          m4_fig2_shap_beeswarm.png   — beeswarm XGBoost top 15
          m4_fig3_shap_dependence.png — dependence plots 4 features
          m4_fig4_seasonal_errors.png — décomposition saisonnière des erreurs
          m4_fig5_shap_seasonal.png   — contributions SHAP par saison
          m4_fig6_uncertainty.png     — intervalles de confiance RF + bootstrap
          m4_fig7_calibration.png     — reliability diagram + QQ-plot

Dépendances : pip install shap scipy
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap

from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. PARAMÈTRES DU SITE
# ══════════════════════════════════════════════════════════════════════════════

LAT_DEG  = 33.57
ALTITUDE = 603

# ══════════════════════════════════════════════════════════════════════════════
# 1. RECONSTRUCTION DU DATASET ML
#    Pipeline identique à Mois 1 + Mois 2 — nécessaire pour cohérence
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MOIS 4 — Chargement et reconstruction du dataset")
print("=" * 60)

files  = sorted(glob.glob('-data-*.csv'))
dfs    = [pd.read_csv(f, sep=';', parse_dates=['Time'], dayfirst=True) for f in files]
df_raw = (pd.concat(dfs, ignore_index=True)
            .sort_values('Time').drop_duplicates(subset='Time').reset_index(drop=True))
df15   = df_raw.set_index('Time').resample('15min').mean(numeric_only=True)
df15.loc[df15['lysimeter weight [g]'] < 0, 'lysimeter weight [g]'] = np.nan
df_h   = df15.resample('1h').mean(numeric_only=True)

# ── ET₀ Penman-Monteith FAO-56 (horaire) ─────────────────────────────────────
lat   = np.radians(LAT_DEG)
T     = df_h['air temperature [degC]']
HR    = df_h['air humidity [%]'].clip(1, 100)
Rs    = df_h['global radiation  [W/m2]'].clip(lower=0)
u2    = df_h['wind speed [m/s]'].clip(lower=0.5)
P     = df_h['air pressure [hPa]']
gamma = 0.665e-3 * P / 1000
es    = 0.6108 * np.exp(17.27 * T / (T + 237.3))
ea    = es * HR / 100; vpd = (es - ea).clip(lower=0)
delta = 4098 * es / (T + 237.3) ** 2
Rs_MJ = Rs * 0.0036
doy   = pd.Series(df_h.index.day_of_year.astype(float), index=df_h.index)
dr    = 1 + 0.033 * np.cos(2*np.pi*doy/365)
sdec  = 0.409 * np.sin(2*np.pi*doy/365 - 1.39)
hf    = pd.Series(df_h.index.hour + df_h.index.minute/60., index=df_h.index)
ha    = (hf - 12) * np.pi / 12
cos_t = (np.sin(lat)*np.sin(sdec) + np.cos(lat)*np.cos(sdec)*np.cos(ha)).clip(lower=0)
Ra    = (12/np.pi) * 4.92 * dr * cos_t
Rns   = 0.77 * Rs_MJ
sig   = 4.903e-9 / 24; Tk = T + 273.16
Rs0   = (0.75 + 2e-5*ALTITUDE) * Ra
fcd   = (1.35*(Rs_MJ/Rs0.clip(lower=0.01)).clip(0.05,1.0)-0.35).clip(0.05,1.0)
Rnl   = sig * Tk**4 * (0.34 - 0.14*np.sqrt(ea.clip(lower=0.001))) * fcd
Rn    = (Rns - Rnl).clip(lower=-0.5)
is_day = pd.Series((df_h.index.hour>=6)&(df_h.index.hour<20), index=df_h.index)
G     = np.where(is_day, 0.1*Rn, 0.5*Rn)
Cn, Cd = 37, 0.24
df_h['ET0'] = ((0.408*delta*(Rn-G) + gamma*(Cn/(T+273))*u2*vpd)
               / (delta + gamma*(1+Cd*u2))).clip(lower=0)

# ── Agrégation journalière ────────────────────────────────────────────────────
day = df15.resample('1D').agg({
    'UMP 01 water content [Vol%]': ['mean', 'std'],
    'UMP 03 water content [Vol%]': ['mean'],
    'lysimeter weight [g]':        ['mean'],
    'air temperature [degC]':      ['mean', 'min', 'max'],
    'air humidity [%]':            ['mean', 'min'],
    'global radiation  [W/m2]':   ['mean'],
    'rain sum [mm]':               ['sum'],
    'wind speed [m/s]':            ['mean'],
    'air pressure [hPa]':          ['mean'],
    'FRT 01 tension [kPa]':        ['mean'],
    'FRT 02 tension [kPa]':        ['mean'],
    'FRT 03 tension [kPa]':        ['mean'],
    'UMP 01 EC [mS/cm]':           ['mean'],
    'UMP 01 temperature [degC]':   ['mean'],
    'water discharge [ml]':        ['sum'],
})
day.columns = ['_'.join(c) for c in day.columns]
day = day.rename(columns={
    'UMP 01 water content [Vol%]_mean': 'SM1',
    'UMP 01 water content [Vol%]_std':  'SM1_std',
    'UMP 03 water content [Vol%]_mean': 'SM3',
    'lysimeter weight [g]_mean':        'Lys_kg',
    'air temperature [degC]_mean':      'T_mean',
    'air temperature [degC]_min':       'T_min',
    'air temperature [degC]_max':       'T_max',
    'air humidity [%]_mean':            'RH',
    'air humidity [%]_min':             'RH_min',
    'global radiation  [W/m2]_mean':   'Rs',
    'rain sum [mm]_sum':                'Rain',
    'wind speed [m/s]_mean':           'Wind',
    'air pressure [hPa]_mean':         'P',
    'FRT 01 tension [kPa]_mean':       'FRT1',
    'FRT 02 tension [kPa]_mean':       'FRT2',
    'FRT 03 tension [kPa]_mean':       'FRT3',
    'UMP 01 EC [mS/cm]_mean':          'EC1',
    'UMP 01 temperature [degC]_mean':  'Ts1',
    'water discharge [ml]_sum':        'Qdis',
})
day['Lys_kg'] /= 1000
day['ET0'] = df_h['ET0'].resample('1D').sum()

# Variables dérivées
day['DT']  = day['T_max'] - day['T_min']
day['VPD'] = 0.6108 * np.exp(17.27*day['T_mean']/(day['T_mean']+237.3)) * (1-day['RH']/100)
day['WB']  = day['Rain'] - day['ET0']
day['rain_flag'] = (day['Rain'] > 1).astype(int)

# Saisonnalité
doy_s = day.index.day_of_year.astype(float)
day['doy_sin']   = np.sin(2*np.pi*doy_s/365)
day['doy_cos']   = np.cos(2*np.pi*doy_s/365)
day['month_sin'] = np.sin(2*np.pi*day.index.month.astype(float)/12)
day['month_cos'] = np.cos(2*np.pi*day.index.month.astype(float)/12)
m_ = day.index.month
day['season']       = np.where(m_.isin([12,1,2]), 0,
                      np.where(m_.isin([3,4,5]),  1,
                      np.where(m_.isin([6,7,8]),  2, 3)))
day['season_label'] = np.where(m_.isin([12,1,2]), 'Hiver',
                      np.where(m_.isin([3,4,5]),  'Printemps',
                      np.where(m_.isin([6,7,8]),  'Ete', 'Automne')))

# Lags
for col in ['SM1', 'FRT1', 'ET0', 'Rain', 'T_mean', 'VPD', 'WB']:
    for lag in [1, 2, 3, 5, 7, 14]:
        day[f'{col}_L{lag}'] = day[col].shift(lag)

# Rolling
for col in ['SM1', 'ET0', 'Rain', 'T_mean', 'VPD']:
    for win in [3, 7, 14]:
        day[f'{col}_R{win}'] = day[col].rolling(win, min_periods=int(win*0.6)).mean()

# Cumulées
day['Rain_C7']  = day['Rain'].rolling(7,  min_periods=3).sum()
day['Rain_C14'] = day['Rain'].rolling(14, min_periods=5).sum()
day['ET0_C7']   = day['ET0'].rolling(7,  min_periods=3).sum()
no_rain = (day['Rain'] < 1).astype(int)
day['dry_days'] = no_rain.groupby((no_rain != no_rain.shift()).cumsum()).cumsum()

day_ml = day.dropna(subset=['SM1', 'SM1_L14']).copy()
print(f"Dataset ML : {day_ml.shape}")

# ── Sélection des features ────────────────────────────────────────────────────
TARGET = 'SM1'
EXCL   = ['SM1_std', 'Lys_kg', 'SM1_L1', 'SM1_L2', 'season_label']
features = [c for c in day_ml.columns
            if c not in EXCL + [TARGET]
            and day_ml[c].dtype in ['float64', 'int64', 'int32', float, int]
            and day_ml[c].notna().mean() > 0.8]

imp = SimpleImputer(strategy='median')
X   = imp.fit_transform(day_ml[features])
y   = day_ml[TARGET].values
dates = day_ml.index

# ── Fold de test pour figures temporelles ──────────────────────────────────────
n = len(X); gap = 30; fold_size = (n - gap) // 6
fold_tr = np.arange(0, int(fold_size * 3))
fold_te = np.arange(int(fold_size * 3) + gap, int(fold_size * 4))
yobs    = y[fold_te]; d_te = dates[fold_te]

print(f"Features : {len(features)} | Train : {len(fold_tr)} | Test : {len(fold_te)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ENTRAÎNEMENT DES MODÈLES FINAUX (sur tout le dataset)
#    Note : pour SHAP, on entraîne sur tout pour avoir plus de données.
#    Pour les résidus, on utilise le fold de test (honnête).
# ══════════════════════════════════════════════════════════════════════════════

print("\nEntraînement des modèles finaux...")

# Ridge final (tout le dataset)
pipe_r_full = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))])
pipe_r_full.fit(X, y)

# Random Forest final
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                            max_features=0.5, n_jobs=-1, random_state=42)
rf.fit(X, y)

# XGBoost final
xp = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
          reg_alpha=0.1, reg_lambda=1.0, random_state=42,
          n_jobs=-1, tree_method='hist', verbosity=0)
xm = xgb.XGBRegressor(**xp)
xm.fit(X, y, verbose=False)

# Modèles entraînés sur fold_tr pour prédire fold_te (évaluation honnête)
pipe_r_fold = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))])
pipe_r_fold.fit(X[fold_tr], y[fold_tr])
yp_r = pipe_r_fold.predict(X[fold_te])

rf_fold = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                 max_features=0.5, n_jobs=-1, random_state=42)
rf_fold.fit(X[fold_tr], y[fold_tr])
yp_rf = rf_fold.predict(X[fold_te])

xm_fold = xgb.XGBRegressor(**xp)
xm_fold.fit(X[fold_tr], y[fold_tr], verbose=False)
yp_xgb = xm_fold.predict(X[fold_te])

print("Modèles entraînés ✓")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSE SHAP
#    TreeExplainer : algorithme exact pour modèles arborescents.
#    Complexité O(T × L × D²) où T = n_arbres, L = feuilles, D = profondeur.
#    Les valeurs SHAP mesurent la contribution marginale de chaque feature
#    à la prédiction individuelle (par rapport à la prédiction moyenne).
# ══════════════════════════════════════════════════════════════════════════════

print("\nCalcul SHAP Random Forest...")
explainer_rf    = shap.TreeExplainer(rf)
shap_values_rf  = explainer_rf.shap_values(X)   # shape (n_obs, n_features)
shap_imp_rf     = pd.Series(np.abs(shap_values_rf).mean(axis=0),
                             index=features).sort_values(ascending=False)

print("Calcul SHAP XGBoost...")
explainer_xgb   = shap.TreeExplainer(xm)
shap_values_xgb = explainer_xgb.shap_values(X)
shap_imp_xgb    = pd.Series(np.abs(shap_values_xgb).mean(axis=0),
                             index=features).sort_values(ascending=False)

print("Top 10 SHAP XGBoost :")
print(shap_imp_xgb.head(10).round(4).to_string())

# Dictionnaire de labels lisibles pour les figures
label_map = {
    'SM1_R3': 'SM rolling 3d',   'SM1_R7': 'SM rolling 7d',
    'SM1_R14': 'SM rolling 14d', 'EC1': 'EC sol UMP01',
    'SM1_L3': 'SM lag 3d',       'SM1_L5': 'SM lag 5d',
    'SM1_L7': 'SM lag 7d',       'SM1_L14': 'SM lag 14d',
    'FRT1': 'FRT-01 tension',    'FRT2': 'FRT-02 tension',
    'FRT3': 'FRT-03 tension',    'T_mean': 'Air temperature',
    'VPD': 'VPD [kPa]',          'ET0': 'ET0 [mm/d]',
    'Rain': 'Precipitation',     'Rain_C7': 'Cumul. rain 7d',
    'Rain_C14': 'Cumul. rain 14d', 'Rain_R14': 'Rolling rain 14d',
    'ET0_C7': 'Cumul. ET0 7d',   'month_sin': 'Month (sin)',
    'doy_cos': 'DOY (cos)',       'doy_sin': 'DOY (sin)',
    'RH': 'Rel. humidity',       'Rs': 'Solar radiation',
    'dry_days': 'Dry days',      'WB': 'Water balance',
    'Ts1': 'Soil temperature',   'SM3': 'SM3 (30cm)',
}
def nice(f): return label_map.get(f, f)

# ── FIGURE 1 : Importance globale SHAP (bar chart RF + XGB) ──────────────────
print("\nFigure 1 — SHAP global...")
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Analyse SHAP — Importance globale des features\n(Mean |SHAP value| [Vol%])',
             fontsize=13, fontweight='bold')

# Groupes de features et couleurs associées
group_colors = {
    'mem':     '#1a6faf',   # mémoire sol (SM lags/rolling)
    'frt':     '#8c564b',   # tensiomètres FRT
    'water':   '#17becf',   # eau (pluie, ET₀, bilan)
    'climate': '#d62728',   # climat (T°, VPD, HR, Rs)
    'season':  '#9467bd',   # saisonnalité
}

def get_group_color(fname):
    if 'SM1_R' in fname or 'SM1_L' in fname: return group_colors['mem']
    if 'FRT' in fname:                        return group_colors['frt']
    if any(x in fname for x in ['Rain', 'ET0', 'WB', 'Qdis']): return group_colors['water']
    if any(x in fname for x in ['T_mean','T_min','T_max','VPD','RH','Rs','Wind']):
                                               return group_colors['climate']
    return group_colors['season']

for ax, imp, title in zip(axes,
                           [shap_imp_rf.head(20), shap_imp_xgb.head(20)],
                           ['Random Forest', 'XGBoost']):
    ypos   = range(len(imp) - 1, -1, -1)
    colors = [get_group_color(f) for f in imp.index]
    bars   = ax.barh(list(ypos), imp.values, color=colors, alpha=0.82,
                     edgecolor='none', height=0.7)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([nice(f) for f in imp.index], fontsize=9.5)
    ax.set_xlabel('Mean |SHAP value| [Vol%]', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i, v in enumerate(imp.values):
        ax.text(v + 0.005, len(imp)-1-i, f'{v:.3f}', va='center', fontsize=7.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_els = [
    Patch(color='#1a6faf', label='Mémoire sol (SM lags/rolling)'),
    Patch(color='#8c564b', label='Tensiomètres FRT'),
    Patch(color='#17becf', label='Eau (pluie, ET₀, bilan)'),
    Patch(color='#d62728', label='Climat (T°, VPD, HR, Rs)'),
    Patch(color='#9467bd', label='Saisonnalité (DOY, mois)'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('m4_fig1_shap_global.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig1_shap_global.png")

# ── FIGURE 2 : SHAP Beeswarm (XGBoost, top 15) ───────────────────────────────
print("Figure 2 — SHAP beeswarm...")
top15       = shap_imp_xgb.head(15).index.tolist()
top15_idx   = [features.index(f) for f in top15]
sv_top      = shap_values_xgb[:, top15_idx]
X_top       = X[:, top15_idx]

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')
np.random.seed(42)

for row_i, feat in enumerate(top15):
    y_pos = len(top15) - 1 - row_i
    sv    = sv_top[:, row_i]
    xv    = X_top[:, row_i]
    # Normalisation 0-1 pour la couleur (percentiles 5–95 pour éviter les outliers)
    p5, p95   = np.nanpercentile(xv, 5), np.nanpercentile(xv, 95)
    xv_norm   = np.clip((xv - p5) / (p95 - p5 + 1e-8), 0, 1)
    jitter    = np.random.uniform(-0.36, 0.36, len(sv))
    ax.scatter(sv, y_pos + jitter, c=xv_norm, cmap='RdBu_r',
               s=6, alpha=0.5, linewidths=0, vmin=0, vmax=1)

ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([nice(f) for f in reversed(top15)], fontsize=10)
ax.set_xlabel('SHAP value (impact sur SM₁) [Vol%]', fontsize=11)
ax.set_title('SHAP Beeswarm — XGBoost (top 15 features)\n'
             'Rouge = valeur haute de la feature | Bleu = valeur basse',
             fontsize=12, fontweight='bold')
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r'), ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Valeur de la feature (normalisée)', fontsize=9)
cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(['Basse', 'Moy.', 'Haute'])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.25)
plt.tight_layout()
plt.savefig('m4_fig2_shap_beeswarm.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig2_shap_beeswarm.png")

# ── FIGURE 3 : SHAP Dependence Plots (4 features clés) ───────────────────────
# Un dependence plot montre SHAP(feature) en fonction de la valeur de la feature,
# coloré par une variable d'interaction — révèle les interactions non-linéaires.
print("Figure 3 — SHAP dependence plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')
fig.suptitle('SHAP Dependence Plots — XGBoost\n'
             'Impact de chaque feature sur la prédiction SM₁', fontsize=13, fontweight='bold')

dep_feats = [
    ('SM1_R3',   'SM rolling 3d [Vol%]',        'FRT1',   'FRT-01 [kPa]'),
    ('FRT2',     'FRT-02 tension [kPa]',         'T_mean', 'T° air [°C]'),
    ('ET0_C7',   'ET₀ cumulée 7j [mm]',          'VPD',    'VPD [kPa]'),
    ('Rain_C7',  'Pluie cumulée 7j [mm]',        'SM1_L3', 'SM lag-3j [Vol%]'),
]

for ax, (feat, feat_lbl, interact, inter_lbl) in zip(axes.flat, dep_feats):
    if feat not in features or interact not in features:
        continue
    fi    = features.index(feat)
    ii    = features.index(interact)
    x_f   = X[:, fi]; sv_f = shap_values_xgb[:, fi]
    x_int = X[:, ii]

    # Clip extremes pour éviter les points aberrants
    p5, p95     = np.nanpercentile(x_f, 5), np.nanpercentile(x_f, 95)
    mask        = (x_f >= p5) & (x_f <= p95)
    vmin, vmax  = np.nanpercentile(x_int, 5), np.nanpercentile(x_int, 95)
    x_int_norm  = np.clip((x_int - vmin) / (vmax - vmin + 1e-8), 0, 1)

    sc = ax.scatter(x_f[mask], sv_f[mask], c=x_int_norm[mask],
                    cmap='coolwarm', s=8, alpha=0.5, linewidths=0)
    # Tendance polynomiale de degré 2
    z  = np.polyfit(x_f[mask], sv_f[mask], 2)
    xs = np.linspace(x_f[mask].min(), x_f[mask].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), 'k-', lw=1.8, alpha=0.8, label='Tendance poly-2')
    ax.axhline(0, color='gray', lw=0.8, ls='--')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(inter_lbl, fontsize=7.5)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Bas', 'Haut'])

    ax.set_xlabel(feat_lbl, fontsize=9.5)
    ax.set_ylabel(f'SHAP({nice(feat)}) [Vol%]', fontsize=9.5)
    ax.set_title(nice(feat), fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m4_fig3_shap_dependence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig3_shap_dependence.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DÉCOMPOSITION SAISONNIÈRE DES ERREURS
#    On prédit sur tout le dataset (mode entraînement) pour avoir une vision
#    globale par saison. Les métriques de généralisation restent celles du Mois 3.
# ══════════════════════════════════════════════════════════════════════════════

print("\nFigure 4 — Décomposition saisonnière...")
yp_rf_all  = rf.predict(X)
yp_xgb_all = xm.predict(X)

df_err = pd.DataFrame({
    'obs':          y,
    'pred_rf':      yp_rf_all,
    'pred_xgb':     yp_xgb_all,
    'err_rf':       yp_rf_all  - y,
    'err_xgb':      yp_xgb_all - y,
    'abs_err_rf':   np.abs(yp_rf_all  - y),
    'abs_err_xgb':  np.abs(yp_xgb_all - y),
    'season':       day_ml['season_label'].values,
    'T_mean':       day_ml['T_mean'].values,
    'Rain':         day_ml['Rain'].values,
}, index=dates)

def nse(o, s):
    """Nash-Sutcliffe Efficiency — standard en hydrologie (>0.60 = satisfaisant)."""
    return float(1 - np.sum((o-s)**2) / np.sum((o-o.mean())**2))

seasons = ['Hiver', 'Printemps', 'Ete', 'Automne']
pal     = {'Hiver': '#4c72b0', 'Printemps': '#55a868', 'Ete': '#c44e52', 'Automne': '#dd8452'}

print("NSE par saison (données entraînement) :")
for s in seasons:
    mask_s = df_err['season'] == s
    yt_    = y[mask_s.values]; yp_ = yp_rf_all[mask_s.values]
    print(f"  {s:<12} RF={nse(yt_, yp_):.3f}  XGB={nse(y[mask_s.values], yp_xgb_all[mask_s.values]):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.patch.set_facecolor('white')
fig.suptitle('Décomposition saisonnière des erreurs de prédiction',
             fontsize=13, fontweight='bold')

# (a) RMSE par saison
ax = axes[0, 0]
for k, (col, lbl) in enumerate([('abs_err_rf','RF'), ('abs_err_xgb','XGB')]):
    rmse_s = df_err.groupby('season')[col].apply(lambda x: np.sqrt((x**2).mean()))
    x_pos  = np.arange(4) + k*0.35 - 0.18
    ax.bar(x_pos, [rmse_s.get(s, 0) for s in seasons], width=0.33,
           color=['#2ca02c', '#e07b39'][k], label=lbl, alpha=0.85)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(seasons, fontsize=9)
ax.set_ylabel('RMSE [Vol%]', fontsize=10)
ax.set_title('RMSE par saison', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (b) Distribution des résidus par saison (RF)
ax = axes[0, 1]
for s in seasons:
    d_ = df_err[df_err['season'] == s]['err_rf'].dropna()
    ax.hist(d_, bins=30, alpha=0.6, color=pal[s], label=s, density=True)
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Résidu RF (Pred−Obs) [Vol%]', fontsize=10)
ax.set_ylabel('Densité', fontsize=10)
ax.set_title('Distribution des résidus RF par saison', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (c) NSE par saison
ax = axes[0, 2]
for k, (yp_all, lbl) in enumerate([(yp_rf_all,'RF'), (yp_xgb_all,'XGB')]):
    nse_s = [nse(y[df_err['season'].values==s], yp_all[df_err['season'].values==s])
             for s in seasons]
    x_pos = np.arange(4) + k*0.35 - 0.18
    ax.bar(x_pos, nse_s, width=0.33, color=['#2ca02c','#e07b39'][k], label=lbl, alpha=0.85)
ax.axhline(0.6, color='red', ls='--', lw=1, label='NSE seuil = 0.6')
ax.axhline(0, color='gray', lw=0.6)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(seasons, fontsize=9)
ax.set_ylabel('NSE', fontsize=10)
ax.set_title('NSE par saison (seuil > 0.6)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (d) Erreur absolue vs T° air
ax = axes[1, 0]
for s in seasons:
    d_ = df_err[df_err['season'] == s]
    ax.scatter(d_['T_mean'], d_['abs_err_rf'], c=pal[s], s=8, alpha=0.4, label=s)
# Tendance globale lissée
order_t  = np.argsort(df_err['T_mean'].dropna())
T_sorted = df_err['T_mean'].dropna().values[order_t]
e_sorted = df_err.loc[df_err['T_mean'].notna(), 'abs_err_rf'].values[order_t]
win      = 40
roll     = [e_sorted[max(0,i-win):i+win].mean() for i in range(len(e_sorted))]
ax.plot(T_sorted, roll, 'k-', lw=2, label='Tendance')
ax.set_xlabel('T° air [°C]', fontsize=10)
ax.set_ylabel('|Erreur RF| [Vol%]', fontsize=10)
ax.set_title('Erreur absolue vs T° air', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, markerscale=2); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (e) RMSE par catégorie de pluie
ax = axes[1, 1]
bins_rain    = [0, 0.5, 2, 5, 10, 50]
labels_rain  = ['0', '0–2', '2–5', '5–10', '>10']
df_err['rain_cat'] = pd.cut(df_err['Rain'], bins=bins_rain, labels=labels_rain)
for k, (col, lbl) in enumerate([('abs_err_rf','RF'), ('abs_err_xgb','XGB')]):
    rmse_r = df_err.groupby('rain_cat', observed=True)[col].apply(
        lambda x: np.sqrt((x**2).mean()))
    ax.bar(np.arange(5) + k*0.35 - 0.18, rmse_r.values, width=0.33,
           color=['#2ca02c','#e07b39'][k], label=lbl, alpha=0.85)
ax.set_xticks(range(5)); ax.set_xticklabels(labels_rain, fontsize=9)
ax.set_xlabel('Pluie [mm/j]', fontsize=10)
ax.set_ylabel('RMSE [Vol%]', fontsize=10)
ax.set_title('RMSE vs intensité de pluie\n(épisodes pluvieux = principale limite)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (f) RMSE glissant 30j (évolution temporelle)
ax = axes[1, 2]
df_err['sq_rf']  = df_err['err_rf']**2
df_err['sq_xgb'] = df_err['err_xgb']**2
for col, c, lbl in [('sq_rf','#2ca02c','RF'), ('sq_xgb','#e07b39','XGB')]:
    roll_rmse = np.sqrt(df_err[col].rolling(30, min_periods=15).mean())
    ax.fill_between(df_err.index, roll_rmse, alpha=0.2, color=c)
    ax.plot(df_err.index, roll_rmse, lw=1.5, color=c, label=lbl)
ax.set_ylabel('RMSE rolling 30j [Vol%]', fontsize=10)
ax.set_title('Évolution temporelle du RMSE\n(stabilité dans le temps)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m4_fig4_seasonal_errors.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig4_seasonal_errors.png")

# ── FIGURE 5 : Contributions SHAP par saison et par groupe ───────────────────
print("Figure 5 — SHAP saisonnier...")
shap_df_xgb             = pd.DataFrame(shap_values_xgb, columns=features, index=dates)
shap_df_xgb['season']   = day_ml['season_label'].values

groups = {
    'Mémoire sol\n(SM lags/roll)': [f for f in features if 'SM1_R' in f or 'SM1_L' in f],
    'Tensiomètres FRT':            [f for f in features if 'FRT' in f],
    'Eau (pluie/ET₀)':             [f for f in features if any(x in f for x in ['Rain','ET0','WB'])],
    'Climat (T°/VPD/Rs)':          [f for f in features if any(x in f for x in ['T_mean','VPD','RH','Rs'])],
}
group_colors_list = ['#1a6faf', '#8c564b', '#17becf', '#d62728']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor('white')
fig.suptitle('Évolution saisonnière des contributions SHAP — XGBoost\n'
             '(quelles features pilotent la prédiction selon la saison ?)',
             fontsize=13, fontweight='bold')

for ax, (grp_name, grp_feats, grp_col) in zip(
        axes.flat,
        [(k, v, c) for (k, v), c in zip(groups.items(), group_colors_list)]):
    grp_feats_ok = [f for f in grp_feats if f in shap_df_xgb.columns]
    if not grp_feats_ok:
        continue
    shap_df_xgb['grp_contrib'] = shap_df_xgb[grp_feats_ok].abs().sum(axis=1)
    season_contrib = shap_df_xgb.groupby('season')['grp_contrib'].mean().reindex(seasons).fillna(0)
    bars = ax.bar(seasons, season_contrib.values, color=grp_col, alpha=0.82, edgecolor='none')
    for bar, v in zip(bars, season_contrib.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(grp_name, fontsize=11, fontweight='bold', color=grp_col)
    ax.set_ylabel('Σ|SHAP| moyen [Vol%]', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, season_contrib.max() * 1.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m4_fig5_shap_seasonal.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig5_shap_seasonal.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. QUANTIFICATION DE L'INCERTITUDE
#    Méthode 1 — Arbres RF individuels :
#      Chaque arbre du RF donne une prédiction → distribution empirique.
#      IC = percentiles de cette distribution.
#    Méthode 2 — Bootstrap :
#      Ré-entraîner N modèles sur sous-échantillons → distribution bootstrap.
# ══════════════════════════════════════════════════════════════════════════════

print("\nFigure 6 — Intervalles de confiance...")

# Méthode 1 : arbres RF individuels sur fold_te
rf_tree_preds = np.array([tree.predict(X[fold_te]) for tree in rf.estimators_])
rf_mean = rf_tree_preds.mean(axis=0)
rf_std  = rf_tree_preds.std(axis=0)
rf_p5   = np.percentile(rf_tree_preds, 5,  axis=0)
rf_p25  = np.percentile(rf_tree_preds, 25, axis=0)
rf_p75  = np.percentile(rf_tree_preds, 75, axis=0)
rf_p95  = np.percentile(rf_tree_preds, 95, axis=0)

cov_90  = np.mean((yobs >= rf_p5)  & (yobs <= rf_p95)) * 100
cov_50  = np.mean((yobs >= rf_p25) & (yobs <= rf_p75)) * 100
width90 = (rf_p95 - rf_p5).mean()
print(f"IC 90% RF : couverture={cov_90:.1f}%  |  Largeur moy={width90:.2f} Vol%")
print(f"IC 50% RF : couverture={cov_50:.1f}%")

# Méthode 2 : Bootstrap (n=50 réplicas)
print("Bootstrap (50 réplicas)...")
np.random.seed(42)
boot_preds = []
for b in range(50):
    idx_boot = np.random.choice(len(fold_tr), len(fold_tr), replace=True)
    rf_b = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                  max_features=0.5, n_jobs=-1, random_state=b)
    rf_b.fit(X[fold_tr[idx_boot]], y[fold_tr[idx_boot]])
    boot_preds.append(rf_b.predict(X[fold_te]))
    if (b+1) % 10 == 0:
        print(f"  Bootstrap {b+1}/50", flush=True)

boot_preds = np.array(boot_preds)
boot_mean  = boot_preds.mean(axis=0)
boot_p5    = np.percentile(boot_preds, 5,  axis=0)
boot_p95   = np.percentile(boot_preds, 95, axis=0)
cov_boot   = np.mean((yobs >= boot_p5) & (yobs <= boot_p95)) * 100
print(f"Bootstrap IC 90% : couverture={cov_boot:.1f}%")

fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
fig.patch.set_facecolor('white')
fig.suptitle('Quantification de l\'incertitude des prédictions\n'
             'IC RF (arbres individuels) + Bootstrap (50 réplicas)',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.fill_between(d_te, rf_p5, rf_p95, alpha=0.2, color='#2ca02c', label='IC 90% (arbres RF)')
ax.fill_between(d_te, rf_p25, rf_p75, alpha=0.35, color='#2ca02c', label='IC 50% (IQR arbres)')
ax.plot(d_te, rf_mean, lw=1.5, color='#2ca02c', label='RF moyen')
ax.plot(d_te, yobs,    lw=1.8, color='black',   label='Observé', zorder=5)
ax.set_ylabel('SM₁ [Vol%]', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.set_title(f'IC RF (arbres) — Couverture 90%: {cov_90:.1f}%  |  50%: {cov_50:.1f}%  |  '
             f'Largeur moy 90%: {width90:.2f} Vol%', fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.fill_between(d_te, boot_p5, boot_p95, alpha=0.25, color='#9467bd',
                label=f'IC Bootstrap 90% (n=50)')
ax.plot(d_te, boot_mean, lw=1.5, color='#9467bd', label='Bootstrap moyen')
ax.plot(d_te, yobs,      lw=1.8, color='black',   label='Observé', zorder=5)
ax.set_ylabel('SM₁ [Vol%]', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.set_title(f'IC Bootstrap — Couverture 90%: {cov_boot:.1f}%', fontsize=10)
ax.grid(alpha=0.3)

ax = axes[2]
width_t = rf_p95 - rf_p5
rain_f  = day_ml['Rain'].iloc[fold_te]
ax.fill_between(d_te, width_t, alpha=0.4, color='#ff7f0e')
ax.plot(d_te, width_t, lw=1.2, color='#ff7f0e',
        label=f'Largeur IC 90% (moy={width_t.mean():.2f} Vol%)')
ax.axhline(width_t.mean(), color='k', ls='--', lw=1)
ax2b = ax.twinx()
ax2b.bar(d_te, rain_f.values, color='#3182bd', alpha=0.35, width=1, label='Pluie')
ax2b.set_ylabel('Pluie [mm/j]', fontsize=9, color='#3182bd')
ax.set_ylabel('Largeur IC 90% [Vol%]', fontsize=10, color='#ff7f0e')
ax.legend(fontsize=9, loc='upper left')
ax2b.legend(fontsize=9, loc='upper right')
ax.set_title('Évolution temporelle de l\'incertitude\n'
             '(pics après épisodes pluvieux = zone de forte incertitude)', fontsize=10)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for ax in axes:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(h_pad=0.8)
plt.savefig('m4_fig6_uncertainty.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig6_uncertainty.png")

# ── FIGURE 7 : Calibration (reliability diagram + incertitude vs erreur + QQ) ─
print("Figure 7 — Calibration et QQ-plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Calibration des intervalles de confiance et analyse des résidus',
             fontsize=12, fontweight='bold')

# (a) Reliability diagram : compare la couverture nominale et empirique
ax = axes[0]
nominal = np.arange(10, 100, 10)
emp_cov = []
for level in nominal:
    p_lo  = np.percentile(rf_tree_preds, (100-level)/2,     axis=0)
    p_hi  = np.percentile(rf_tree_preds, 100-(100-level)/2, axis=0)
    emp_cov.append(np.mean((yobs >= p_lo) & (yobs <= p_hi)) * 100)

ax.plot([0,100], [0,100], 'k--', lw=1, label='Calibration parfaite')
ax.plot(nominal, emp_cov, 'o-', color='#2ca02c', lw=2, markersize=6, label='RF (arbres)')
ax.fill_between(nominal, emp_cov, nominal, alpha=0.1, color='#2ca02c')
ax.set_xlabel('Couverture nominale (%)', fontsize=10)
ax.set_ylabel('Couverture empirique (%)', fontsize=10)
ax.set_title('Diagramme de fiabilité\n(calibration des IC)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (b) Incertitude prédictive (std RF) vs erreur absolue réelle
ax = axes[1]
err_abs = np.abs(yobs - rf_mean)
ax.scatter(rf_std, err_abs, alpha=0.35, s=15, color='#2ca02c')
z_  = np.polyfit(rf_std, err_abs, 1)
xs_ = np.linspace(rf_std.min(), rf_std.max(), 100)
ax.plot(xs_, np.poly1d(z_)(xs_), 'r-', lw=1.5,
        label=f'r = {np.corrcoef(rf_std, err_abs)[0,1]:.3f}')
ax.set_xlabel('Std RF (incertitude prédictive) [Vol%]', fontsize=10)
ax.set_ylabel('|Erreur| observée [Vol%]', fontsize=10)
ax.set_title('Incertitude prédictive\nvs erreur réelle', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (c) QQ-plot des résidus RF (test de normalité)
ax = axes[2]
residuals = yobs - rf_mean
(osm, osr), (slope, intercept, r) = sp_stats.probplot(residuals, dist='norm')
ax.scatter(osm, osr, s=8, alpha=0.5, color='#2ca02c')
xs_qq = np.array([min(osm), max(osm)])
ax.plot(xs_qq, slope*xs_qq + intercept, 'r-', lw=1.5, label=f'R² = {r**2:.3f}')
ax.set_xlabel('Quantiles théoriques N(0,1)', fontsize=10)
ax.set_ylabel('Quantiles observés des résidus', fontsize=10)
ax.set_title('QQ-plot des résidus RF\n(normalité des erreurs)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m4_fig7_calibration.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ m4_fig7_calibration.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MOIS 4 TERMINÉ")
print("=" * 60)
print(f"Top 5 features SHAP XGBoost :")
for f, v in shap_imp_xgb.head(5).items():
    print(f"  {nice(f):<22} SHAP = {v:.4f} Vol%")

print(f"\nCalibration IC RF (arbres individuels) :")
print(f"  IC 90%  : {cov_90:.1f}%  (nominal = 90%)")
print(f"  IC 50%  : {cov_50:.1f}%  (nominal = 50%)")
print(f"  Largeur IC 90% : {width90:.2f} Vol%")
print(f"\nCalibration Bootstrap (n=50) :")
print(f"  IC 90%  : {cov_boot:.1f}%")
print("\n7 figures générées (m4_fig1 → m4_fig7)")

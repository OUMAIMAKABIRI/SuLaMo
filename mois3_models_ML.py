"""
════════════════════════════════════════════════════════════════════════════════
MOIS 3 — Développement et Évaluation des Modèles ML
  • Ridge Regression (baseline)
  • Random Forest
  • XGBoost
  • LSTM (deep learning séquentiel)
  • Validation croisée temporelle (5 folds, gap=30 jours)
  • Métriques : RMSE, MAE, R², NSE, PBIAS
════════════════════════════════════════════════════════════════════════════════

Entrée  : fichiers -data-*.csv (même dossier)
Sorties : m3_fig1_perf_table.png, m3_fig2_obs_pred.png,
          m3_fig3_timeseries.png, m3_fig4_importance.png,
          m3_fig5_error_analysis.png, m3_results_summary.csv

Dépendances : pip install scikit-learn xgboost tensorflow
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # silence TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. PARAMÈTRES DU SITE
# ══════════════════════════════════════════════════════════════════════════════

LAT_DEG  = 33.57
ALTITUDE = 603

# ══════════════════════════════════════════════════════════════════════════════
# 1. RECONSTRUCTION DU DATASET ML
#    (pipeline complet Mois 1 + Mois 2 concentré en une fonction)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset():
    """Charge, nettoie et enrichit le dataset ML (Mois 1 + Mois 2 condensé)."""
    print("Chargement et construction du dataset...", flush=True)

    # ── Chargement ─────────────────────────────────────────────────────────────
    files  = sorted(glob.glob('-data-*.csv'))
    dfs    = [pd.read_csv(f, sep=';', parse_dates=['Time'], dayfirst=True) for f in files]
    df_raw = (pd.concat(dfs, ignore_index=True)
                .sort_values('Time').drop_duplicates(subset='Time').reset_index(drop=True))
    df15   = df_raw.set_index('Time').resample('15min').mean(numeric_only=True)
    df15.loc[df15['lysimeter weight [g]'] < 0, 'lysimeter weight [g]'] = np.nan
    df_h   = df15.resample('1h').mean(numeric_only=True)

    # ── ET₀ Penman-Monteith FAO-56 (horaire) ──────────────────────────────────
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
    Rns   = 0.77 * Rs_MJ; sig = 4.903e-9/24; Tk = T + 273.16
    Rs0   = (0.75 + 2e-5*ALTITUDE) * Ra
    fcd   = (1.35*(Rs_MJ/Rs0.clip(lower=0.01)).clip(0.05,1.0) - 0.35).clip(0.05,1.0)
    Rnl   = sig * Tk**4 * (0.34 - 0.14*np.sqrt(ea.clip(lower=0.001))) * fcd
    Rn    = (Rns - Rnl).clip(lower=-0.5)
    is_day= pd.Series((df_h.index.hour>=6)&(df_h.index.hour<20), index=df_h.index)
    G     = np.where(is_day, 0.1*Rn, 0.5*Rn)
    Cn, Cd = 37, 0.24
    df_h['ET0'] = ((0.408*delta*(Rn-G) + gamma*(Cn/(T+273))*u2*vpd)
                   / (delta + gamma*(1+Cd*u2))).clip(lower=0)

    # ── Agrégation journalière ─────────────────────────────────────────────────
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
    rn = {
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
    }
    day = day.rename(columns=rn)
    day['Lys_kg'] /= 1000
    day['ET0'] = df_h['ET0'].resample('1D').sum()

    # Dérivées
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
    day['season'] = np.where(m_.isin([12,1,2]), 0,
                    np.where(m_.isin([3,4,5]),  1,
                    np.where(m_.isin([6,7,8]),  2, 3)))

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
    print(f"  Dataset ML : {day_ml.shape}")
    return day_ml


# ── Charger le dataset ────────────────────────────────────────────────────────
day_ml = build_dataset()

# ── Sélection des features ────────────────────────────────────────────────────
TARGET = 'SM1'

# On exclut les colonnes redondantes ou quasi-data-leakage
EXCL = ['SM1_std', 'Lys_kg', 'SM1_L1', 'SM1_L2']
features = [c for c in day_ml.columns
            if c not in EXCL + [TARGET]
            and day_ml[c].dtype in ['float64', 'int64', 'int32', float, int]
            and day_ml[c].notna().mean() > 0.8]

X_df = day_ml[features]
y    = day_ml[TARGET].values
dates = day_ml.index

# Imputation des NaN résiduels (médiane)
imp = SimpleImputer(strategy='median')
X   = imp.fit_transform(X_df)

print(f"Features utilisées : {len(features)}")
print(f"Observations       : {len(y)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTRIQUES (standard hydrologie)
# ══════════════════════════════════════════════════════════════════════════════

def nse(obs, sim):
    """Nash-Sutcliffe Efficiency — standard en hydrologie (>0.6 = bon)"""
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - obs.mean())**2)

def get_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'RMSE':  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'MAE':   round(float(mean_absolute_error(y_true, y_pred)), 3),
        'R2':    round(float(r2_score(y_true, y_pred)), 3),
        'NSE':   round(float(nse(y_true, y_pred)), 3),
        'PBIAS': round(float(100 * (y_pred - y_true).sum() / y_true.sum()), 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 3. FOLDS DE VALIDATION CROISÉE TEMPORELLE
#    Principe : train toujours AVANT test, gap=30j pour éviter l'autocorrélation
# ══════════════════════════════════════════════════════════════════════════════

n          = len(X)
gap        = 30          # jours entre fin train et début test
n_splits   = 5
fold_size  = (n - gap) // (n_splits + 1)

folds = []
for i in range(n_splits):
    tr_end  = int(fold_size * (i + 2))
    ts_start = tr_end + gap
    ts_end   = min(ts_start + fold_size, n)
    if ts_end > ts_start:
        folds.append((np.arange(0, tr_end), np.arange(ts_start, ts_end)))

print(f"\nCV : {len(folds)} folds, gap={gap} jours")
for i, (tr, te) in enumerate(folds):
    print(f"  Fold {i+1} : train={len(tr)} jours | test={len(te)} jours "
          f"({dates[te[0]].date()} → {dates[te[-1]].date()})")

all_results = {}

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODÈLE 1 : RIDGE REGRESSION (baseline linéaire)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1/4] Ridge Regression (baseline)...", flush=True)
preds_r, true_r, dates_r = [], [], []

for tr, te in folds:
    pipe = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))])
    pipe.fit(X[tr], y[tr])
    preds_r.extend(pipe.predict(X[te]))
    true_r.extend(y[te]); dates_r.extend(dates[te])

res_r = get_metrics(np.array(true_r), np.array(preds_r), 'Ridge (baseline)')
print(f"   {res_r}")
all_results['Ridge'] = {'cv_pred': np.array(preds_r), 'cv_true': np.array(true_r),
                        'cv_dates': dates_r, 'metrics': res_r}

# Modèle final entraîné sur tout
pipe_r_final = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))])
pipe_r_final.fit(X, y)

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODÈLE 2 : RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

print("[2/4] Random Forest...", flush=True)
preds_rf, true_rf, dates_rf = [], [], []

for tr, te in folds:
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        max_features=0.5, n_jobs=-1, random_state=42)
    rf.fit(X[tr], y[tr])
    preds_rf.extend(rf.predict(X[te]))
    true_rf.extend(y[te]); dates_rf.extend(dates[te])

res_rf = get_metrics(np.array(true_rf), np.array(preds_rf), 'Random Forest')
print(f"   {res_rf}")
all_results['RF'] = {'cv_pred': np.array(preds_rf), 'cv_true': np.array(true_rf),
                     'cv_dates': dates_rf, 'metrics': res_rf}

# Modèle final (plus d'arbres)
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=3,
    max_features=0.5, n_jobs=-1, random_state=42)
rf_final.fit(X, y)
rf_imp = pd.Series(rf_final.feature_importances_, index=features).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# 6. MODÈLE 3 : XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

print("[3/4] XGBoost...", flush=True)
preds_xgb, true_xgb, dates_xgb = [], [], []

xgb_params = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, tree_method='hist', verbosity=0)

for tr, te in folds:
    xm = xgb.XGBRegressor(**xgb_params)
    xm.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
    preds_xgb.extend(xm.predict(X[te]))
    true_xgb.extend(y[te]); dates_xgb.extend(dates[te])

res_xgb = get_metrics(np.array(true_xgb), np.array(preds_xgb), 'XGBoost')
print(f"   {res_xgb}")
all_results['XGB'] = {'cv_pred': np.array(preds_xgb), 'cv_true': np.array(true_xgb),
                      'cv_dates': dates_xgb, 'metrics': res_xgb}

xgb_final = xgb.XGBRegressor(**xgb_params)
xgb_final.fit(X, y, verbose=False)
xgb_imp = pd.Series(xgb_final.feature_importances_, index=features).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# 7. MODÈLE 4 : LSTM (deep learning — séries temporelles)
#    Architecture : LSTM(64) → LSTM(32) → Dense(16) → Dense(1)
#    Séquences de SEQ_LEN=14 jours (cohérent avec autocorrélation SM1)
# ══════════════════════════════════════════════════════════════════════════════

print("[4/4] LSTM...", flush=True)
SEQ_LEN = 14     # jours de contexte

def make_sequences(Xarr, yarr, seq_len):
    """Découpe des arrays en séquences temporelles pour LSTM."""
    Xs = [Xarr[i - seq_len:i] for i in range(seq_len, len(Xarr))]
    ys = [yarr[i] for i in range(seq_len, len(yarr))]
    return np.array(Xs), np.array(ys)

preds_lstm, true_lstm, dates_lstm = [], [], []

for fi, (tr, te) in enumerate(folds):
    t0 = time.time()

    # Normalisation (fit sur train seulement)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
    ytr = y[tr]; yte = y[te]

    # Construction des séquences
    Xs_tr, ys_tr = make_sequences(Xtr, ytr, SEQ_LEN)
    Xs_te, ys_te = make_sequences(Xte, yte, SEQ_LEN)
    d_te = dates[te[SEQ_LEN:]]
    if len(Xs_te) == 0:
        continue

    # Architecture LSTM
    tf.keras.backend.clear_session()
    model = Sequential([
        Input(shape=(SEQ_LEN, X.shape[1])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5)
    ]
    model.fit(Xs_tr, ys_tr, validation_data=(Xs_te, ys_te),
              epochs=150, batch_size=32, callbacks=callbacks, verbose=0)

    yp = model.predict(Xs_te, verbose=0).flatten()
    preds_lstm.extend(yp); true_lstm.extend(ys_te); dates_lstm.extend(d_te)

    fold_rmse = np.sqrt(mean_squared_error(ys_te, yp))
    fold_r2   = r2_score(ys_te, yp)
    print(f"   Fold {fi+1} : RMSE={fold_rmse:.3f}  R²={fold_r2:.3f}  ({time.time()-t0:.0f}s)",
          flush=True)

res_lstm = get_metrics(np.array(true_lstm), np.array(preds_lstm), 'LSTM')
print(f"   Global : {res_lstm}")
all_results['LSTM'] = {'cv_pred': np.array(preds_lstm), 'cv_true': np.array(true_lstm),
                       'cv_dates': dates_lstm, 'metrics': res_lstm}

# ══════════════════════════════════════════════════════════════════════════════
# 8. TABLEAU RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"{'Modèle':<22} {'RMSE':>6} {'MAE':>6} {'R²':>6} {'NSE':>6} {'PBIAS':>8}")
print("=" * 65)
for k in all_results:
    m = all_results[k]['metrics']
    print(f"{m['Model']:<22} {m['RMSE']:>6.3f} {m['MAE']:>6.3f} "
          f"{m['R2']:>6.3f} {m['NSE']:>6.3f} {m['PBIAS']:>7.2f}%")
print("=" * 65)

colors_m = ['#1a6faf', '#2ca02c', '#e07b39', '#d62728']

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIGURE 1 — TABLEAU PERFORMANCES + RADAR
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 6)); fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

# Tableau coloré
ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
models = [all_results[k]['metrics']['Model'] for k in all_results]
rmse_v = [all_results[k]['metrics']['RMSE']  for k in all_results]
mae_v  = [all_results[k]['metrics']['MAE']   for k in all_results]
r2_v   = [all_results[k]['metrics']['R2']    for k in all_results]
nse_v  = [all_results[k]['metrics']['NSE']   for k in all_results]
pbias_v= [all_results[k]['metrics']['PBIAS'] for k in all_results]

def cmap_cell(val, vmin, vmax, good='low'):
    n = (val - vmin) / (vmax - vmin + 1e-6)
    if good == 'low': n = 1 - n
    return (min(1, 1 - n*0.5), min(1, 0.5 + n*0.4), 0.5)

cell_colors = []
for i in range(len(models)):
    cell_colors.append([
        (0.95, 0.95, 0.95),
        cmap_cell(rmse_v[i], min(rmse_v), max(rmse_v), 'low'),
        cmap_cell(mae_v[i],  min(mae_v),  max(mae_v),  'low'),
        cmap_cell(r2_v[i],   min(r2_v),   max(r2_v),   'high'),
        cmap_cell(nse_v[i],  min(nse_v),  max(nse_v),  'high'),
        (0.95, 0.95, 0.95),
    ])

tab_data = [[models[i], f'{rmse_v[i]:.3f}', f'{mae_v[i]:.3f}',
             f'{r2_v[i]:.3f}', f'{nse_v[i]:.3f}', f'{pbias_v[i]:.1f}']
            for i in range(len(models))]
cols = ['Modèle', 'RMSE\n[Vol%]', 'MAE\n[Vol%]', 'R²', 'NSE', 'PBIAS\n(%)']
tbl  = ax0.table(cellText=tab_data, colLabels=cols, cellLoc='center',
                 loc='center', cellColours=cell_colors)
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_height(0.18)
    if r == 0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
ax0.set_title('Comparaison des performances\n(CV temporelle, 5 folds, gap=30j)',
              fontsize=11, fontweight='bold', pad=12)

# Radar chart
ax1 = fig.add_subplot(gs[1], polar=True)
cats   = ['R²', 'NSE', '1-RMSE\n(norm)', '1-MAE\n(norm)', '|PBIAS|\n(inv)']
N      = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]

for ki, (k, c) in enumerate(zip(all_results.keys(), colors_m)):
    m_ = all_results[k]['metrics']
    vals = [
        m_['R2'],
        max(0, m_['NSE']),
        1 - m_['RMSE'] / (max(rmse_v) + 0.1),
        1 - m_['MAE']  / (max(mae_v)  + 0.1),
        1 - abs(m_['PBIAS']) / 20,
    ]
    vals += vals[:1]
    ax1.plot(angles, vals, lw=1.8, color=c, label=m_['Model'].split()[0])
    ax1.fill(angles, vals, alpha=0.08, color=c)

ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(cats, fontsize=8)
ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax1.set_title('Radar des performances\n(normalisé)', fontsize=10, fontweight='bold', pad=15)

plt.savefig('m3_fig1_perf_table.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("\n✓ m3_fig1_perf_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE 2 — OBSERVED vs PREDICTED (scatter 1:1)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(15, 11)); fig.patch.set_facecolor('white')
fig.suptitle('Observations vs Prédictions — Validation croisée temporelle',
             fontsize=13, fontweight='bold')

for ax, (k, c) in zip(axes.flat, zip(all_results.keys(), colors_m)):
    d  = all_results[k]; m_ = d['metrics']
    yt = d['cv_true']; yp = d['cv_pred']
    lims = [min(min(yt), min(yp)) - 0.5, max(max(yt), max(yp)) + 0.5]
    ax.scatter(yt, yp, alpha=0.35, s=10, color=c, zorder=2)
    ax.plot(lims, lims, 'k--', lw=1, zorder=3, label='1:1')
    xs = np.linspace(lims[0], lims[1], 100)
    ax.plot(xs, np.poly1d(np.polyfit(yt, yp, 1))(xs), color=c, lw=1.5, label='Régression')
    ax.set_xlabel('SM₁ observée [Vol%]', fontsize=10)
    ax.set_ylabel('SM₁ prédite [Vol%]', fontsize=10)
    ax.set_title(f"{m_['Model']}\nR²={m_['R2']} | NSE={m_['NSE']} | RMSE={m_['RMSE']} Vol%",
                 fontsize=10, fontweight='bold')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m3_fig2_obs_pred.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ m3_fig2_obs_pred.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURE 3 — SÉRIE TEMPORELLE (zoom sur un fold test)
# ══════════════════════════════════════════════════════════════════════════════

# Utilise le fold 3 (milieu de la période, ~2025)
fold_tr = np.arange(0, int(fold_size * 3))
fold_te = np.arange(int(fold_size * 3) + gap, int(fold_size * 4))
yobs = y[fold_te]; d_te = dates[fold_te]

# Prédictions sur ce fold
pipe_r = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10))])
pipe_r.fit(X[fold_tr], y[fold_tr]); yp_r = pipe_r.predict(X[fold_te])

rf_ = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                              max_features=0.5, n_jobs=-1, random_state=42)
rf_.fit(X[fold_tr], y[fold_tr]); yp_rf = rf_.predict(X[fold_te])

xm_ = xgb.XGBRegressor(**xgb_params)
xm_.fit(X[fold_tr], y[fold_tr], verbose=False); yp_xgb = xm_.predict(X[fold_te])

# Extraire prédictions LSTM pour ce fold depuis le stockage CV
lstm_d = np.array(all_results['LSTM']['cv_dates'])
lstm_p = all_results['LSTM']['cv_pred']
f_start = dates[fold_te[0]]; f_end = dates[fold_te[-1]]
mask_lstm = np.array([(pd.Timestamp(d) >= f_start) & (pd.Timestamp(d) <= f_end) for d in lstm_d])
lstm_d_f = [pd.Timestamp(dd) for dd in lstm_d[mask_lstm]]
lstm_p_f = lstm_p[mask_lstm]

fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
fig.patch.set_facecolor('white')
fig.suptitle(f'Séries temporelles — Fold test : {d_te[0].strftime("%b %Y")} → {d_te[-1].strftime("%b %Y")}',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(d_te, yobs, lw=2, color='black', label='Observé', zorder=5)
ax.plot(d_te, yp_r,  lw=1.3, color='#aaaaaa', ls='--', label=f'Ridge  R²={r2_score(yobs,yp_r):.3f}')
ax.plot(d_te, yp_rf, lw=1.5, color='#2ca02c',          label=f'RF     R²={r2_score(yobs,yp_rf):.3f}')
ax.set_ylabel('SM₁ [Vol%]', fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Ridge vs Random Forest', fontsize=10)

ax = axes[1]
ax.plot(d_te, yobs,    lw=2,   color='black',   label='Observé', zorder=5)
ax.plot(d_te, yp_xgb, lw=1.5, color='#e07b39', label=f'XGBoost R²={r2_score(yobs,yp_xgb):.3f}')
if len(lstm_p_f) > 0:
    ax.plot(lstm_d_f, lstm_p_f, lw=1.5, color='#d62728', label='LSTM')
ax.set_ylabel('SM₁ [Vol%]', fontsize=10); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('XGBoost vs LSTM', fontsize=10)

ax = axes[2]
ax.axhline(0, color='black', lw=0.8, ls='--')
for yp, c, lbl in zip([yp_r, yp_rf, yp_xgb], ['#aaa', '#2ca02c', '#e07b39'], ['Ridge', 'RF', 'XGB']):
    ax.plot(d_te, yobs - yp, lw=1, color=c, label=lbl, alpha=0.85)
ax.set_ylabel('Résidus (Obs − Pred) [Vol%]', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Analyse des résidus', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for ax in axes: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('m3_fig3_timeseries.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ m3_fig3_timeseries.png")

# ══════════════════════════════════════════════════════════════════════════════
# 12. FIGURE 4 — IMPORTANCE DES VARIABLES (RF + XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 9)); fig.patch.set_facecolor('white')
fig.suptitle('Importance des variables — Random Forest vs XGBoost (Top 20)',
             fontsize=12, fontweight='bold')

for ax, imp_ser, title, col in zip(axes, [rf_imp, xgb_imp], ['Random Forest', 'XGBoost'],
                                    ['#2ca02c', '#e07b39']):
    top20 = imp_ser.head(20); ypos = range(len(top20) - 1, -1, -1)
    ax.barh(list(ypos), top20.values, color=col, alpha=0.8, edgecolor='none')
    ax.set_yticks(list(ypos)); ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xlabel('Importance', fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
    for i, v in enumerate(top20.values):
        ax.text(v + 0.0005, len(top20) - 1 - i, f'{v:.3f}', va='center', fontsize=7.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('m3_fig4_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ m3_fig4_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 13. FIGURE 5 — ANALYSE DES ERREURS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5)); fig.patch.set_facecolor('white')
fig.suptitle('Analyse des erreurs et stabilité par fold', fontsize=12, fontweight='bold')

# (a) RMSE par tranche de test
ax = axes[0]; w = 0.18; x = np.arange(4)
for j, (k, c) in enumerate(zip(all_results.keys(), colors_m)):
    yt = np.array(all_results[k]['cv_true']); yp_ = np.array(all_results[k]['cv_pred'])
    n_f = len(yt); fs = n_f // 4
    fold_rmses = [float(np.sqrt(mean_squared_error(yt[i*fs:(i+1)*fs], yp_[i*fs:(i+1)*fs])))
                  for i in range(4) if i*fs < n_f and len(yt[i*fs:(i+1)*fs]) > 0]
    xi = x[:len(fold_rmses)] + j*w - 0.27
    ax.bar(xi, fold_rmses, width=w, color=c, label=k.split()[0], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(4)], fontsize=9)
ax.set_ylabel('RMSE [Vol%]', fontsize=10); ax.legend(fontsize=8)
ax.set_title('RMSE par période de test', fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (b) Distribution des résidus
ax = axes[1]
for k, c in zip(all_results.keys(), colors_m):
    err = np.array(all_results[k]['cv_pred']) - np.array(all_results[k]['cv_true'])
    ax.hist(err, bins=40, alpha=0.5, color=c, label=k.split()[0], density=True)
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Erreur [Vol%]', fontsize=10); ax.set_ylabel('Densité', fontsize=10)
ax.set_title('Distribution des résidus', fontsize=10); ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (c) Erreur absolue vs valeur observée
ax = axes[2]
for k, c in zip(all_results.keys(), colors_m):
    yt = np.array(all_results[k]['cv_true'])
    err_abs = np.abs(np.array(all_results[k]['cv_pred']) - yt)
    order = np.argsort(yt); win = 20
    roll = [err_abs[order][max(0, i-win):i+win].mean() for i in range(len(err_abs))]
    ax.plot(yt[order], roll, lw=1.5, color=c, label=k.split()[0], alpha=0.85)
ax.set_xlabel('SM₁ observée [Vol%]', fontsize=10); ax.set_ylabel('|Erreur| moy. [Vol%]', fontsize=10)
ax.set_title('Erreur absolue vs valeur observée', fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('m3_fig5_error_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("✓ m3_fig5_error_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 14. EXPORT DES RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

rows = [all_results[k]['metrics'] for k in all_results]
pd.DataFrame(rows).to_csv('m3_results_summary.csv', index=False)

print("\n" + "=" * 60)
print("MOIS 3 TERMINÉ")
print("=" * 60)
print("Fichiers générés :")
print("  m3_results_summary.csv")
print("  m3_fig1_perf_table.png  — tableau + radar")
print("  m3_fig2_obs_pred.png    — scatter 1:1")
print("  m3_fig3_timeseries.png  — séries temporelles")
print("  m3_fig4_importance.png  — feature importance")
print("  m3_fig5_error_analysis.png — analyse erreurs")

"""
════════════════════════════════════════════════════════════════════════════════
MOIS 2 — Ingénierie des Features
  • ET₀ Penman-Monteith FAO-56 (horaire)
  • Variables dérivées (VPD, bilan hydrique, amplitude thermique)
  • Encodage saisonnier (sin/cos)
  • Lag features (mémoire temporelle du sol)
  • Rolling windows & pluie cumulée
════════════════════════════════════════════════════════════════════════════════

Entrée  : fichiers -data-*.csv (même dossier)
Sorties : dataset_ML_features.csv
          m2_fig1_ET0.png, m2_fig2_seasonality.png,
          m2_fig3_xcorr.png, m2_fig4_corr.png
"""

# ── Dépendances ───────────────────────────────────────────────────────────────
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. PARAMÈTRES DU SITE  (à adapter selon votre station)
# ══════════════════════════════════════════════════════════════════════════════

LAT_DEG  = 33.57    # latitude (degrés) — Casablanca / Maroc
ALTITUDE = 603      # altitude (m) — déduite de P_moy ≈ 940 hPa

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT (identique Mois 1 — copiez-collez ou importez)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MOIS 2 — CHARGEMENT")
print("=" * 60)

files = sorted(glob.glob('-data-*.csv'))
dfs   = [pd.read_csv(f, sep=';', parse_dates=['Time'], dayfirst=True) for f in files]
df_raw = (pd.concat(dfs, ignore_index=True)
            .sort_values('Time')
            .drop_duplicates(subset='Time')
            .reset_index(drop=True))

df15 = df_raw.set_index('Time').resample('15min').mean(numeric_only=True)
df15.loc[df15['lysimeter weight [g]'] < 0, 'lysimeter weight [g]'] = np.nan

df_h  = df15.resample('1h').mean(numeric_only=True)
print(f"Grille 15 min : {len(df15)} pts | Horaire : {len(df_h)} pts")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ET₀ PENMAN-MONTEITH FAO-56 (pas horaire)
#    Référence : Allen et al., 1998, FAO Irrigation and Drainage Paper 56
#    Équation 53 (horaire)
# ══════════════════════════════════════════════════════════════════════════════

print("\nCalcul ET₀ Penman-Monteith FAO-56...")
lat = np.radians(LAT_DEG)

# Variables météo horaires
T   = df_h['air temperature [degC]']
HR  = df_h['air humidity [%]'].clip(1, 100)
Rs  = df_h['global radiation  [W/m2]'].clip(lower=0)          # W/m²
u2  = df_h['wind speed [m/s]'].clip(lower=0.5)                 # vent min FAO = 0.5 m/s
P   = df_h['air pressure [hPa]']                               # hPa

# Constante psychrométrique γ [kPa/°C]
gamma = 0.665e-3 * P / 1000

# Pression vapeur saturante et réelle [kPa]
es    = 0.6108 * np.exp(17.27 * T / (T + 237.3))
ea    = es * HR / 100
vpd   = (es - ea).clip(lower=0)                               # déficit

# Pente Δ de la courbe es(T) [kPa/°C]
delta = 4098 * es / (T + 237.3) ** 2

# Rayonnement net [MJ/m²/h]
Rs_MJ = Rs * 0.0036                                           # W/m² → MJ/m²/h

# Rayonnement extra-terrestre Ra [MJ/m²/h]
doy  = pd.Series(df_h.index.day_of_year.astype(float), index=df_h.index)
dr   = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)            # distance soleil-terre
sdec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)         # déclinaison solaire
hf   = pd.Series(df_h.index.hour + df_h.index.minute / 60., index=df_h.index)
ha   = (hf - 12) * np.pi / 12                                # angle horaire
cos_theta = (np.sin(lat) * np.sin(sdec)
             + np.cos(lat) * np.cos(sdec) * np.cos(ha)).clip(lower=0)
Ra   = (12 / np.pi) * 4.92 * dr * cos_theta                 # MJ/m²/h

# Rayonnement net courtes ondes Rns (albédo gazon = 0.23)
Rns  = 0.77 * Rs_MJ

# Rayonnement net grandes ondes Rnl
sig  = 4.903e-9 / 24                                         # MJ/m²/h/K⁴
Tk   = T + 273.16
Rs0  = (0.75 + 2e-5 * ALTITUDE) * Ra                        # ciel clair
fcd  = (1.35 * (Rs_MJ / Rs0.clip(lower=0.01)).clip(0.05, 1.0) - 0.35).clip(0.05, 1.0)
Rnl  = sig * Tk ** 4 * (0.34 - 0.14 * np.sqrt(ea.clip(lower=0.001))) * fcd

# Rayonnement net total
Rn   = (Rns - Rnl).clip(lower=-0.5)

# Flux chaleur du sol G (FAO-56, Eq 45-46)
is_day = pd.Series((df_h.index.hour >= 6) & (df_h.index.hour < 20), index=df_h.index)
G = np.where(is_day, 0.1 * Rn, 0.5 * Rn)

# ET₀ [mm/h] — FAO-56 Eq 53
Cn, Cd = 37, 0.24                                            # coefficients horaires
num   = 0.408 * delta * (Rn - G) + gamma * (Cn / (T + 273)) * u2 * vpd
denom = delta + gamma * (1 + Cd * u2)
df_h['ET0_mm_h'] = (num / denom).clip(lower=0)

print(f"  ET₀ annuelle ≈ {df_h['ET0_mm_h'].mean()*8760:.0f} mm/an  (réf. semi-aride : 900–1400)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. AGRÉGATION JOURNALIÈRE + VARIABLES DÉRIVÉES
# ══════════════════════════════════════════════════════════════════════════════

print("Construction du dataset journalier...")

day = df15.resample('1D').agg({
    'UMP 01 water content [Vol%]': ['mean', 'min', 'max', 'std'],
    'UMP 03 water content [Vol%]': ['mean', 'min', 'max'],
    'lysimeter weight [g]':        ['mean'],
    'air temperature [degC]':      ['mean', 'min', 'max'],
    'air humidity [%]':            ['mean', 'min'],
    'global radiation  [W/m2]':   ['mean', 'sum'],
    'rain sum [mm]':               ['sum', 'max'],
    'wind speed [m/s]':            ['mean', 'max'],
    'air pressure [hPa]':          ['mean'],
    'FRT 01 tension [kPa]':        ['mean'],
    'FRT 02 tension [kPa]':        ['mean'],
    'FRT 03 tension [kPa]':        ['mean'],
    'UMP 01 EC [mS/cm]':           ['mean'],
    'UMP 01 temperature [degC]':   ['mean'],
    'water discharge [ml]':        ['sum'],
})
day.columns = ['_'.join(c) for c in day.columns]

# Renommer pour lisibilité
rn = {
    'UMP 01 water content [Vol%]_mean': 'SM1',
    'UMP 01 water content [Vol%]_min':  'SM1_min',
    'UMP 01 water content [Vol%]_max':  'SM1_max',
    'UMP 01 water content [Vol%]_std':  'SM1_std',
    'UMP 03 water content [Vol%]_mean': 'SM3',
    'UMP 03 water content [Vol%]_min':  'SM3_min',
    'UMP 03 water content [Vol%]_max':  'SM3_max',
    'lysimeter weight [g]_mean':        'Lys_kg',
    'air temperature [degC]_mean':      'T_mean',
    'air temperature [degC]_min':       'T_min',
    'air temperature [degC]_max':       'T_max',
    'air humidity [%]_mean':            'RH',
    'air humidity [%]_min':             'RH_min',
    'global radiation  [W/m2]_mean':   'Rs_mean',
    'global radiation  [W/m2]_sum':    'Rs_sum',
    'rain sum [mm]_sum':                'Rain',
    'rain sum [mm]_max':                'Rain_max',
    'wind speed [m/s]_mean':           'Wind',
    'wind speed [m/s]_max':            'Wind_max',
    'air pressure [hPa]_mean':         'P',
    'FRT 01 tension [kPa]_mean':       'FRT1',
    'FRT 02 tension [kPa]_mean':       'FRT2',
    'FRT 03 tension [kPa]_mean':       'FRT3',
    'UMP 01 EC [mS/cm]_mean':          'EC1',
    'UMP 01 temperature [degC]_mean':  'Ts1',
    'water discharge [ml]_sum':        'Qdis',
}
day = day.rename(columns=rn)
day['Lys_kg'] /= 1000                                        # g → kg

# ET₀ journalière (somme horaire)
day['ET0'] = df_h['ET0_mm_h'].resample('1D').sum()

# Variables dérivées
day['DT']  = day['T_max'] - day['T_min']                    # amplitude thermique diurne
day['VPD'] = (0.6108 * np.exp(17.27 * day['T_mean'] / (day['T_mean'] + 237.3))
              * (1 - day['RH'] / 100))                      # déficit pression vapeur [kPa]
day['WB']  = day['Rain'] - day['ET0']                       # bilan hydrique [mm/j]
day['dLys'] = day['Lys_kg'].diff()                          # variation poids (ETR proxy)
day['rain_flag'] = (day['Rain'] > 1.0).astype(int)          # indicateur événement pluvieux

# ══════════════════════════════════════════════════════════════════════════════
# 4. ENCODAGE DE LA SAISONNALITÉ
# ══════════════════════════════════════════════════════════════════════════════

doy_s = day.index.day_of_year.astype(float)

# Encodage cyclique sin/cos (préserve la continuité 31 déc / 1 jan)
day['doy_sin']   = np.sin(2 * np.pi * doy_s / 365)
day['doy_cos']   = np.cos(2 * np.pi * doy_s / 365)
day['month_sin'] = np.sin(2 * np.pi * day.index.month.astype(float) / 12)
day['month_cos'] = np.cos(2 * np.pi * day.index.month.astype(float) / 12)
day['doy_norm']  = doy_s / 365.0                            # jour normalisé [0,1]

# Saison (catégorie numérique pour les modèles)
m = day.index.month
day['season_label'] = np.where(m.isin([12, 1, 2]),  'Hiver',
                      np.where(m.isin([3, 4, 5]),    'Printemps',
                      np.where(m.isin([6, 7, 8]),    'Ete', 'Automne')))
day['season_num'] = day['season_label'].map({'Hiver': 0, 'Printemps': 1, 'Ete': 2, 'Automne': 3})

# ══════════════════════════════════════════════════════════════════════════════
# 5. LAG FEATURES (mémoire temporelle du sol)
# ══════════════════════════════════════════════════════════════════════════════

# Lags individuels (jours)
lag_cols = ['SM1', 'FRT1', 'ET0', 'Rain', 'T_mean', 'VPD', 'WB']
for col in lag_cols:
    for lag in [1, 2, 3, 5, 7, 14]:
        day[f'{col}_L{lag}'] = day[col].shift(lag)

# Rolling windows (moyennes mobiles)
roll_cols = ['SM1', 'ET0', 'Rain', 'T_mean', 'VPD']
for col in roll_cols:
    for win in [3, 7, 14, 30]:
        day[f'{col}_R{win}'] = day[col].rolling(win, min_periods=int(win * 0.7)).mean()

# Pluie cumulée (recharge)
day['Rain_C3']  = day['Rain'].rolling(3,  min_periods=1).sum()
day['Rain_C7']  = day['Rain'].rolling(7,  min_periods=3).sum()
day['Rain_C14'] = day['Rain'].rolling(14, min_periods=5).sum()
day['Rain_C30'] = day['Rain'].rolling(30, min_periods=14).sum()

# ET₀ cumulée (stress hydrique accumulé)
day['ET0_C7']  = day['ET0'].rolling(7,  min_periods=3).sum()
day['ET0_C14'] = day['ET0'].rolling(14, min_periods=7).sum()

# Jours secs consécutifs
no_rain = (day['Rain'] < 1.0).astype(int)
day['dry_days'] = no_rain.groupby((no_rain != no_rain.shift()).cumsum()).cumsum()

# Dataset ML final (supprimer les premières lignes avec NaN dus aux lags max=14)
day_ml = day.dropna(subset=['SM1', 'SM1_L14']).copy()

print(f"Dataset ML final : {day_ml.shape}")
print(f"  Base + dérivées : {sum(1 for c in day_ml.columns if 'L' not in c and 'R' not in c and 'C' not in c)} features")
print(f"  Lag features    : {sum(1 for c in day_ml.columns if '_L' in c)} features")
print(f"  Rolling         : {sum(1 for c in day_ml.columns if '_R' in c)} features")
print(f"  Cumulées        : {sum(1 for c in day_ml.columns if '_C' in c)} features")
print(f"  TOTAL           : {day_ml.shape[1]} colonnes")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 1 — ET₀ ET BILAN HYDRIQUE
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.patch.set_facecolor('white')
fig.suptitle('Mois 2 — ET₀ Penman-Monteith & bilans hydriques', fontsize=13, fontweight='bold')

ax = axes[0]
ax.fill_between(day.index, day['ET0'].interpolate(), alpha=0.4, color='#e07b39')
ax.plot(day.index, day['ET0'].interpolate(), lw=1, color='#e07b39')
ax.set_ylabel('ET₀ [mm/j]', fontsize=10)
ax.set_title('Évapotranspiration de référence (Penman-Monteith FAO-56)', fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
wb = day['WB'].interpolate()
ax.bar(day.index, day['Rain'].fillna(0), color='#3182bd', alpha=0.7, label='Pluie', width=1)
ax.plot(day.index, -day['ET0'].interpolate(), lw=1, color='#e07b39', label='-ET₀')
ax.fill_between(day.index, wb.clip(lower=0), alpha=0.3, color='green', label='Excédent')
ax.fill_between(day.index, wb.clip(upper=0), alpha=0.3, color='red', label='Déficit')
ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_ylabel('mm/j', fontsize=10)
ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.2)
ax.set_title('Bilan hydrique journalier (Pluie − ET₀)', fontsize=10)

ax = axes[2]
ax.plot(day.index, day['Rain_C7'].fillna(0),  lw=1.2, color='#1f77b4', label='Σ pluie 7j')
ax.plot(day.index, day['Rain_C14'].fillna(0), lw=1.2, color='#2ca02c', label='Σ pluie 14j')
ax.plot(day.index, day['Rain_C30'].fillna(0), lw=1.5, color='#9467bd', label='Σ pluie 30j')
ax.set_ylabel('Pluie cumulée [mm]', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Features de mémoire hydrique (pluie cumulée)', fontsize=10)

ax = axes[3]
ax.plot(day.index, day['SM1'].interpolate(),    lw=1,   color='#1a6faf', alpha=0.6, label='SM₁ brut')
ax.plot(day.index, day['SM1_L1'].interpolate(), lw=1,   color='#d62728', alpha=0.7, ls='--', label='SM₁ lag-1j')
ax.plot(day.index, day['SM1_R7'].interpolate(), lw=1.8, color='#2ca02c', label='SM₁ rolling 7j')
ax.set_ylabel('Humidité sol [Vol%]', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('SM brute vs lag-1 vs rolling-7j', fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

for ax in axes:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(h_pad=0.8)
plt.savefig('m2_fig1_ET0.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✓ m2_fig1_ET0.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 2 — ENCODAGE SAISONNALITÉ
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor('white')
fig.suptitle('Encodage de la saisonnalité et features temporelles', fontsize=13, fontweight='bold')
pal   = {'Hiver': '#4c72b0', 'Printemps': '#55a868', 'Ete': '#c44e52', 'Automne': '#dd8452'}
order = ['Hiver', 'Printemps', 'Ete', 'Automne']

# (a) Cercle sin/cos
ax = axes[0, 0]
sc = ax.scatter(day['doy_sin'], day['doy_cos'],
                c=day.index.day_of_year, cmap='hsv', s=4, alpha=0.4)
plt.colorbar(sc, ax=ax, label="Jour de l'année")
ax.set_xlabel('doy_sin'); ax.set_ylabel('doy_cos')
ax.set_title('Encodage cyclique sin/cos\n(continuité jan/déc)', fontsize=9)
ax.set_aspect('equal'); ax.grid(alpha=0.3)

# (b) SM1 vs sin(DOY) par saison
ax = axes[0, 1]
for s, grp in day.dropna(subset=['SM1', 'season_label']).groupby('season_label'):
    if s in pal:
        ax.scatter(grp['doy_sin'], grp['SM1'], c=pal[s], s=5, alpha=0.35, label=s)
ax.set_xlabel('doy_sin'); ax.set_ylabel('SM₁ [Vol%]')
ax.set_title('SM₁ vs sin(DOY) par saison', fontsize=9)
ax.legend(fontsize=8, markerscale=2); ax.grid(alpha=0.3)

# (c) ET₀ mensuelle
ax = axes[0, 2]
et0_m = day.groupby(day.index.month)['ET0'].mean()
colors_m = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 12))
ax.bar(range(1, 13), et0_m.values, color=colors_m)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=8)
ax.set_ylabel('ET₀ moy. [mm/j]'); ax.set_title('ET₀ moyenne mensuelle', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (d) VPD par saison
ax = axes[1, 0]
sns.boxplot(data=day.dropna(subset=['VPD', 'season_label']),
            x='season_label', y='VPD', order=order, palette=pal,
            ax=ax, linewidth=0.8, fliersize=2)
ax.set_xlabel(''); ax.set_ylabel('VPD [kPa]')
ax.set_title('Déficit pression vapeur par saison', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (e) Jours secs par saison
ax = axes[1, 1]
sns.boxplot(data=day.dropna(subset=['dry_days', 'season_label']),
            x='season_label', y='dry_days', order=order, palette=pal,
            ax=ax, linewidth=0.8, fliersize=2)
ax.set_xlabel(''); ax.set_ylabel('Jours secs consécutifs')
ax.set_title('Séquences sèches par saison', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (f) Autocorrélation SM1
ax = axes[1, 2]
sm_ = day['SM1'].dropna()
acf = [sm_.autocorr(lag=i) for i in range(0, 31)]
ci  = 1.96 / np.sqrt(len(sm_))
ax.bar(range(31), acf, color=['#1a6faf' if v > 0 else '#d62728' for v in acf], alpha=0.7)
ax.axhline(0, color='k', lw=0.7)
ax.axhline(ci, color='gray', ls='--', lw=0.8)
ax.axhline(-ci, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Lag (jours)'); ax.set_ylabel('Autocorrélation')
ax.set_title('Autocorrélation SM₁ → choix des lags', fontsize=9)
ax.grid(alpha=0.3)

for ax in axes.flat:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('m2_fig2_seasonality.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ m2_fig2_seasonality.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 3 — CORRÉLATION CROISÉE (cross-correlation) SM1 vs features
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor('white')
fig.suptitle('Corrélation croisée SM₁ vs features à différents lags\n(justification du choix des lags)',
             fontsize=13, fontweight='bold')

predictors = [
    ('Rain',  'Pluie [mm/j]',          '#3182bd'),
    ('ET0',   'ET₀ [mm/j]',            '#e07b39'),
    ('T_mean', 'T° air [°C]',          '#d62728'),
    ('FRT1',  'Tension FRT-01 [kPa]',  '#8c564b'),
    ('VPD',   'VPD [kPa]',             '#9467bd'),
    ('WB',    'Bilan hydrique [mm]',   '#2ca02c'),
]
for ax, (pred, lbl, clr) in zip(axes.flat, predictors):
    xcorr = [(lag, day['SM1'].corr(day[pred].shift(lag))) for lag in range(-7, 22)]
    lags_a = [x[0] for x in xcorr]; corr_a = [x[1] for x in xcorr]
    best_lag = lags_a[int(np.argmax(np.abs(corr_a)))]
    ax.bar(lags_a, corr_a, color=[clr if v >= 0 else '#aaa' for v in corr_a], alpha=0.75)
    ax.axhline(0, color='k', lw=0.7)
    ax.axvline(best_lag, color='red', ls='--', lw=1.2, label=f'Best lag = {best_lag}j')
    ax.set_xlabel('Lag (jours)', fontsize=9); ax.set_ylabel('r (Pearson)', fontsize=9)
    ax.set_title(lbl, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('m2_fig3_xcorr.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ m2_fig3_xcorr.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIGURE 4 — HEATMAP CORRÉLATIONS TOUTES FEATURES
# ══════════════════════════════════════════════════════════════════════════════

feat_sel = [
    'SM1', 'SM1_std', 'SM3', 'Lys_kg',
    'ET0', 'Rain', 'Rain_C7', 'T_mean', 'DT', 'RH',
    'VPD', 'WB', 'Rs_mean', 'Wind', 'P',
    'FRT1', 'FRT2', 'FRT3',
    'dry_days', 'doy_sin', 'doy_cos',
    'SM1_L3', 'SM1_L7', 'SM1_R7', 'SM1_R14',
    'Rain_L1', 'Rain_L3', 'ET0_L1', 'ET0_L3', 'Rain_R7',
]
feat_sel = [c for c in feat_sel if c in day_ml.columns]

corr_feat = day_ml[feat_sel].corr()
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('white')
mask = np.triu(np.ones_like(corr_feat, dtype=bool))
sns.heatmap(corr_feat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.3, ax=ax,
            annot_kws={'size': 6.5}, square=True)
ax.set_title('Corrélations Pearson — toutes features vs SM₁ (dataset ML enrichi)',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('m2_fig4_corr.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ m2_fig4_corr.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. EXPORT DU DATASET ML
# ══════════════════════════════════════════════════════════════════════════════

day_ml.to_csv('dataset_ML_features.csv')

print("\n" + "=" * 60)
print("MOIS 2 TERMINÉ")
print("=" * 60)
print(f"  dataset_ML_features.csv  → {day_ml.shape}")
print("  4 figures générées (m2_fig1 → m2_fig4)")
print("\nTop corrélations avec SM1 :")
feat_num = [c for c in day_ml.columns if day_ml[c].dtype in ['float64', 'int64'] and c != 'SM1']
corr_sm1 = day_ml[['SM1'] + feat_num].corr()['SM1'].drop('SM1').sort_values(key=abs, ascending=False)
print(corr_sm1.head(15).round(3).to_string())

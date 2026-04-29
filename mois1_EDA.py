"""
════════════════════════════════════════════════════════════════════════════════
MOIS 1 — Collecte, Exploration & Nettoyage des Données
Sujet : Modélisation ML de l'Humidité du Sol — Conditions Semi-Arides
════════════════════════════════════════════════════════════════════════════════

Fichiers d'entrée attendus dans le même dossier :
  - -data-*.csv          (données lysimètre, séparateur ';')
  - Weather_data-*.csv   (données météo externes, optionnel)

Sorties générées :
  - dataset_15min_clean.csv
  - dataset_daily_clean.csv
  - fig1_series_temporelles.png
  - fig2_correlation.png
  - fig3_saisonnalite.png
  - fig4_missing.png
  - fig5_scatter.png
  - fig6_zoom_rain.png
"""

# ── Dépendances ───────────────────────────────────────────────────────────────
# pip install pandas numpy matplotlib seaborn scipy
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # sans interface graphique
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET FUSION DES FICHIERS LYSIMÈTRE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MOIS 1 — CHARGEMENT ET FUSION")
print("=" * 60)

# Lire tous les fichiers -data-*.csv
files = sorted(glob.glob('-data-*.csv'))
print(f"{len(files)} fichiers trouvés :")
dfs = []
for f in files:
    df = pd.read_csv(f, sep=';', parse_dates=['Time'], dayfirst=True)
    df['source_file'] = f
    dfs.append(df)
    print(f"  {f} → {len(df)} lignes, {df.shape[1]} colonnes")

# Concaténer, trier par temps, supprimer les doublons
df_raw = (pd.concat(dfs, ignore_index=True)
            .sort_values('Time')
            .drop_duplicates(subset='Time')
            .reset_index(drop=True))

print(f"\nDataset brut : {df_raw.shape}")
print(f"Période      : {df_raw.Time.min()} → {df_raw.Time.max()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RÉÉCHANTILLONNAGE SUR GRILLE RÉGULIÈRE 15 MINUTES
# ══════════════════════════════════════════════════════════════════════════════

# Les fichiers ont deux fréquences entrelacées (~2 min et ~13 min)
# → resample à 15 min élimine l'artefact et homogénéise la grille
df_15 = df_raw.set_index('Time').resample('15min').mean(numeric_only=True)

# Corriger les valeurs aberrantes connues
df_15.loc[df_15['lysimeter weight [g]'] < 0, 'lysimeter weight [g]'] = np.nan
# UMP 02 défaillant (toujours = 0 → marquer NaN)
if 'UMP 02 water content [Vol%]' in df_15.columns:
    df_15['UMP 02 water content [Vol%]'] = np.where(
        df_15['UMP 02 water content [Vol%]'] == 0, np.nan,
        df_15['UMP 02 water content [Vol%]'])

print(f"\nGrille 15 min : {len(df_15)} points × {df_15.shape[1]} colonnes")

# ══════════════════════════════════════════════════════════════════════════════
# 3. AGRÉGATION JOURNALIÈRE
# ══════════════════════════════════════════════════════════════════════════════

df_day = df_15.resample('1D').agg({
    'UMP 01 water content [Vol%]': 'mean',
    'UMP 03 water content [Vol%]': 'mean',
    'lysimeter weight [g]':        'mean',
    'air temperature [degC]':      'mean',
    'air humidity [%]':            'mean',
    'global radiation  [W/m2]':   'mean',
    'rain sum [mm]':               'sum',
    'wind speed [m/s]':            'mean',
    'FRT 01 tension [kPa]':        'mean',
    'FRT 02 tension [kPa]':        'mean',
    'FRT 03 tension [kPa]':        'mean',
    'water discharge [ml]':        'sum',
})
df_day.columns = [
    'SM_UMP01', 'SM_UMP03', 'Lys_weight',
    'T_air', 'Humidity', 'Radiation', 'Rain', 'Wind',
    'FRT01_tension', 'FRT02_tension', 'FRT03_tension', 'Discharge'
]

print(f"Journalier   : {len(df_day)} jours")

# ══════════════════════════════════════════════════════════════════════════════
# 4. STATISTIQUES DESCRIPTIVES ET VALEURS MANQUANTES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("VALEURS MANQUANTES (grille 15 min) :")
print("─" * 60)
key_cols = [
    'UMP 01 water content [Vol%]', 'UMP 03 water content [Vol%]',
    'lysimeter weight [g]', 'air temperature [degC]',
    'air humidity [%]', 'global radiation  [W/m2]',
    'rain sum [mm]', 'wind speed [m/s]',
    'FRT 01 tension [kPa]', 'FRT 02 tension [kPa]', 'FRT 03 tension [kPa]',
]
for c in key_cols:
    if c in df_15.columns:
        pct = df_15[c].isna().mean() * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {c[:42]:<44} {bar} {pct:5.1f}%")

print("\nSTATISTIQUES DESCRIPTIVES (journalier) :")
print(df_day[['SM_UMP01', 'SM_UMP03', 'T_air', 'Rain', 'Radiation',
              'FRT01_tension']].describe().round(2).to_string())

print("\nCORRÉLATIONS avec SM_UMP01 :")
corr = df_day.corr()['SM_UMP01'].drop('SM_UMP01').sort_values(key=abs, ascending=False)
print(corr.round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — SÉRIE TEMPORELLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
fig.patch.set_facecolor('white')

# (a) Humidité du sol
ax = axes[0]
ax.fill_between(df_day.index, df_day['SM_UMP01'].interpolate(), alpha=0.35, color='#1a6faf')
ax.plot(df_day.index, df_day['SM_UMP01'].interpolate(), lw=1.2, color='#1a6faf', label='UMP 01')
ax.fill_between(df_day.index, df_day['SM_UMP03'].interpolate(), alpha=0.25, color='#e07b39')
ax.plot(df_day.index, df_day['SM_UMP03'].interpolate(), lw=1.2, color='#e07b39', label='UMP 03')
ax.set_ylabel('Humidité sol [Vol%]', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Série temporelle — Lysimètre (mai 2023 → mars 2026)', fontsize=13, fontweight='bold')

# (b) Poids lysimètre
ax = axes[1]
lw_ = df_day['Lys_weight'].interpolate()
ax.plot(df_day.index, lw_ / 1e6, lw=1, color='#2ca02c')
ax.fill_between(df_day.index, lw_ / 1e6, lw_.min() / 1e6, alpha=0.2, color='#2ca02c')
ax.set_ylabel('Poids lysimètre [tonnes]', fontsize=10); ax.grid(alpha=0.3)

# (c) Pluie + Température
ax = axes[2]; ax2b = ax.twinx()
ax.bar(df_day.index, df_day['Rain'].fillna(0), color='#3182bd', alpha=0.6, label='Pluie')
ax2b.plot(df_day.index, df_day['T_air'].interpolate(), lw=1, color='#d62728', label='T°air')
ax.set_ylabel('Pluie [mm/j]', fontsize=10, color='#3182bd')
ax2b.set_ylabel('T° air [°C]', fontsize=10, color='#d62728')
ax.legend(loc='upper left', fontsize=8); ax2b.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

# (d) Rayonnement + Humidité
ax = axes[3]; ax3b = ax.twinx()
ax.plot(df_day.index, df_day['Radiation'].interpolate(), lw=1, color='#ff7f0e', label='Rayonnement')
ax3b.plot(df_day.index, df_day['Humidity'].interpolate(), lw=1, color='#17becf', label='HR', alpha=0.8)
ax.set_ylabel('Rayonnement [W/m²]', fontsize=10, color='#ff7f0e')
ax3b.set_ylabel('Humidité air [%]', fontsize=10, color='#17becf')
ax.legend(loc='upper left', fontsize=8); ax3b.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)

# (e) Tensions matricielles FRT
ax = axes[4]
for col, lbl, clr in zip(['FRT01_tension', 'FRT02_tension', 'FRT03_tension'],
                          ['FRT-01', 'FRT-02', 'FRT-03'],
                          ['#8c564b', '#e377c2', '#bcbd22']):
    ax.plot(df_day.index, df_day[col].interpolate(), lw=1, label=lbl, color=clr)
ax.set_ylabel('Tension matricielle [kPa]', fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

for ax in axes:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(h_pad=0.8)
plt.savefig('fig1_series_temporelles.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✓ fig1_series_temporelles.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — MATRICE DE CORRÉLATION
# ══════════════════════════════════════════════════════════════════════════════

corr_vars = ['SM_UMP01', 'SM_UMP03', 'Lys_weight', 'T_air', 'Humidity',
             'Radiation', 'Rain', 'Wind', 'FRT01_tension', 'FRT02_tension',
             'FRT03_tension', 'Discharge']
labels = ['SM UMP01', 'SM UMP03', 'Poids lys.', 'T° air', 'Humidité',
          'Rayonnement', 'Pluie', 'Vent', 'FRT-01', 'FRT-02', 'FRT-03', 'Décharge']

corr_m = df_day[corr_vars].corr()
mask = np.triu(np.ones_like(corr_m, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('white')
sns.heatmap(corr_m, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax,
            xticklabels=labels, yticklabels=labels, annot_kws={'size': 8})
ax.set_title('Matrice de corrélation (Pearson) — données journalières',
             fontsize=12, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('fig2_correlation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig2_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 3 — ANALYSE SAISONNIÈRE
# ══════════════════════════════════════════════════════════════════════════════

df_day['season'] = df_day.index.month.map(
    lambda m: 'Hiver' if m in [12, 1, 2] else
              'Printemps' if m in [3, 4, 5] else
              'Été' if m in [6, 7, 8] else 'Automne'
)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor('white')
fig.suptitle('Analyse saisonnière — Humidité du sol & variables climatiques',
             fontsize=13, fontweight='bold', y=1.01)

pal = {'Hiver': '#4c72b0', 'Printemps': '#55a868', 'Été': '#c44e52', 'Automne': '#dd8452'}
order = ['Hiver', 'Printemps', 'Été', 'Automne']
pairs = [
    ('SM_UMP01', 'Humidité sol UMP01 [Vol%]'),
    ('SM_UMP03', 'Humidité sol UMP03 [Vol%]'),
    ('T_air', 'Température air [°C]'),
    ('Rain', 'Pluie [mm/j]'),
    ('Radiation', 'Rayonnement [W/m²]'),
    ('FRT01_tension', 'Tension matricielle FRT01 [kPa]'),
]
for ax, (col, lbl) in zip(axes.flat, pairs):
    d_ = df_day.dropna(subset=[col, 'season'])
    sns.boxplot(data=d_, x='season', y=col, order=order, palette=pal,
                ax=ax, linewidth=0.8, fliersize=2)
    ax.set_xlabel(''); ax.set_ylabel(lbl, fontsize=9)
    ax.set_title(lbl.split('[')[0].strip(), fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_saisonnalite.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig3_saisonnalite.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 4 — VALEURS MANQUANTES + COMPLÉTUDE
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# Barplot manquants
miss_cols = [
    'UMP 01 water content [Vol%]', 'UMP 03 water content [Vol%]',
    'lysimeter weight [g]', 'air temperature [degC]', 'air humidity [%]',
    'global radiation  [W/m2]', 'rain sum [mm]', 'wind speed [m/s]',
    'FRT 01 tension [kPa]', 'FRT 02 tension [kPa]', 'FRT 03 tension [kPa]',
    'water discharge [ml]',
]
miss_labels = ['SM UMP01', 'SM UMP03', 'Poids lys.', 'T° air', 'Humidité',
               'Rayonnement', 'Pluie', 'Vent', 'FRT-01', 'FRT-02', 'FRT-03', 'Décharge']
miss_pct = [df_15[c].isna().mean() * 100 for c in miss_cols if c in df_15.columns]
miss_labels = miss_labels[:len(miss_pct)]

clrs = ['#d62728' if p > 20 else '#ff7f0e' if p > 5 else '#2ca02c' for p in miss_pct]
ax1.barh(miss_labels, miss_pct, color=clrs, edgecolor='none', height=0.65)
ax1.axvline(5, color='gray', ls='--', lw=0.8, alpha=0.7)
ax1.axvline(20, color='red', ls='--', lw=0.8, alpha=0.5)
for i, pct in enumerate(miss_pct):
    ax1.text(pct + 0.3, i, f'{pct:.1f}%', va='center', fontsize=8.5)
ax1.set_xlabel('Données manquantes (%)', fontsize=10)
ax1.set_title('Taux de valeurs manquantes (grille 15 min)', fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3)

# Heatmap complétude mensuelle
df_15['year'] = df_15.index.year
df_15['month'] = df_15.index.month
pivot = df_15.groupby(['year', 'month'])['UMP 01 water content [Vol%]'].apply(
    lambda x: (1 - x.isna().mean()) * 100).unstack()
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGn', ax=ax2,
            xticklabels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'],
            linewidths=0.3, vmin=0, vmax=100, annot_kws={'size': 8})
ax2.set_title('Complétude SM UMP01 par mois (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Mois', fontsize=10); ax2.set_ylabel('Année', fontsize=10)

plt.tight_layout()
plt.savefig('fig4_missing.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig4_missing.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIGURE 5 — SCATTER PLOTS (SM vs prédicteurs)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor('white')
fig.suptitle('Relations humidité sol (UMP01) vs variables prédictives', fontsize=13, fontweight='bold')

pairs_sc = [
    ('FRT01_tension', 'Tension FRT-01 [kPa]', '#8c564b'),
    ('T_air',         'Température air [°C]', '#d62728'),
    ('Humidity',      'Humidité air [%]',     '#1f77b4'),
    ('Radiation',     'Rayonnement [W/m²]',   '#ff7f0e'),
    ('Rain',          'Pluie [mm/j]',         '#2ca02c'),
    ('Wind',          'Vent [m/s]',           '#9467bd'),
    ('Lys_weight',    'Poids lysimètre [g]',  '#17becf'),
    ('SM_UMP03',      'SM UMP03 [Vol%]',      '#e377c2'),
]
for ax, (col, lbl, clr) in zip(axes.flat, pairs_sc):
    d_ = df_day[['SM_UMP01', col]].dropna()
    ax.scatter(d_[col], d_['SM_UMP01'], alpha=0.25, s=8, color=clr)
    z = np.polyfit(d_[col], d_['SM_UMP01'], 1)
    xs = np.linspace(d_[col].min(), d_[col].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'k--', lw=1.2, alpha=0.8)
    r = d_.corr().iloc[0, 1]
    ax.set_title(f'r = {r:.3f}', fontsize=9)
    ax.set_xlabel(lbl, fontsize=8.5); ax.set_ylabel('SM UMP01 [Vol%]', fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('fig5_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig5_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE 6 — ZOOM ÉPISODE PLUVIEUX
# ══════════════════════════════════════════════════════════════════════════════

df_day['Rain_roll'] = df_day['Rain'].rolling(3).sum()
best_day = df_day['Rain_roll'].idxmax()
dz = df_day.loc[best_day - pd.Timedelta(days=10):best_day + pd.Timedelta(days=20)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.patch.set_facecolor('white')
fig.suptitle(f'Zoom sur un épisode pluvieux majeur', fontsize=12, fontweight='bold')

ax1.bar(dz.index, dz['Rain'].fillna(0), color='#3182bd', label='Pluie [mm/j]')
ax1.set_ylabel('Pluie [mm/j]', fontsize=10); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.fill_between(dz.index, dz['SM_UMP01'].interpolate(), alpha=0.4, color='#1a6faf')
ax2.plot(dz.index, dz['SM_UMP01'].interpolate(), lw=1.5, color='#1a6faf', label='UMP01')
ax2.fill_between(dz.index, dz['SM_UMP03'].interpolate(), alpha=0.25, color='#e07b39')
ax2.plot(dz.index, dz['SM_UMP03'].interpolate(), lw=1.5, color='#e07b39', label='UMP03')
ax2.set_ylabel('Humidité sol [Vol%]', fontsize=10); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

ax3.plot(dz.index, dz['FRT01_tension'].interpolate(), lw=1.2, color='#8c564b', label='FRT-01')
ax3.plot(dz.index, dz['FRT02_tension'].interpolate(), lw=1.2, color='#e377c2', label='FRT-02')
ax3.plot(dz.index, dz['FRT03_tension'].interpolate(), lw=1.2, color='#bcbd22', label='FRT-03')
ax3.set_ylabel('Tension matricielle [kPa]', fontsize=10)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig6_zoom_rain.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig6_zoom_rain.png")

# ══════════════════════════════════════════════════════════════════════════════
# 11. EXPORT DES DATASETS NETTOYÉS
# ══════════════════════════════════════════════════════════════════════════════

df_15.to_csv('dataset_15min_clean.csv')
df_day.to_csv('dataset_daily_clean.csv')

print("\n" + "=" * 60)
print("MOIS 1 TERMINÉ")
print("=" * 60)
print(f"  dataset_15min_clean.csv  → {len(df_15)} lignes × {df_15.shape[1]} colonnes")
print(f"  dataset_daily_clean.csv  → {len(df_day)} lignes × {df_day.shape[1]} colonnes")
print("  6 figures générées (fig1 → fig6)")

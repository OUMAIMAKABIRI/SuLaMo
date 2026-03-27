#!/usr/bin/env python3
"""
=============================================================================
  Prétraitement des données - PFE : ML-based Soil Moisture Modeling
  Zone semi-aride de Tafilalet (Errachidia, Maroc)
  Données : Lysimètre + Station météo | Période : 2023-2026
=============================================================================
"""

import pandas as pd
import numpy as np
import glob
import os

# ─────────────────────────────────────────────────────────────────────────────
# 0. CHEMINS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/"                # dossier contenant les CSV bruts
OUTPUT_FILE = "soil_moisture_preprocessed.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES LYSIMÈTRE (12 fichiers CSV)
# ─────────────────────────────────────────────────────────────────────────────
lys_files = sorted(glob.glob(os.path.join(DATA_DIR, "-data-*.csv")))
print(f"[1] {len(lys_files)} fichiers lysimètre trouvés")

dfs = []
for f in lys_files:
    df = pd.read_csv(f, sep=";", encoding="utf-8-sig", low_memory=False)
    dfs.append(df)

lys_raw = pd.concat(dfs, ignore_index=True)
print(f"    Lignes brutes : {len(lys_raw):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARSING DU TEMPS + TRI + DÉDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
lys_raw["Time"] = pd.to_datetime(lys_raw["Time"], dayfirst=True, errors="coerce")
lys_raw = (lys_raw
           .dropna(subset=["Time"])
           .sort_values("Time")
           .drop_duplicates(subset="Time")
           .reset_index(drop=True))
print(f"[2] Après tri/dédup : {len(lys_raw):,} lignes")
print(f"    Période : {lys_raw['Time'].min()} → {lys_raw['Time'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SÉLECTION DES COLONNES UTILES
# ─────────────────────────────────────────────────────────────────────────────
KEEP = {
    # Cibles : humidité volumique du sol (Vol%)
    "UMP 01 water content [Vol%]": "SM_UMP01_vol",   # profondeur 1 (cible principale)
    "UMP 03 water content [Vol%]": "SM_UMP03_vol",   # profondeur 3

    # Potentiel hydrique (tension matricielle, kPa)
    "FRT 01 tension [kPa]":  "T_FRT01_kPa",
    "FRT 02 tension [kPa]":  "T_FRT02_kPa",
    "FRT 03 tension [kPa]":  "T_FRT03_kPa",

    # Température du sol
    "UMP 01 temperature [degC]": "Tsoil_UMP01_C",
    "UMP 03 temperature [degC]": "Tsoil_UMP03_C",

    # Bilan hydrique lysimètre
    "lysimeter weight [g]":  "lys_weight_g",
    "water discharge [ml]":  "lys_discharge_ml",
    "rain sum [mm]":         "rain_mm",

    # Variables climatiques internes au capteur
    "air humidity [%]":      "RH_pct",
    "air temperature [degC]":"T_air_C",
    "global radiation  [W/m2]": "Rg_Wm2",
    "wind speed [m/s]":      "wind_ms",
}

lys = lys_raw[["Time"] + list(KEEP.keys())].rename(columns=KEEP).copy()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVERSION NUMÉRIQUE + REMPLACEMENT DES VALEURS ABERRANTES
# ─────────────────────────────────────────────────────────────────────────────
for col in lys.columns[1:]:
    lys[col] = pd.to_numeric(lys[col], errors="coerce")

# Valeur sentinel -1000 → NaN (capteur inactif ou non connecté)
sentinel_cols = [c for c in lys.columns if c != "Time"]
for col in sentinel_cols:
    lys.loc[lys[col] < -900, col] = np.nan

# Plages physiques plausibles
lys.loc[lys["SM_UMP01_vol"]  < 0,   "SM_UMP01_vol"]   = np.nan
lys.loc[lys["SM_UMP01_vol"]  > 60,  "SM_UMP01_vol"]   = np.nan   # max sol = ~55-60%
lys.loc[lys["SM_UMP03_vol"]  < 0,   "SM_UMP03_vol"]   = np.nan
lys.loc[lys["SM_UMP03_vol"]  > 60,  "SM_UMP03_vol"]   = np.nan
lys.loc[lys["lys_weight_g"]  < 0,   "lys_weight_g"]   = np.nan
lys.loc[lys["lys_weight_g"]  > 1e6, "lys_weight_g"]   = np.nan
lys.loc[lys["RH_pct"]        < 0,   "RH_pct"]         = np.nan
lys.loc[lys["RH_pct"]        > 100, "RH_pct"]         = np.nan
lys.loc[lys["Rg_Wm2"]        < 0,   "Rg_Wm2"]         = np.nan
lys.loc[lys["rain_mm"]       < 0,   "rain_mm"]         = np.nan

print(f"[4] Nettoyage terminé")

# ─────────────────────────────────────────────────────────────────────────────
# 5. RÉÉCHANTILLONNAGE À PAS DE TEMPS RÉGULIER (15 MINUTES)
# ─────────────────────────────────────────────────────────────────────────────
lys_15min = (lys.set_index("Time")
               .resample("15min")
               .mean())
print(f"[5] Rééchantillonné à 15 min : {len(lys_15min):,} lignes")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHARGEMENT + MERGE DONNÉES MÉTÉO
# ─────────────────────────────────────────────────────────────────────────────
weather = pd.read_csv(
    os.path.join(DATA_DIR, "Weather_data-_14-05-2023_to_14-05-2026.csv"),
    sep=";", encoding="utf-8-sig", low_memory=False
)
weather.columns = [c.strip() for c in weather.columns]
weather["Time"] = pd.to_datetime(weather["Time"], dayfirst=True, errors="coerce")
weather = (weather.dropna(subset=["Time"])
                  .sort_values("Time")
                  .drop_duplicates(subset="Time"))

# Garder uniquement les colonnes météo non présentes dans lys
weather_keep = {
    "air humidity [%]":          "RH_weather_pct",
    "air pressure [hPa]":        "P_hPa",
    "air temperature [degC]":    "T_weather_C",
    "global radiation  [W/m2]":  "Rg_weather_Wm2",
    "rain sum [mm]":             "rain_weather_mm",
    "wind speed [m/s]":          "wind_weather_ms",
}
weather = weather[["Time"] + list(weather_keep.keys())].rename(columns=weather_keep)
for col in weather.columns[1:]:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

weather_15min = weather.set_index("Time").resample("15min").mean()

# Merge (inner = seulement les timestamps communs)
merged = lys_15min.join(weather_15min, how="inner")
print(f"[6] Après merge météo : {len(merged):,} lignes | {merged.shape[1]} colonnes")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

# --- 7a. ETo par méthode Hargreaves-Samani (température + rayonnement) ---
#     ETo [mm/15min] = 0.0023 * (T + 17.8) * sqrt(ΔT) * Ra
#     Approximation : Ra ≈ Rg / 0.75  (rayonnement extra-terrestre)
#     On utilise T journalière → ici simplifiée à l'instant t
T_use = merged["T_weather_C"].fillna(merged["T_air_C"])
Rg_use = merged["Rg_weather_Wm2"].fillna(merged["Rg_Wm2"])
merged["ETo_mm"] = (0.0023 * (T_use + 17.8) * (T_use.clip(lower=0) ** 0.5)
                    * Rg_use * 0.0036 * 0.25).clip(lower=0)

# --- 7b. Statistiques glissantes (fenêtres 24h et 7 jours) ---
# 96 pas = 24h, 672 pas = 7j à résolution 15 min
for window, label in [(96, "24h"), (672, "7d")]:
    merged[f"T_roll_{label}"]     = T_use.rolling(window, min_periods=1).mean()
    merged[f"rain_cum_{label}"]   = merged["rain_weather_mm"].rolling(window, min_periods=1).sum()
    merged[f"Rg_roll_{label}"]    = Rg_use.rolling(window, min_periods=1).mean()
    merged[f"ETo_cum_{label}"]    = merged["ETo_mm"].rolling(window, min_periods=1).sum()

# --- 7c. Lags de la variable cible (mémoire du sol) ---
TARGET = "SM_UMP01_vol"
for lag in [4, 8, 96, 192]:   # 1h, 2h, 24h, 48h
    merged[f"SM_lag_{lag*15}min"] = merged[TARGET].shift(lag)

# --- 7d. Variables temporelles ---
merged["hour"]   = merged.index.hour
merged["month"]  = merged.index.month
merged["DOY"]    = merged.index.dayofyear    # jour de l'année (1-366)
merged["season"] = merged["month"].map(
    lambda m: "winter" if m in [12,1,2]
         else "spring" if m in [3,4,5]
         else "summer" if m in [6,7,8]
         else "autumn"
)
# Encodage sinus/cosinus (preserve la cyclicité pour les modèles ML)
merged["sin_DOY"]   = np.sin(2 * np.pi * merged["DOY"] / 365)
merged["cos_DOY"]   = np.cos(2 * np.pi * merged["DOY"] / 365)
merged["sin_hour"]  = np.sin(2 * np.pi * merged["hour"] / 24)
merged["cos_hour"]  = np.cos(2 * np.pi * merged["hour"] / 24)

# --- 7e. Variation du poids du lysimètre (proxy evapotranspiration réelle) ---
merged["delta_weight_g"] = merged["lys_weight_g"].diff(4)   # variation sur 1h

print(f"[7] Feature engineering → {merged.shape[1]} variables")

# ─────────────────────────────────────────────────────────────────────────────
# 8. INTERPOLATION DES LACUNES COURTES (≤ 2h)
# ─────────────────────────────────────────────────────────────────────────────
interp_cols = [TARGET, "SM_UMP03_vol", "T_FRT01_kPa", "T_FRT02_kPa", "T_FRT03_kPa",
               "lys_weight_g", "T_weather_C", "RH_weather_pct",
               "Rg_weather_Wm2", "rain_weather_mm", "wind_weather_ms"]
interp_cols = [c for c in interp_cols if c in merged.columns]

for col in interp_cols:
    merged[col] = merged[col].interpolate(method="time", limit=8)  # 8 * 15min = 2h

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUPPRESSION DES LIGNES SANS VARIABLE CIBLE
# ─────────────────────────────────────────────────────────────────────────────
df_clean = merged.dropna(subset=[TARGET]).reset_index()
print(f"[9] Dataset final : {len(df_clean):,} lignes | {df_clean.shape[1]} variables")
print(f"    Lacunes restantes (%) :\n"
      + df_clean.isnull().mean().mul(100).round(1)[
            df_clean.isnull().any()].sort_values(ascending=False).head(8).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 10. SPLIT TEMPOREL TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────────────────────────────────────
df_clean["Time"] = pd.to_datetime(df_clean["Time"])
train = df_clean[df_clean["Time"] < "2025-01-01"]
val   = df_clean[(df_clean["Time"] >= "2025-01-01") & (df_clean["Time"] < "2025-10-01")]
test  = df_clean[df_clean["Time"] >= "2025-10-01"]
print(f"\n[10] Split temporel :")
print(f"     Train : {len(train):,} lignes  ({train.Time.min().date()} → {train.Time.max().date()})")
print(f"     Val   : {len(val):,} lignes   ({val.Time.min().date()} → {val.Time.max().date()})")
print(f"     Test  : {len(test):,} lignes  ({test.Time.min().date()} → {test.Time.max().date()})")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────
df_clean.to_csv(OUTPUT_FILE, index=False)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print(f"\n[DONE] Fichiers sauvegardés : {OUTPUT_FILE}, train.csv, val.csv, test.csv")

"""
════════════════════════════════════════════════════════════════════════════════
MOIS 6 — Figures Publication-Ready & Rédaction Scientifique
════════════════════════════════════════════════════════════════════════════════

Ce script génère les figures finales publication-ready (300 dpi, style
IEEE/Elsevier) à partir du pipeline complet Mois 1→5.

Résultats attendus du Mois 5 (repris ici) :
  Ridge   : NSE = 0.850 | RMSE = 1.921 Vol% | PBIAS = +8.4%
  RF      : NSE = 0.757 | RMSE = 2.447 Vol% | PBIAS = +1.7%
  XGBoost : NSE = 0.749 | RMSE = 2.491 Vol% | PBIAS = +2.6%
  Stacking: NSE = 0.776 | RMSE = 2.364 Vol% | PBIAS = +1.1%  ← amélioration

Sorties :
  pub_fig1_overview.png     — Vue d'ensemble 3 ans (4 panneaux)
  pub_fig2_scatter.png      — Scatter obs vs pred 1:1 (3 modèles)
  pub_fig3_timeseries.png   — Série temporelle fold test + résidus + NSE glissant
  pub_fig4_shap.png         — SHAP bar + beeswarm XGBoost
  pub_fig5_seasonal.png     — Analyse saisonnière (4 panels)
  pub_fig6_table.png        — Tableau métriques formaté

Dépendances : pip install pandas numpy matplotlib seaborn scipy
              pip install scikit-learn xgboost tensorflow shap

Usage : python mois6_publication.py
        → Nécessite les fichiers -data-*.csv dans le même dossier
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
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import shap

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

warnings.filterwarnings('ignore')

# ── Style publication (IEEE / Elsevier) ───────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       300,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'lines.linewidth':  1.4,
})

C = {
    'obs':   '#1a1a2e',
    'ridge': '#6c757d',
    'rf':    '#2d6a4f',
    'xgb':   '#e07b39',
    'stack': '#1d3557',
    'rain':  '#1d3557',
    'et0':   '#e76f51',
}

print("=" * 60)
print("  MOIS 6 — FIGURES PUBLICATION-READY")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 1. RECONSTRUCTION DU DATASET (pipeline complet Mois 1 + 2)
# ══════════════════════════════════════════════════════════════════════════════

print("Chargement du dataset...", flush=True)
LAT_DEG, ALTITUDE = 33.57, 603
lat = np.radians(LAT_DEG)

files  = sorted(glob.glob('-data-*.csv'))
dfs    = [pd.read_csv(f, sep=';', parse_dates=['Time'], dayfirst=True) for f in files]
df_raw = (pd.concat(dfs, ignore_index=True)
            .sort_values('Time').drop_duplicates(subset='Time').reset_index(drop=True))
df15   = df_raw.set_index('Time').resample('15min').mean(numeric_only=True)
df15.loc[df15['lysimeter weight [g]'] < 0, 'lysimeter weight [g]'] = np.nan
df_h   = df15.resample('1h').mean(numeric_only=True)

# ET₀ Penman-Monteith FAO-56
T=df_h['air temperature [degC]']; HR=df_h['air humidity [%]'].clip(1,100)
Rs=df_h['global radiation  [W/m2]'].clip(lower=0); u2=df_h['wind speed [m/s]'].clip(lower=0.5)
P=df_h['air pressure [hPa]']
gamma=0.665e-3*P/1000; es=0.6108*np.exp(17.27*T/(T+237.3)); ea=es*HR/100; vpd=(es-ea).clip(lower=0)
delta=4098*es/(T+237.3)**2; Rs_MJ=Rs*0.0036
doy_h=pd.Series(df_h.index.day_of_year.astype(float),index=df_h.index)
dr=1+0.033*np.cos(2*np.pi*doy_h/365); sdec=0.409*np.sin(2*np.pi*doy_h/365-1.39)
hf=pd.Series(df_h.index.hour+df_h.index.minute/60.,index=df_h.index); ha=(hf-12)*np.pi/12
cos_t=(np.sin(lat)*np.sin(sdec)+np.cos(lat)*np.cos(sdec)*np.cos(ha)).clip(lower=0)
Ra=(12/np.pi)*4.92*dr*cos_t; Rns=0.77*Rs_MJ; sig=4.903e-9/24; Tk=T+273.16
Rs0=(0.75+2e-5*ALTITUDE)*Ra; fcd=(1.35*(Rs_MJ/Rs0.clip(lower=0.01)).clip(0.05,1.0)-0.35).clip(0.05,1.0)
Rnl=sig*Tk**4*(0.34-0.14*np.sqrt(ea.clip(lower=0.001)))*fcd; Rn=(Rns-Rnl).clip(lower=-0.5)
is_day=pd.Series((df_h.index.hour>=6)&(df_h.index.hour<20),index=df_h.index)
G=np.where(is_day,0.1*Rn,0.5*Rn)
df_h['ET0']=((0.408*delta*(Rn-G)+gamma*(37/(T+273))*u2*vpd)/(delta+gamma*(1+0.24*u2))).clip(lower=0)

# Agrégation journalière
day = df15.resample('1D').agg({
    'UMP 01 water content [Vol%]':['mean','std'],'UMP 03 water content [Vol%]':['mean'],
    'lysimeter weight [g]':['mean'],'air temperature [degC]':['mean','min','max'],
    'air humidity [%]':['mean','min'],'global radiation  [W/m2]':['mean'],
    'rain sum [mm]':['sum'],'wind speed [m/s]':['mean'],'air pressure [hPa]':['mean'],
    'FRT 01 tension [kPa]':['mean'],'FRT 02 tension [kPa]':['mean'],
    'FRT 03 tension [kPa]':['mean'],'UMP 01 EC [mS/cm]':['mean'],
    'UMP 01 temperature [degC]':['mean'],'water discharge [ml]':['sum'],
})
day.columns=['_'.join(c) for c in day.columns]
rn={'UMP 01 water content [Vol%]_mean':'SM1','UMP 01 water content [Vol%]_std':'SM1_std',
    'UMP 03 water content [Vol%]_mean':'SM3','lysimeter weight [g]_mean':'Lys_kg',
    'air temperature [degC]_mean':'T_mean','air temperature [degC]_min':'T_min',
    'air temperature [degC]_max':'T_max','air humidity [%]_mean':'RH','air humidity [%]_min':'RH_min',
    'global radiation  [W/m2]_mean':'Rs','rain sum [mm]_sum':'Rain','wind speed [m/s]_mean':'Wind',
    'air pressure [hPa]_mean':'P','FRT 01 tension [kPa]_mean':'FRT1',
    'FRT 02 tension [kPa]_mean':'FRT2','FRT 03 tension [kPa]_mean':'FRT3',
    'UMP 01 EC [mS/cm]_mean':'EC1','UMP 01 temperature [degC]_mean':'Ts1',
    'water discharge [ml]_sum':'Qdis'}
day=day.rename(columns=rn); day['Lys_kg']/=1000
day['ET0']=df_h['ET0'].resample('1D').sum()
day['DT']=day['T_max']-day['T_min']
day['VPD']=0.6108*np.exp(17.27*day['T_mean']/(day['T_mean']+237.3))*(1-day['RH']/100)
day['WB']=day['Rain']-day['ET0']; day['rain_flag']=(day['Rain']>1).astype(int)
doy_s=day.index.day_of_year.astype(float)
day['doy_sin']=np.sin(2*np.pi*doy_s/365); day['doy_cos']=np.cos(2*np.pi*doy_s/365)
day['month_sin']=np.sin(2*np.pi*day.index.month.astype(float)/12)
day['month_cos']=np.cos(2*np.pi*day.index.month.astype(float)/12)
m_=day.index.month
day['season']=np.where(m_.isin([12,1,2]),0,np.where(m_.isin([3,4,5]),1,np.where(m_.isin([6,7,8]),2,3)))
day['season_label']=np.where(m_.isin([12,1,2]),'Hiver',
                    np.where(m_.isin([3,4,5]),'Printemps',
                    np.where(m_.isin([6,7,8]),'Ete','Automne')))

for col in ['SM1','FRT1','ET0','Rain','T_mean','VPD','WB']:
    for lag in [1,2,3,5,7,14]: day[f'{col}_L{lag}']=day[col].shift(lag)
for col in ['SM1','ET0','Rain','T_mean','VPD']:
    for win in [3,7,14]: day[f'{col}_R{win}']=day[col].rolling(win,min_periods=int(win*0.6)).mean()
day['Rain_C7']=day['Rain'].rolling(7,min_periods=3).sum()
day['Rain_C14']=day['Rain'].rolling(14,min_periods=5).sum()
day['ET0_C7']=day['ET0'].rolling(7,min_periods=3).sum()
no_rain=(day['Rain']<1).astype(int)
day['dry_days']=no_rain.groupby((no_rain!=no_rain.shift()).cumsum()).cumsum()

day_ml=day.dropna(subset=['SM1','SM1_L14']).copy()
TARGET='SM1'; EXCL=['SM1_std','Lys_kg','SM1_L1','SM1_L2','season_label']
features=[c for c in day_ml.columns
          if c not in EXCL+[TARGET]
          and day_ml[c].dtype in ['float64','int64','int32',float,int]
          and day_ml[c].notna().mean()>0.8]
imp=SimpleImputer(strategy='median')
X=imp.fit_transform(day_ml[features]); y=day_ml[TARGET].values; dates=day_ml.index
print(f"Dataset: {day_ml.shape} | Features: {len(features)}")

# ── Métriques ─────────────────────────────────────────────────────────────────
def nse(o,s): return float(1-np.sum((o-s)**2)/np.sum((o-o.mean())**2))
def pbias(o,s): return float(100*(s-o).sum()/o.sum())

# ── CV et entraînement final ───────────────────────────────────────────────────
n=len(X); gap=30; fold_size=(n-gap)//6
folds=[]
for i in range(5):
    tr_end=int(fold_size*(i+2)); ts=tr_end+gap; te_end=min(ts+fold_size,n)
    if te_end>ts: folds.append((np.arange(0,tr_end),np.arange(ts,te_end)))

xp=dict(n_estimators=500,max_depth=5,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.7,min_child_weight=3,reg_alpha=0.1,reg_lambda=1.0,
        random_state=42,n_jobs=-1,tree_method='hist',verbosity=0)

print("Entraînement des modèles (CV + final)...", flush=True)

# CV pour scatter / métriques
all_true,all_pred_r,all_pred_rf,all_pred_xgb=[],[],[],[]
all_dates_cv=[]
for tr,te in folds:
    pipe=Pipeline([('sc',StandardScaler()),('m',Ridge(alpha=10))])
    pipe.fit(X[tr],y[tr]); all_pred_r.extend(pipe.predict(X[te]))
    rf_=RandomForestRegressor(n_estimators=200,min_samples_leaf=3,max_features=0.5,n_jobs=-1,random_state=42)
    rf_.fit(X[tr],y[tr]); all_pred_rf.extend(rf_.predict(X[te]))
    xm_=xgb.XGBRegressor(**xp); xm_.fit(X[tr],y[tr],verbose=False)
    all_pred_xgb.extend(xm_.predict(X[te]))
    all_true.extend(y[te]); all_dates_cv.extend(dates[te])

yt=np.array(all_true); yp_r_cv=np.array(all_pred_r)
yp_rf_cv=np.array(all_pred_rf); yp_xgb_cv=np.array(all_pred_xgb)
dates_cv=pd.DatetimeIndex(all_dates_cv)

metrics={
    'Ridge':   {'RMSE':float(np.sqrt(mean_squared_error(yt,yp_r_cv))),
                'MAE':float(mean_absolute_error(yt,yp_r_cv)),
                'R2':float(r2_score(yt,yp_r_cv)),
                'NSE':nse(yt,yp_r_cv),'PBIAS':pbias(yt,yp_r_cv)},
    'Random Forest':{'RMSE':float(np.sqrt(mean_squared_error(yt,yp_rf_cv))),
                'MAE':float(mean_absolute_error(yt,yp_rf_cv)),
                'R2':float(r2_score(yt,yp_rf_cv)),
                'NSE':nse(yt,yp_rf_cv),'PBIAS':pbias(yt,yp_rf_cv)},
    'XGBoost': {'RMSE':float(np.sqrt(mean_squared_error(yt,yp_xgb_cv))),
                'MAE':float(mean_absolute_error(yt,yp_xgb_cv)),
                'R2':float(r2_score(yt,yp_xgb_cv)),
                'NSE':nse(yt,yp_xgb_cv),'PBIAS':pbias(yt,yp_xgb_cv)},
}
print("Métriques CV :")
for k,v in metrics.items():
    print(f"  {k:<15} RMSE={v['RMSE']:.3f} NSE={v['NSE']:.3f} PBIAS={v['PBIAS']:+.1f}%")

# Modèles finaux (tout le dataset) pour SHAP
rf_final=RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.5,n_jobs=-1,random_state=42)
rf_final.fit(X,y)
xm_final=xgb.XGBRegressor(**xp); xm_final.fit(X,y,verbose=False)

# SHAP
print("Calcul SHAP...", flush=True)
expl_xgb=shap.TreeExplainer(xm_final)
shap_values_xgb=expl_xgb.shap_values(X)
shap_imp_xgb=pd.Series(np.abs(shap_values_xgb).mean(axis=0),index=features).sort_values(ascending=False)

expl_rf=shap.TreeExplainer(rf_final)
shap_values_rf=expl_rf.shap_values(X)
shap_imp_rf=pd.Series(np.abs(shap_values_rf).mean(axis=0),index=features).sort_values(ascending=False)

# Fold test pour fig3
fold_tr3=np.arange(0,int(fold_size*3)); fold_te3=np.arange(int(fold_size*3)+gap,int(fold_size*4))
yobs3=y[fold_te3]; d_te3=dates[fold_te3]
pipe_r3=Pipeline([('sc',StandardScaler()),('m',Ridge(alpha=10))]); pipe_r3.fit(X[fold_tr3],y[fold_tr3])
yp_r3=pipe_r3.predict(X[fold_te3])
rf3=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=0.5,n_jobs=-1,random_state=42)
rf3.fit(X[fold_tr3],y[fold_tr3]); yp_rf3=rf3.predict(X[fold_te3])
xm3=xgb.XGBRegressor(**xp); xm3.fit(X[fold_tr3],y[fold_tr3],verbose=False)
yp_xgb3=xm3.predict(X[fold_te3])

label_map={'SM1_R3':'SM rolling 3d','SM1_R7':'SM rolling 7d','SM1_R14':'SM rolling 14d',
           'EC1':'EC UMP01','SM1_L3':'SM lag 3d','SM1_L5':'SM lag 5d','SM1_L7':'SM lag 7d',
           'SM1_L14':'SM lag 14d','FRT1':'FRT-01 tension','FRT2':'FRT-02 tension',
           'FRT3':'FRT-03 tension','T_mean':'Air temp.','VPD':'VPD','ET0':'ET₀',
           'Rain':'Precipitation','Rain_C7':'Cumul. rain 7d','Rain_C14':'Cumul. rain 14d',
           'ET0_C7':'Cumul. ET₀ 7d','month_sin':'Month (sin)','doy_cos':'DOY (cos)'}
def nice(f): return label_map.get(f,f)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Vue d'ensemble 3 ans
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig. 1 — Vue d'ensemble...", flush=True)
fig=plt.figure(figsize=(14,11)); gs=gridspec.GridSpec(4,1,hspace=0.38,left=0.09,right=0.97,top=0.93,bottom=0.08)
ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
ax3=fig.add_subplot(gs[2],sharex=ax1); ax4=fig.add_subplot(gs[3],sharex=ax1)

ax1.fill_between(dates,day_ml['SM1'].interpolate(),alpha=0.25,color=C['rf'])
ax1.plot(dates,day_ml['SM1'].interpolate(),lw=1.2,color=C['rf'],label='UMP01 (10 cm)')
ax1.plot(dates,day_ml['SM3'].interpolate(),lw=1,color=C['xgb'],alpha=0.7,ls='--',label='UMP03 (30 cm)')
ax1.set_ylabel('Soil moisture\n[Vol%]',fontsize=10); ax1.legend(loc='upper right',ncol=2)
ax1.set_ylim(0,50); ax1.text(0.01,0.92,'(a)',transform=ax1.transAxes,fontsize=10,fontweight='bold')

ax2b=ax2.twinx()
ax2.bar(dates,day_ml['Rain'].fillna(0),color=C['rain'],alpha=0.6,width=1,label='Precipitation')
ax2b.plot(dates,day_ml['ET0'].interpolate(),lw=1,color=C['et0'],label='ET₀ (PM-FAO56)')
ax2.set_ylabel('Precipitation\n[mm d⁻¹]',fontsize=10,color=C['rain'])
ax2b.set_ylabel('ET₀\n[mm d⁻¹]',fontsize=10,color=C['et0'])
ax2.tick_params(axis='y',colors=C['rain']); ax2b.tick_params(axis='y',colors=C['et0'])
l1,lb1=ax2.get_legend_handles_labels(); l2,lb2=ax2b.get_legend_handles_labels()
ax2.legend(l1+l2,lb1+lb2,loc='upper right',ncol=2)
ax2.set_ylim(0,ax2.get_ylim()[1]*1.3)
ax2.text(0.01,0.92,'(b)',transform=ax2.transAxes,fontsize=10,fontweight='bold')
ax2b.spines['top'].set_visible(False)

ax3b=ax3.twinx()
ax3.plot(dates,day_ml['T_mean'].interpolate(),lw=1,color='#e63946',label='T_air')
ax3.fill_between(dates,day_ml['T_min'].interpolate(),day_ml['T_max'].interpolate(),alpha=0.12,color='#e63946')
ax3b.plot(dates,day_ml['RH'].interpolate(),lw=1,color='#457b9d',alpha=0.85,label='RH')
ax3.set_ylabel('Air temp.\n[°C]',fontsize=10,color='#e63946'); ax3b.set_ylabel('Rel. humidity\n[%]',fontsize=10,color='#457b9d')
ax3.tick_params(axis='y',colors='#e63946'); ax3b.tick_params(axis='y',colors='#457b9d')
l1,lb1=ax3.get_legend_handles_labels(); l2,lb2=ax3b.get_legend_handles_labels()
ax3.legend(l1+l2,lb1+lb2,loc='upper right',ncol=2)
ax3.text(0.01,0.92,'(c)',transform=ax3.transAxes,fontsize=10,fontweight='bold')
ax3b.spines['top'].set_visible(False)

for col,lbl,clr,ls in [('FRT1','FRT-01','#8c564b','-'),('FRT2','FRT-02','#c77dff','--'),('FRT3','FRT-03','#4cc9f0',':')]:
    if col in day_ml.columns:
        ax4.plot(dates,day_ml[col].interpolate(),lw=1,color=clr,ls=ls,label=lbl,alpha=0.85)
ax4.set_ylabel('Matric tension\n[kPa]',fontsize=10); ax4.legend(loc='upper right',ncol=3)
ax4.text(0.01,0.92,'(d)',transform=ax4.transAxes,fontsize=10,fontweight='bold')
ax4.set_xlabel('Date',fontsize=10)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
for ax in [ax1,ax2,ax3]: plt.setp(ax.get_xticklabels(),visible=False)
fig.suptitle('Fig. 1. Time series of lysimeter measurements and meteorological variables\nat the semi-arid experimental site (May 2023 – March 2026)',
             fontsize=10,style='italic',y=0.99)
plt.savefig('pub_fig1_overview.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig1_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter obs vs pred (3 modèles)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig. 2 — Scatter...", flush=True)
fig,axes=plt.subplots(1,3,figsize=(14,5))
fig.subplots_adjust(wspace=0.32,left=0.08,right=0.98,top=0.85,bottom=0.14)
model_keys=['Ridge','Random Forest','XGBoost']
model_preds=[yp_r_cv,yp_rf_cv,yp_xgb_cv]
model_colors=[C['ridge'],C['rf'],C['xgb']]
panel_ids=['(a)','(b)','(c)']
for ax,key,yp,col,pid in zip(axes,model_keys,model_preds,model_colors,panel_ids):
    m=metrics[key]
    lims=[max(0,min(yt.min(),yp.min())-1),max(yt.max(),yp.max())+1]
    ax.scatter(yt,yp,s=7,alpha=0.35,color=col,linewidths=0,rasterized=True)
    ax.plot(lims,lims,'k--',lw=1,label='1:1 line')
    z=np.polyfit(yt,yp,1); xs=np.linspace(lims[0],lims[1],200)
    ax.plot(xs,np.poly1d(z)(xs),color=col,lw=1.5,label='Regression')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel('Observed SM₁ [Vol%]',fontsize=10); ax.set_ylabel('Predicted SM₁ [Vol%]',fontsize=10)
    ax.set_title(key,fontsize=11,fontweight='bold')
    txt=(f"RMSE = {m['RMSE']:.2f} Vol%\nMAE  = {m['MAE']:.2f} Vol%\n"
         f"R²   = {m['R2']:.3f}\nNSE  = {m['NSE']:.3f}\nPBIAS = {m['PBIAS']:+.1f}%")
    ax.text(0.04,0.97,txt,transform=ax.transAxes,fontsize=8,va='top',ha='left',
            family='monospace',bbox=dict(boxstyle='round,pad=0.4',facecolor='white',alpha=0.85,edgecolor='#cccccc',lw=0.5))
    ax.text(0.97,0.04,pid,transform=ax.transAxes,fontsize=10,fontweight='bold',ha='right',va='bottom')
    ax.legend(loc='lower right',fontsize=8)
fig.suptitle('Fig. 2. Scatter plots of observed vs. predicted soil moisture (SM₁)\nduring temporal cross-validation (5 folds, 30-day gap)',fontsize=9.5,style='italic')
plt.savefig('pub_fig2_scatter.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig2_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Série temporelle test + résidus + NSE glissant
# ══════════════════════════════════════════════════════════════════════════════
print("Fig. 3 — Série temporelle...", flush=True)
rain3=day_ml['Rain'].iloc[fold_te3]
fig=plt.figure(figsize=(14,9)); gs=gridspec.GridSpec(3,1,height_ratios=[3,1.8,1.8],hspace=0.08,
               left=0.09,right=0.97,top=0.90,bottom=0.10)
ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1); ax3=fig.add_subplot(gs[2],sharex=ax1)

ax1.plot(d_te3,yobs3,lw=2,color=C['obs'],label='Observed',zorder=6)
ax1.plot(d_te3,yp_r3,lw=1.2,color=C['ridge'],ls='--',alpha=0.8,
         label=f'Ridge (RMSE={np.sqrt(mean_squared_error(yobs3,yp_r3)):.2f})')
ax1.plot(d_te3,yp_rf3,lw=1.4,color=C['rf'],
         label=f'Random Forest (RMSE={np.sqrt(mean_squared_error(yobs3,yp_rf3)):.2f})')
ax1.plot(d_te3,yp_xgb3,lw=1.4,color=C['xgb'],ls='-.',
         label=f'XGBoost (RMSE={np.sqrt(mean_squared_error(yobs3,yp_xgb3)):.2f})')
ax1b=ax1.twinx()
ax1b.bar(d_te3,rain3.values,color=C['rain'],alpha=0.4,width=1,label='Precip.')
ax1b.set_ylabel('Precip. [mm d⁻¹]',fontsize=8.5,color=C['rain']); ax1b.set_ylim(0,rain3.max()*5)
ax1b.spines['top'].set_visible(False)
ax1.set_ylabel('Soil moisture SM₁\n[Vol%]',fontsize=10); ax1.legend(loc='upper right',fontsize=8.5,ncol=2)
ax1.text(0.01,0.95,'(a)',transform=ax1.transAxes,fontsize=10,fontweight='bold')
plt.setp(ax1.get_xticklabels(),visible=False)

res_rf=yobs3-yp_rf3
ax2.axhline(0,color='k',lw=0.8,ls='--')
ax2.fill_between(d_te3,res_rf,0,where=res_rf>0,alpha=0.3,color=C['rf'])
ax2.fill_between(d_te3,res_rf,0,where=res_rf<0,alpha=0.3,color='#e63946')
ax2.plot(d_te3,res_rf,lw=1,color=C['rf'],label='RF residuals')
ax2.set_ylabel('Residuals\n[Vol%]',fontsize=10); ax2.legend(loc='upper right',fontsize=8)
ax2.text(0.01,0.92,'(b)',transform=ax2.transAxes,fontsize=10,fontweight='bold')
ax2.set_ylim(-10,10); ax2.yaxis.set_major_locator(ticker.MultipleLocator(4))
plt.setp(ax2.get_xticklabels(),visible=False)

win=30
def nse_roll(o,p,w): return [nse(o[max(0,i-w):i+1],p[max(0,i-w):i+1]) for i in range(len(o))]
nse_rf_r=nse_roll(yobs3,yp_rf3,win); nse_xgb_r=nse_roll(yobs3,yp_xgb3,win)
ax3.fill_between(d_te3,nse_rf_r,0.6,where=np.array(nse_rf_r)>0.6,alpha=0.2,color=C['rf'])
ax3.plot(d_te3,nse_rf_r,lw=1.2,color=C['rf'],label='RF (rolling NSE)')
ax3.plot(d_te3,nse_xgb_r,lw=1.2,color=C['xgb'],ls='-.',label='XGB (rolling NSE)')
ax3.axhline(0.6,color='#e63946',ls='--',lw=1,label='NSE = 0.60 threshold')
ax3.axhline(0,color='k',lw=0.5)
ax3.set_ylabel('Rolling NSE\n(30-day window)',fontsize=10); ax3.set_xlabel('Date',fontsize=10)
ax3.legend(loc='lower right',fontsize=8); ax3.set_ylim(-0.5,1.05)
ax3.text(0.01,0.92,'(c)',transform=ax3.transAxes,fontsize=10,fontweight='bold')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
fig.suptitle(f'Fig. 3. Temporal validation (test fold: {d_te3[0].strftime("%b %Y")} – {d_te3[-1].strftime("%b %Y")})\n'
             '(a) Observed vs. predicted SM₁; (b) RF residuals; (c) Rolling 30-day NSE',fontsize=9.5,style='italic')
plt.savefig('pub_fig3_timeseries.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig3_timeseries.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — SHAP bar + beeswarm
# ══════════════════════════════════════════════════════════════════════════════
print("Fig. 4 — SHAP...", flush=True)
top_n=12
top_feats_xgb=shap_imp_xgb.head(top_n).index.tolist()
union_feats=[f for f in top_feats_xgb if f in shap_imp_rf.index][:10]
fig,axes=plt.subplots(1,2,figsize=(15,7))
fig.subplots_adjust(wspace=0.40,left=0.15,right=0.98,top=0.88,bottom=0.10)

ax=axes[0]
xgb_vals=shap_imp_xgb[union_feats].values; rf_vals=shap_imp_rf[union_feats].values
ylbls=[nice(f) for f in union_feats]; y_pos=np.arange(len(union_feats))
ax.barh(y_pos+0.2,xgb_vals,0.38,color='#e07b39',alpha=0.85,label='XGBoost')
ax.barh(y_pos-0.2,rf_vals, 0.38,color='#2d6a4f',alpha=0.85,label='Random Forest')
ax.set_yticks(y_pos); ax.set_yticklabels(list(reversed(ylbls)),fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value| [Vol%]',fontsize=10)
ax.set_title('(a) Feature importance (SHAP)',fontsize=11,fontweight='bold')
ax.legend(fontsize=9,loc='lower right')

ax=axes[1]
top12_idx=[features.index(f) for f in top_feats_xgb if f in features]
top12_ok=[features[i] for i in top12_idx]; sv_top=shap_values_xgb[:,top12_idx]; X_top=X[:,top12_idx]
np.random.seed(42)
for row_i,feat in enumerate(top12_ok):
    y_pos_b=len(top12_ok)-1-row_i; sv=sv_top[:,row_i]; xv=X_top[:,row_i]
    p5,p95=np.nanpercentile(xv,5),np.nanpercentile(xv,95)
    xv_norm=np.clip((xv-p5)/(p95-p5+1e-8),0,1)
    jitter=np.random.uniform(-0.36,0.36,len(sv))
    sc=ax.scatter(sv,y_pos_b+jitter,c=xv_norm,cmap='RdBu_r',s=5,alpha=0.45,linewidths=0,vmin=0,vmax=1,rasterized=True)
ax.axvline(0,color='k',lw=0.8,ls='--')
ax.set_yticks(range(len(top12_ok)))
ax.set_yticklabels([nice(f) for f in reversed(top12_ok)],fontsize=9)
ax.set_xlabel('SHAP value (impact on SM₁) [Vol%]',fontsize=10)
ax.set_title('(b) SHAP beeswarm — XGBoost',fontsize=11,fontweight='bold')
cbar=plt.colorbar(sc,ax=ax,shrink=0.55,pad=0.03,aspect=20)
cbar.set_label('Feature value (normalized)',fontsize=8)
cbar.set_ticks([0,0.5,1]); cbar.set_ticklabels(['Low','Mid','High'],fontsize=8)
fig.suptitle('Fig. 4. SHAP-based feature importance analysis\n'
             '(a) Mean |SHAP| for RF and XGBoost; (b) SHAP value distribution for XGBoost',
             fontsize=9.5,style='italic')
plt.savefig('pub_fig4_shap.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig4_shap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Analyse saisonnière
# ══════════════════════════════════════════════════════════════════════════════
print("Fig. 5 — Saisonnalité...", flush=True)
yp_rf_all=rf_final.predict(X); yp_xgb_all=xm_final.predict(X)
df_res=pd.DataFrame({'obs':y,'rf':yp_rf_all,'xgb':yp_xgb_all,
                     'err_rf':yp_rf_all-y,'err_xgb':yp_xgb_all-y,
                     'season':day_ml['season_label'].values,
                     'T_mean':day_ml['T_mean'].values,'Rain':day_ml['Rain'].values},index=dates)
seasons=['Hiver','Printemps','Ete','Automne']; season_en={'Hiver':'Winter','Printemps':'Spring','Ete':'Summer','Automne':'Autumn'}
pal={'Hiver':'#4c72b0','Printemps':'#55a868','Ete':'#c44e52','Automne':'#dd8452'}
fig,axes=plt.subplots(2,2,figsize=(13,10))
fig.subplots_adjust(hspace=0.38,wspace=0.35,left=0.1,right=0.97,top=0.90,bottom=0.10)

ax=axes[0,0]
data_box=[df_res[df_res['season']==s]['obs'].dropna().values for s in seasons]
bp=ax.boxplot(data_box,patch_artist=True,notch=False,widths=0.55,
              medianprops=dict(color='black',lw=1.5),
              whiskerprops=dict(lw=0.8),capprops=dict(lw=0.8),flierprops=dict(marker='.',ms=2,alpha=0.3))
for patch,s in zip(bp['boxes'],seasons): patch.set_facecolor(pal[s]); patch.set_alpha(0.75)
ax.set_xticklabels([season_en[s] for s in seasons],fontsize=9)
ax.set_ylabel('Observed SM₁ [Vol%]',fontsize=10)
ax.set_title('(a) SM₁ distribution by season',fontsize=11,fontweight='bold')

ax=axes[0,1]
rmse_rf=[np.sqrt(mean_squared_error(df_res[df_res['season']==s]['obs'],df_res[df_res['season']==s]['rf'])) for s in seasons]
rmse_xgb=[np.sqrt(mean_squared_error(df_res[df_res['season']==s]['obs'],df_res[df_res['season']==s]['xgb'])) for s in seasons]
x_=np.arange(4)
ax.bar(x_-0.2,rmse_rf,0.38,color=C['rf'],alpha=0.85,label='Random Forest')
ax.bar(x_+0.2,rmse_xgb,0.38,color=C['xgb'],alpha=0.85,label='XGBoost')
ax.set_xticks(x_); ax.set_xticklabels([season_en[s] for s in seasons],fontsize=9)
ax.set_ylabel('RMSE [Vol%]',fontsize=10); ax.set_title('(b) RMSE by season',fontsize=11,fontweight='bold')
ax.legend(fontsize=9)

ax=axes[1,0]
for s in seasons:
    d_=df_res[df_res['season']==s]
    ax.scatter(d_['T_mean'],d_['err_rf'],c=pal[s],s=8,alpha=0.4,label=season_en[s])
ax.axhline(0,color='k',lw=0.8,ls='--')
z_=np.polyfit(df_res['T_mean'].dropna(),df_res.loc[df_res['T_mean'].notna(),'err_rf'],1)
xs_=np.linspace(df_res['T_mean'].min(),df_res['T_mean'].max(),100)
ax.plot(xs_,np.poly1d(z_)(xs_),'k-',lw=1.5,alpha=0.7,label='Trend')
ax.set_xlabel('Air temperature [°C]',fontsize=10); ax.set_ylabel('RF residuals [Vol%]',fontsize=10)
ax.set_title('(c) Residuals vs. air temperature',fontsize=11,fontweight='bold')
ax.legend(fontsize=8,markerscale=2,ncol=2)

ax=axes[1,1]
for s in seasons:
    d_=df_res[(df_res['season']==s)&(df_res['Rain']<30)]
    ax.scatter(d_['Rain'],np.abs(d_['err_rf']),c=pal[s],s=8,alpha=0.4,label=season_en[s])
d_nr=df_res[df_res['Rain']<30]
z2=np.polyfit(d_nr['Rain'],np.abs(d_nr['err_rf']),1)
xs2=np.linspace(0,30,100); ax.plot(xs2,np.poly1d(z2)(xs2),'k-',lw=1.5,alpha=0.7)
ax.set_xlabel('Daily precipitation [mm d⁻¹]',fontsize=10)
ax.set_ylabel('|RF residuals| [Vol%]',fontsize=10)
ax.set_title('(d) Absolute residuals vs. precipitation',fontsize=11,fontweight='bold')
ax.legend(fontsize=8,markerscale=2,ncol=2)

for ax in axes.flat: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Fig. 5. Seasonal analysis of model performance and residuals\nSoil moisture dynamics in a semi-arid lysimeter (2023–2026)',fontsize=9.5,style='italic')
plt.savefig('pub_fig5_seasonal.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig5_seasonal.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Tableau métriques publication
# ══════════════════════════════════════════════════════════════════════════════
print("Fig. 6 — Tableau métriques...", flush=True)
fig,ax=plt.subplots(figsize=(12,5)); fig.patch.set_facecolor('white'); ax.axis('off')
fig.subplots_adjust(left=0.04,right=0.96,top=0.82,bottom=0.04)
cols_h=['Model','RMSE\n[Vol%]','MAE\n[Vol%]','R²\n[-]','NSE\n[-]','PBIAS\n[%]','Assessment']
rows_data=[
    ['Ridge Regression ★','1.921','1.055','0.850','0.850','+8.4%','Best NSE/RMSE — slight +bias'],
    ['Random Forest',     '2.447','1.279','0.757','0.757','+1.7%','Satisfactory — low bias'],
    ['XGBoost',           '2.491','1.491','0.749','0.749','+2.6%','Satisfactory — low bias'],
    ['Stacking (RF+XGB)', '2.364','1.608','0.776','0.776','+1.1%','Best PBIAS — bias corrected ✓'],
]
rmse_vals=[float(r[1]) for r in rows_data]
cell_clrs=[]
for i,row in enumerate(rows_data):
    bg=(0.88,0.96,0.88) if float(row[1])==min(rmse_vals) else (0.97,0.97,0.97)
    cell_clrs.append([bg]*len(cols_h))
tbl=ax.table(cellText=rows_data,colLabels=cols_h,cellLoc='center',loc='center',
             cellColours=cell_clrs,bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
for (r,c),cell in tbl.get_celld().items():
    cell.set_height(0.2)
    if r==0: cell.set_facecolor('#1d3557'); cell.set_text_props(color='white',fontsize=10.5,fontweight='bold')
    if c==0 and r>0: cell.set_text_props(fontweight='bold')
ax.set_title('Table 1. Performance metrics of the ML models\n'
             'Temporal cross-validation (5-fold walk-forward, gap=30 days, n=1018 daily obs.)',
             fontsize=10,style='italic',pad=14)
plt.savefig('pub_fig6_table.png',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(); print("  ✓ pub_fig6_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MOIS 6 TERMINÉ — 6 figures publication-ready")
print("=" * 60)
print("\nFigures générées (300 dpi) :")
for f in ['pub_fig1_overview.png','pub_fig2_scatter.png','pub_fig3_timeseries.png',
          'pub_fig4_shap.png','pub_fig5_seasonal.png','pub_fig6_table.png']:
    print(f"  ✓ {f}")
print("\nMétriques finales :")
for k,v in metrics.items():
    print(f"  {k:<18} RMSE={v['RMSE']:.3f}  NSE={v['NSE']:.3f}  PBIAS={v['PBIAS']:+.1f}%")
print("\nTop 5 features SHAP XGBoost :")
for f,v in shap_imp_xgb.head(5).items():
    print(f"  {nice(f):<22} = {v:.4f} Vol%")
print("\nJournal cible : Agricultural Water Management (Q1, IF ~7.5)")
print("Action suivante : Compléter l'article (noms + affiliation) et soumettre")

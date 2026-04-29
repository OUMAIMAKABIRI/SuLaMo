"""
MOIS 4 — Validation, SHAP & Incertitude
(voir mois3_models_ML.py pour le dataset et les modèles)
Dépendances : pip install shap scipy
"""
# ─── Reprend exactement après mois3_models_ML.py ─────────────────────────────
# Assurez-vous que rf, xm, X, y, features, dates, fold_tr, fold_te, yobs, d_te
# sont disponibles dans votre session (ou relancez mois3 avant ce script).

import shap, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. SHAP — TreeExplainer (RF + XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

print("Calcul SHAP Random Forest...", flush=True)
explainer_rf  = shap.TreeExplainer(rf)
shap_values_rf= explainer_rf.shap_values(X)
shap_imp_rf   = pd.Series(shap_values_rf, index=features).abs().mean().sort_values(ascending=False)
# → dataframe pour analyses ultérieures
shap_df_rf = pd.DataFrame(np.abs(shap_values_rf), columns=features, index=dates)

print("Calcul SHAP XGBoost...", flush=True)
explainer_xgb  = shap.TreeExplainer(xm)
shap_values_xgb= explainer_xgb.shap_values(X)
shap_imp_xgb   = pd.Series(shap_values_xgb, index=features).abs().mean().sort_values(ascending=False)
shap_df_xgb    = pd.DataFrame(shap_values_xgb, columns=features, index=dates)

print("Top 10 SHAP XGBoost :")
print(shap_imp_xgb.head(10).round(4).to_string())

# ── FIGURE 1 : Importance globale SHAP bar ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Analyse SHAP — Importance globale (Mean |SHAP value|)',
             fontsize=13, fontweight='bold')
for ax, imp, title in zip(axes, [shap_imp_rf.head(20), shap_imp_xgb.head(20)],
                           ['Random Forest', 'XGBoost']):
    ypos = range(len(imp)-1, -1, -1)
    ax.barh(list(ypos), imp.values, alpha=0.8, edgecolor='none')
    ax.set_yticks(list(ypos)); ax.set_yticklabels(imp.index, fontsize=9)
    ax.set_xlabel('Mean |SHAP| [Vol%]', fontsize=10); ax.set_title(title, fontsize=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('m4_fig1_shap_global.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ── FIGURE 2 : SHAP Beeswarm (XGB top 15) ───────────────────────────────────
top15 = shap_imp_xgb.head(15).index.tolist()
top15_idx = [features.index(f) for f in top15]
sv_top = shap_values_xgb[:, top15_idx]; X_top = X[:, top15_idx]
fig, ax = plt.subplots(figsize=(12, 9)); fig.patch.set_facecolor('white')
np.random.seed(42)
for row_i, feat in enumerate(top15):
    y_pos = len(top15) - 1 - row_i
    sv = sv_top[:, row_i]; xv = X_top[:, row_i]
    xv_norm = np.clip((xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-8),0,1)
    jitter = np.random.uniform(-0.35, 0.35, len(sv))
    ax.scatter(sv, y_pos+jitter, c=xv_norm, cmap='RdBu_r', s=6, alpha=0.5, vmin=0, vmax=1)
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_yticks(range(len(top15))); ax.set_yticklabels(list(reversed(top15)), fontsize=10)
ax.set_xlabel('SHAP value [Vol%]', fontsize=11)
ax.set_title('SHAP Beeswarm — XGBoost (top 15)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.25)
plt.tight_layout(); plt.savefig('m4_fig2_shap_beeswarm.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 2. DÉCOMPOSITION SAISONNIÈRE DES ERREURS
# ══════════════════════════════════════════════════════════════════════════════

yp_rf_all  = rf.predict(X); yp_xgb_all = xm.predict(X)
df_err = pd.DataFrame({'obs':y,'err_rf':yp_rf_all-y,'err_xgb':yp_xgb_all-y,
                        'abs_rf':np.abs(yp_rf_all-y),'abs_xgb':np.abs(yp_xgb_all-y),
                        'season':day_ml['season_label'].values,
                        'T_mean':day_ml['T_mean'].values,'Rain':day_ml['Rain'].values},
                       index=dates)

def nse(o,s): return 1-np.sum((o-s)**2)/np.sum((o-o.mean())**2)

season_order = ['Hiver','Printemps','Ete','Automne']
pal = {'Hiver':'#4c72b0','Printemps':'#55a868','Ete':'#c44e52','Automne':'#dd8452'}

# Afficher NSE par saison
print("\nNSE par saison (RF / XGB) :")
for s in season_order:
    mask_s = df_err['season'] == s
    yt_=y[mask_s.values]; yp_rf_=yp_rf_all[mask_s.values]; yp_xgb_=yp_xgb_all[mask_s.values]
    print(f"  {s:<12} RF={nse(yt_,yp_rf_):.3f}  XGB={nse(yt_,yp_xgb_):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.patch.set_facecolor('white')
fig.suptitle('Décomposition saisonnière des erreurs', fontsize=13, fontweight='bold')

# RMSE par saison
ax = axes[0,0]
for k, (col, lbl) in enumerate([('abs_rf','RF'),('abs_xgb','XGB')]):
    rmse_s = df_err.groupby('season')[col].apply(lambda x: np.sqrt((x**2).mean()))
    x_pos = np.arange(4) + k*0.35 - 0.18
    ax.bar(x_pos, [rmse_s.get(s,0) for s in season_order], width=0.33,
           color=['#2ca02c','#e07b39'][k], label=lbl, alpha=0.85)
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(season_order)
ax.set_ylabel('RMSE [Vol%]'); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title('RMSE par saison'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Distribution résidus par saison
ax = axes[0,1]
for s in season_order:
    ax.hist(df_err[df_err['season']==s]['err_rf'], bins=30, alpha=0.6, color=pal[s], label=s, density=True)
ax.axvline(0,color='k',lw=1,ls='--'); ax.set_xlabel('Résidu RF [Vol%]')
ax.set_title('Distribution résidus par saison'); ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# NSE par saison
ax = axes[0,2]
for k,(yp_all,lbl) in enumerate([(yp_rf_all,'RF'),(yp_xgb_all,'XGB')]):
    nse_s=[nse(y[df_err['season'].values==s],yp_all[df_err['season'].values==s]) for s in season_order]
    x_pos=np.arange(4)+k*0.35-0.18
    ax.bar(x_pos,nse_s,width=0.33,color=['#2ca02c','#e07b39'][k],label=lbl,alpha=0.85)
ax.axhline(0.6,color='red',ls='--',lw=1,label='Seuil NSE=0.6')
ax.set_xticks([0,1,2,3]); ax.set_xticklabels(season_order)
ax.set_ylabel('NSE'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
ax.set_title('NSE par saison'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Erreur vs T° air
ax = axes[1,0]
for s in season_order:
    d_=df_err[df_err['season']==s]
    ax.scatter(d_['T_mean'],d_['abs_rf'],c=pal[s],s=8,alpha=0.4,label=s)
ax.set_xlabel('T° air [°C]'); ax.set_ylabel('|Erreur RF| [Vol%]')
ax.set_title('Erreur absolue vs T° air'); ax.legend(fontsize=8,markerscale=2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(alpha=0.3)

# RMSE vs pluie
ax = axes[1,1]
df_err['rain_cat']=pd.cut(df_err['Rain'],[0,0.5,2,5,10,50],labels=['0','0-2','2-5','5-10','>10'])
for k,(col,lbl) in enumerate([('abs_rf','RF'),('abs_xgb','XGB')]):
    rmse_r=df_err.groupby('rain_cat',observed=True)[col].apply(lambda x:np.sqrt((x**2).mean()))
    ax.bar(np.arange(5)+k*0.35-0.18,rmse_r.values,width=0.33,color=['#2ca02c','#e07b39'][k],label=lbl,alpha=0.85)
ax.set_xticks([0,1,2,3,4]); ax.set_xticklabels(['0','0-2','2-5','5-10','>10'])
ax.set_xlabel('Pluie [mm/j]'); ax.set_ylabel('RMSE [Vol%]'); ax.legend()
ax.set_title('RMSE vs intensité pluie'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# RMSE rolling temporel
ax = axes[1,2]
for col,c,lbl in [('abs_rf','#2ca02c','RF'),('abs_xgb','#e07b39','XGB')]:
    roll=np.sqrt(df_err[col].pow(2).rolling(30,min_periods=15).mean())
    ax.fill_between(df_err.index,roll,alpha=0.2,color=c)
    ax.plot(df_err.index,roll,lw=1.5,color=c,label=lbl)
ax.set_ylabel('RMSE rolling 30j [Vol%]'); ax.legend()
ax.set_title('RMSE temporel (stabilité)'); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(); plt.savefig('m4_fig4_seasonal_errors.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTIFICATION DE L'INCERTITUDE (arbres RF)
# ══════════════════════════════════════════════════════════════════════════════

rf_tree_preds = np.array([t.predict(X[fold_te]) for t in rf.estimators_])
rf_p5  = np.percentile(rf_tree_preds, 5, axis=0)
rf_p95 = np.percentile(rf_tree_preds, 95, axis=0)
rf_p25 = np.percentile(rf_tree_preds, 25, axis=0)
rf_p75 = np.percentile(rf_tree_preds, 75, axis=0)
rf_std = rf_tree_preds.std(axis=0)
cov_90 = np.mean((yobs>=rf_p5)&(yobs<=rf_p95))*100
cov_50 = np.mean((yobs>=rf_p25)&(yobs<=rf_p75))*100
print(f"\nIC 90% couverture: {cov_90:.1f}% | IC 50%: {cov_50:.1f}%")

# Bootstrap
np.random.seed(42)
boot_preds=[]; n_boot=50
for b in range(n_boot):
    idx=np.random.choice(len(fold_tr),len(fold_tr),replace=True)
    rf_b=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,max_features=0.5,n_jobs=-1,random_state=b)
    rf_b.fit(X[fold_tr[idx]],y[fold_tr[idx]]); boot_preds.append(rf_b.predict(X[fold_te]))
boot_preds=np.array(boot_preds)
boot_p5=np.percentile(boot_preds,5,axis=0); boot_p95=np.percentile(boot_preds,95,axis=0)
cov_boot=np.mean((yobs>=boot_p5)&(yobs<=boot_p95))*100

fig,axes=plt.subplots(3,1,figsize=(16,15),sharex=True); fig.patch.set_facecolor('white')
fig.suptitle('Quantification de l\'incertitude des prédictions\n'
             'RF (arbres individuels) + Bootstrap',fontsize=13,fontweight='bold')

ax=axes[0]
ax.fill_between(d_te,rf_p5,rf_p95,alpha=0.2,color='#2ca02c',label='IC 90%')
ax.fill_between(d_te,rf_p25,rf_p75,alpha=0.35,color='#2ca02c',label='IC 50%')
ax.plot(d_te,rf_tree_preds.mean(axis=0),lw=1.5,color='#2ca02c',label='RF moyen')
ax.plot(d_te,yobs,lw=1.8,color='black',label='Observé',zorder=5)
ax.set_ylabel('SM₁ [Vol%]'); ax.legend(fontsize=9)
ax.set_title(f'IC RF (arbres) — Couverture 90%:{cov_90:.1f}% | 50%:{cov_50:.1f}%',fontsize=10)
ax.grid(alpha=0.3)

ax=axes[1]
ax.fill_between(d_te,boot_p5,boot_p95,alpha=0.25,color='#9467bd',label=f'IC Bootstrap 90%')
ax.plot(d_te,boot_preds.mean(axis=0),lw=1.5,color='#9467bd',label='Bootstrap moyen')
ax.plot(d_te,yobs,lw=1.8,color='black',label='Observé',zorder=5)
ax.set_ylabel('SM₁ [Vol%]'); ax.legend(fontsize=9)
ax.set_title(f'IC Bootstrap — Couverture:{cov_boot:.1f}%',fontsize=10); ax.grid(alpha=0.3)

ax=axes[2]
width_t=rf_p95-rf_p5
ax.fill_between(d_te,width_t,alpha=0.4,color='#ff7f0e')
ax.plot(d_te,width_t,lw=1.2,color='#ff7f0e',label=f'Largeur IC 90% (moy={width_t.mean():.2f})')
ax.axhline(width_t.mean(),color='k',ls='--',lw=1)
ax.set_ylabel('Largeur IC [Vol%]'); ax.legend(fontsize=9)
ax.set_title('Évolution de l\'incertitude dans le temps'); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
for ax in axes: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(h_pad=0.8)
plt.savefig('m4_fig6_uncertainty.png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

# Diagramme de fiabilité
fig,axes=plt.subplots(1,3,figsize=(15,5)); fig.patch.set_facecolor('white')
nominal=np.arange(10,100,10)
emp_cov=[np.mean((yobs>=np.percentile(rf_tree_preds,(100-l)/2,axis=0))&
                  (yobs<=np.percentile(rf_tree_preds,100-(100-l)/2,axis=0)))*100 for l in nominal]
axes[0].plot([0,100],[0,100],'k--',lw=1)
axes[0].plot(nominal,emp_cov,'o-',color='#2ca02c',lw=2,markersize=6)
axes[0].set_xlabel('Couverture nominale (%)'); axes[0].set_ylabel('Couverture empirique (%)')
axes[0].set_title('Diagramme de fiabilité'); axes[0].grid(alpha=0.3)
axes[1].scatter(rf_std,np.abs(yobs-rf_tree_preds.mean(axis=0)),alpha=0.35,s=15,color='#2ca02c')
r_val=np.corrcoef(rf_std,np.abs(yobs-rf_tree_preds.mean(axis=0)))[0,1]
axes[1].set_xlabel('Std RF (incertitude)'); axes[1].set_ylabel('|Erreur| observée')
axes[1].set_title(f'Incertitude vs erreur réelle (r={r_val:.3f})'); axes[1].grid(alpha=0.3)
residuals=yobs-rf_tree_preds.mean(axis=0)
(osm,osr),(slope,intercept,r)=sp_stats.probplot(residuals,dist='norm')
axes[2].scatter(osm,osr,s=8,alpha=0.5,color='#2ca02c')
xs_=np.array([min(osm),max(osm)]); axes[2].plot(xs_,slope*xs_+intercept,'r-',lw=1.5)
axes[2].set_xlabel('Quantiles théoriques'); axes[2].set_ylabel('Quantiles résidus')
axes[2].set_title(f'QQ-plot résidus (R²={r**2:.3f})'); axes[2].grid(alpha=0.3)
for ax in axes: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig('m4_fig7_calibration.png',dpi=150,bbox_inches='tight',facecolor='white')
plt.close()

print("\nMOIS 4 TERMINÉ — 7 figures générées")

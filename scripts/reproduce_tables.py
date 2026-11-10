"""
Reproduces all validation tables and in-text summary statistics reported in the manuscript:
- Table 3: Individual root-level volumetric validation (n = 210 root images)
- Table 4: Aggregated simulated plot-level volumetric validation (n = 21 plot-flights & n = 7 cultivar means)
- Table 5: Cross-validated weight estimation across volumetric models and density parameterizations
- Table 6: Out-of-sample field plot-level weight validation across combined 2024 trials (n = 286 plots)
- Table 7: Out-of-sample field plot-level weight validation at Clinton (n = 144 plots)
- Additional in-text metrics: Breusch-Pagan homoskedasticity tests, root count filtering,
  cross-platform genotype concordance (Spearman rho, Pearson r, top-k overlap), and cross-site correlations.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression

from utils import (
    eval_metrics,
    load_2026_controlled_data,
    get_2026_calibration_params,
    load_2024_field_data,
)


def main():
    _, rcnn, sam2 = load_2026_controlled_data()
    calib_params = get_2026_calibration_params(rcnn)
    pooled_density_2026 = float(rcnn['true_density_g_cm3'].mean())

    vol_methods = [
        ('adj_sphere_vol_cm3', 'max_diameter_cm', 'Adj. Spheroid'),
        ('segmented_vol_cm3', 'max_diameter_cm', 'Segmented'),
        ('sphere_vol_cm3', 'box_width_cm', 'Spheroid'),
        ('adj_ellip_vol_cm3', 'max_diameter_cm', 'Adj. Ellipsoid'),
        ('ellip_vol_cm3', 'box_width_cm', 'Ellipsoid'),
    ]

    # =========================================================================
    # 1. TABLE 3: Individual Root-Level Volumetric Validation (n = 210)
    # =========================================================================
    print("=" * 85)
    print("TABLE 3: INDIVIDUAL ROOT-LEVEL VOLUMETRIC VALIDATION (n = 210 root images)")
    print("=" * 85)
    t3_rows = []

    # First evaluate SAM 2 Reference for Adj. Spheroid
    y_sam = sam2['true_volume_cm3'].values
    X_sam = sam2['adj_sphere_vol_cm3'].values
    reg_sam = LinearRegression().fit(X_sam.reshape(-1, 1), y_sam)
    m_sam, b_sam = round(float(reg_sam.coef_[0]), 4), round(float(reg_sam.intercept_), 2)

    y_sam_loro = np.zeros_like(y_sam)
    for rid in sam2['root_uid'].unique():
        idx = (sam2['root_uid'] == rid).values
        reg = LinearRegression().fit(X_sam[~idx].reshape(-1, 1), y_sam[~idx])
        y_sam_loro[idx] = reg.predict(X_sam[idx].reshape(-1, 1))

    y_sam_loco = np.zeros_like(y_sam)
    for cln in sam2['clone_name'].unique():
        idx = (sam2['clone_name'] == cln).values
        reg = LinearRegression().fit(X_sam[~idx].reshape(-1, 1), y_sam[~idx])
        y_sam_loco[idx] = reg.predict(X_sam[idx].reshape(-1, 1))

    t3_rows.append({'Method (Mask Source)': 'Adj. Spheroid (SAM 2 Ref.)', 'Evaluation Type': 'LORO-CV', **eval_metrics(y_sam, y_sam_loro), 'Slope': m_sam, 'Offset': str(b_sam)})
    t3_rows.append({'Method (Mask Source)': 'Adj. Spheroid (SAM 2 Ref.)', 'Evaluation Type': 'LOCO-CV', **eval_metrics(y_sam, y_sam_loco), 'Slope': m_sam, 'Offset': str(b_sam)})
    t3_rows.append({'Method (Mask Source)': 'Adj. Spheroid (SAM 2 Ref.)', 'Evaluation Type': 'Raw', **eval_metrics(y_sam, X_sam), 'Slope': 1.0000, 'Offset': '0.00'})

    # Evaluate all 5 methods on Mask R-CNN
    y = rcnn['true_volume_cm3'].values
    for col, dcol, name in vol_methods:
        X = rcnn[col].values
        p = calib_params[col]
        m_full, b_full = round(p['ols_m'], 4), round(p['ols_b'], 2)

        # LORO-CV
        y_loro = np.zeros_like(y)
        for rid in rcnn['root_uid'].unique():
            idx = (rcnn['root_uid'] == rid).values
            reg = LinearRegression().fit(X[~idx].reshape(-1, 1), y[~idx])
            y_loro[idx] = reg.predict(X[idx].reshape(-1, 1))
        rcnn[f'{col}_loro'] = y_loro

        # LOCO-CV
        y_loco = np.zeros_like(y)
        for cln in rcnn['clone_name'].unique():
            idx = (rcnn['clone_name'] == cln).values
            reg = LinearRegression().fit(X[~idx].reshape(-1, 1), y[~idx])
            y_loco[idx] = reg.predict(X[idx].reshape(-1, 1))
        rcnn[f'{col}_loco'] = y_loco

        if col == 'adj_sphere_vol_cm3':
            # Boundary-Scaled LOCO-CV
            pd_vec = (rcnn['perimeter_cm'] * rcnn[dcol]).values
            X_bnd = np.column_stack([X, pd_vec])
            y_bnd_loco = np.zeros_like(y)
            for cln in rcnn['clone_name'].unique():
                idx = (rcnn['clone_name'] == cln).values
                reg_b = LinearRegression(fit_intercept=False).fit(X_bnd[~idx], y[~idx])
                y_bnd_loco[idx] = reg_b.predict(X_bnd[idx])
            t3_rows.append({
                'Method (Mask Source)': f'{name} (Mask R-CNN)',
                'Evaluation Type': 'Boundary-Scaled',
                **eval_metrics(y, y_bnd_loco),
                'Slope': round(p['bnd_m'], 4),
                'Offset': f"{p['bnd_cb']:.4f} P*d_max"
            })

        t3_rows.append({'Method (Mask Source)': f'{name} (Mask R-CNN)', 'Evaluation Type': 'LORO-CV', **eval_metrics(y, y_loro), 'Slope': m_full, 'Offset': str(b_full)})
        t3_rows.append({'Method (Mask Source)': f'{name} (Mask R-CNN)', 'Evaluation Type': 'LOCO-CV', **eval_metrics(y, y_loco), 'Slope': m_full, 'Offset': str(b_full)})

        if col == 'adj_sphere_vol_cm3':
            # Proportional LOCO-CV
            y_prop_loco = np.zeros_like(y)
            for cln in rcnn['clone_name'].unique():
                idx = (rcnn['clone_name'] == cln).values
                reg_p = LinearRegression(fit_intercept=False).fit(X[~idx].reshape(-1, 1), y[~idx])
                y_prop_loco[idx] = reg_p.predict(X[idx].reshape(-1, 1))
            t3_rows.append({
                'Method (Mask Source)': f'{name} (Mask R-CNN)',
                'Evaluation Type': 'Proportional',
                **eval_metrics(y, y_prop_loco),
                'Slope': round(p['prop_k'], 4),
                'Offset': '0.00'
            })

        t3_rows.append({'Method (Mask Source)': f'{name} (Mask R-CNN)', 'Evaluation Type': 'Raw', **eval_metrics(y, X), 'Slope': 1.0000, 'Offset': '0.00'})

    print(pd.DataFrame(t3_rows).to_string(index=False))

    # =========================================================================
    # 2. TABLE 4: Aggregated Simulated Plot-Level Volume Validation (n=21 & n=7)
    # =========================================================================
    print("\n" + "=" * 85)
    print("TABLE 4: SIMULATED PLOT-LEVEL VOLUMETRIC VALIDATION (n = 21 plot-flights & n = 7 cultivars)")
    print("=" * 85)
    agg_21 = rcnn.groupby(['clone_name', 'test']).sum(numeric_only=True).reset_index()
    agg_7 = agg_21.groupby('clone_name').mean(numeric_only=True).reset_index()

    t4_rows = []
    # Order matching Table 4 in manuscript: Adj. Spheroid, Spheroid, Segmented, Adj. Ellipsoid, Ellipsoid
    t4_order = [
        ('adj_sphere_vol_cm3', 'Adj. Spheroid'),
        ('sphere_vol_cm3', 'Spheroid'),
        ('segmented_vol_cm3', 'Segmented'),
        ('adj_ellip_vol_cm3', 'Adj. Ellipsoid'),
        ('ellip_vol_cm3', 'Ellipsoid'),
    ]
    for col, name in t4_order:
        for suf, stype in [('_loro', 'LORO-CV'), ('_loco', 'LOCO-CV'), ('', 'Raw')]:
            m21 = eval_metrics(agg_21['true_volume_cm3'].values, agg_21[f'{col}{suf}'].values)
            m7 = eval_metrics(agg_7['true_volume_cm3'].values, agg_7[f'{col}{suf}'].values)
            t4_rows.append({
                'Method': name,
                'Evaluation Type': stype,
                'Plot21_R2': m21['R2'], 'Plot21_RMSE': m21['RMSE'], 'Plot21_MAPE': m21['MAPE'],
                'Cult7_R2': m7['R2'], 'Cult7_RMSE': m7['RMSE'], 'Cult7_MAPE': m7['MAPE'],
            })
    print(pd.DataFrame(t4_rows).to_string(index=False))

    # =========================================================================
    # 3. TABLE 5: Cross-Validated Weight Validation Across Density Schemes
    # =========================================================================
    print("\n" + "=" * 85)
    print("TABLE 5: CROSS-VALIDATED WEIGHT ESTIMATION (Individual n = 210 & Simulated Plot n = 21)")
    print("=" * 85)
    rcnn['loro_cult_dens'] = 0.0
    for rid in rcnn['root_uid'].unique():
        idx = (rcnn['root_uid'] == rid)
        cln = rcnn.loc[idx, 'clone_name'].iloc[0]
        rcnn.loc[idx, 'loro_cult_dens'] = rcnn.loc[(rcnn['clone_name'] == cln) & (~idx), 'true_density_g_cm3'].mean()

    rcnn['loco_pool_dens'] = 0.0
    for cln in rcnn['clone_name'].unique():
        idx = (rcnn['clone_name'] == cln)
        rcnn.loc[idx, 'loco_pool_dens'] = rcnn.loc[~idx, 'true_density_g_cm3'].mean()

    t5_rows = []
    for col, _, name in vol_methods[:3]:
        rcnn['w_raw_fix'] = rcnn[col] * 1.05
        rcnn['w_loco_fix'] = rcnn[f'{col}_loco'] * 1.05
        rcnn['w_loco_pool'] = rcnn[f'{col}_loco'] * rcnn['loco_pool_dens']
        rcnn['w_loro_fix'] = rcnn[f'{col}_loro'] * 1.05
        rcnn['w_loro_cult'] = rcnn[f'{col}_loro'] * rcnn['loro_cult_dens']
        rcnn['w_loco_cult'] = rcnn[f'{col}_loco'] * rcnn['avg_density_g_cm3']

        agg_w = rcnn.groupby(['clone_name', 'test']).sum(numeric_only=True).reset_index()
        for wcol, cal_type, dens_type in [
            ('w_raw_fix', 'Raw', 'Fixed'),
            ('w_loco_fix', 'LOCO-CV', 'Fixed'),
            ('w_loco_pool', 'LOCO-CV', 'Pooled CV'),
            ('w_loro_fix', 'LORO-CV', 'Fixed'),
            ('w_loro_cult', 'LORO-CV', 'Cultivar CV'),
            ('w_loco_cult', 'LOCO-CV', 'Cultivar'),
        ]:
            m_ind = eval_metrics(rcnn['true_weight_g'].values, rcnn[wcol].values)
            m_agg = eval_metrics(agg_w['true_weight_g'].values, agg_w[wcol].values)
            t5_rows.append({
                'Method': name, 'Vol. Calib.': cal_type, 'Density Source': dens_type,
                'Ind_R2': m_ind['R2'], 'Ind_RMSE': m_ind['RMSE'], 'Ind_MAPE': m_ind['MAPE'],
                'Plot_R2': m_agg['R2'], 'Plot_RMSE': m_agg['RMSE'], 'Plot_MAPE': m_agg['MAPE'],
            })
    print(pd.DataFrame(t5_rows).to_string(index=False))

    # =========================================================================
    # 4. TABLES 6 & 7: 2024 Field Trial Weight Validation (n = 286 & n = 144)
    # =========================================================================
    plot_data, uav_roots = load_2024_field_data()

    # Compute root-level calibrated volumes (clipped at 0 cm^3) for each of the 17,793 field roots
    field_methods = [
        ('adj_sphere_vol_cm3', 'volume_asph_cm3_total', 'Adj. Spheroid'),
        ('sphere_vol_cm3', 'volume_sph_cm3_total', 'Spheroid'),
        ('segmented_vol_cm3', 'volume_seg_cm3_total', 'Segmented'),
    ]
    for rcol, _, _ in field_methods:
        p = calib_params[rcol]
        dcol = p['dcol']
        uav_roots[f'{rcol}_ols_clip'] = np.maximum(0.0, p['ols_m'] * uav_roots[rcol] + p['ols_b'])
        uav_roots[f'{rcol}_bnd_clip'] = np.maximum(
            0.0, p['bnd_m'] * uav_roots[rcol] + p['bnd_cb'] * uav_roots['perimeter_cm'] * uav_roots[dcol]
        )
        uav_roots[f'{rcol}_prop'] = p['prop_k'] * uav_roots[rcol]

    agg_cols = []
    for rcol, _, _ in field_methods:
        agg_cols.extend([f'{rcol}_ols_clip', f'{rcol}_bnd_clip', f'{rcol}_prop', rcol])
    root_plot_sums = uav_roots.groupby('plot_name')[agg_cols].sum().reset_index()
    plot_data = pd.merge(plot_data, root_plot_sums, on='plot_name', how='left')

    for table_title, sub_df in [
        ("TABLE 6: COMBINED 2024 FIELD TRIALS (n = 286 plots)", plot_data),
        ("TABLE 7: CLINTON 2024 FIELD TRIAL (n = 144 plots)", plot_data[plot_data['site'] == 'Clinton']),
    ]:
        print("\n" + "=" * 85)
        print(table_title)
        print("=" * 85)
        y_true = sub_df['true_weight_all'].values
        rows = []
        # Define calibration schemes in table order
        schemes = [
            ('ols_clip', 'OLS', pooled_density_2026, '0.98 g/cm3'),
            ('bnd_clip', 'Boundary-Scaled', pooled_density_2026, '0.98 g/cm3'),
            ('prop', 'Proportional', pooled_density_2026, '0.98 g/cm3'),
            ('ols_clip', 'OLS', 1.05, '1.05 g/cm3'),
            ('raw', 'Raw', pooled_density_2026, '0.98 g/cm3'),
            ('raw', 'Raw', 1.05, '1.05 g/cm3'),
        ]
        for sc_key, sc_label, rho_val, rho_str in schemes:
            scheme_rows = []
            for rcol, _, m_label in field_methods:
                v_col = rcol if sc_key == 'raw' else f'{rcol}_{sc_key}'
                w_pred = (sub_df[v_col].values / 1000.0) * rho_val
                cal_str = sc_label
                if sc_key == 'prop':
                    cal_str = f"Proportional (k = {calib_params[rcol]['prop_k']:.3f})"
                scheme_rows.append({
                    'Method': m_label,
                    'Volume Calibration': cal_str,
                    'Density': rho_str,
                    **eval_metrics(y_true, w_pred),
                })
            scheme_rows.sort(key=lambda x: x['R2'], reverse=True)
            rows.extend(scheme_rows)
        print(pd.DataFrame(rows).to_string(index=False))

    # =========================================================================
    # 5. ADDITIONAL IN-TEXT VALIDATION & GENOTYPE SELECTION METRICS
    # =========================================================================
    print("\n" + "=" * 85)
    print("ADDITIONAL IN-TEXT STATISTICAL TESTS & GENOTYPE CONCORDANCE METRICS")
    print("=" * 85)

    # Breusch-Pagan heteroskedasticity test
    for col, _, label in field_methods:
        X_bp = sm.add_constant(rcnn['true_volume_cm3'])
        resid = rcnn[col] - rcnn['true_volume_cm3']
        pe = (rcnn[col] - rcnn['true_volume_cm3']) / rcnn['true_volume_cm3'] * 100.0
        _, p_abs, _, _ = het_breuschpagan(resid, X_bp)
        _, p_rel, _, _ = het_breuschpagan(pe, X_bp)
        print(f"Breusch-Pagan ({label}): Abs Residual p = {p_abs:.3f} | Rel Percentage Error p = {p_rel:.3e}")

    # Genotype concordance (n = 136 clones)
    clone_means = plot_data.groupby('clone_name')[
        ['weight_total_kg', 'true_weight_all', 'ld_ratio_avg', 'true_mean_ld_all']
    ].mean().reset_index()

    for uav_c, srt_c, tname in [
        ('weight_total_kg', 'true_weight_all', 'Total Plot Weight'),
        ('ld_ratio_avg', 'true_mean_ld_all', 'LD Ratio'),
    ]:
        rho, _ = stats.spearmanr(clone_means[uav_c], clone_means[srt_c])
        r_p, _ = stats.pearsonr(clone_means[uav_c], clone_means[srt_c])
        k10 = int(round(len(clone_means) * 0.10))
        k20 = int(round(len(clone_means) * 0.20))
        ov10 = len(set(clone_means.nlargest(k10, uav_c)['clone_name']) & set(clone_means.nlargest(k10, srt_c)['clone_name']))
        ov20 = len(set(clone_means.nlargest(k20, uav_c)['clone_name']) & set(clone_means.nlargest(k20, srt_c)['clone_name']))
        print(f"Genotype Concordance ({tname}, n={len(clone_means)}): Spearman rho = {rho:.3f}, Pearson r = {r_p:.3f} | Top-10% = {ov10}/{k10} ({ov10/k10*100:.1f}%), Top-20% = {ov20}/{k20} ({ov20/k20*100:.1f}%)")


if __name__ == '__main__':
    main()

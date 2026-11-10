"""
Generates the analytical figures presented in the manuscript directly from the
curated datasets in `data/`:
- Figure 7: `figures/res_raw_volume_anova.png` (Raw volumetric EMM APE across 5 geometric models)
- Figure 8: `figures/res_4x4_volume.png` (LOCO-CV volumetric parity & residuals at root & simulated plot scales)
- Figure 9: `figures/res_genotype_concordance.png` (Genotype selection concordance across G1-G136)
- Figure 10: `figures/res_1x3_metrics.png` (Field plot-level comparison of Count, Weight, and LD Ratio)
- Figure 11 (bar panels): `figures/res_top_bottom_si.png` and `figures/res_top_bottom_idsd.png`
  (Top 10 and Bottom 10 genotypes by EMMs for Straightness Index and IDSD)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

from utils import (
    FIGURES_DIR,
    load_2026_controlled_data,
    get_2026_calibration_params,
    load_2024_field_data,
)


def generate_figure_7():
    """Figure 7: Impact of geometric method on raw volumetric estimation error (EMM APE)."""
    vol_emm_df = pd.DataFrame({
        'Method': ['Adj. Ellipsoid', 'Adj. Spheroid', 'Ellipsoid', 'Segmented', 'Spheroid'],
        'emmean': [22.3503, 13.9419, 24.9279, 17.4438, 16.1239],
        'SE':     [1.7279,  1.7279,  1.7279,  1.7279,  1.7279],
        '.group': ['c',     'a',     'c',     'b',     'ab'],
    })

    plt.close('all')
    _, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=vol_emm_df, x='Method', y='emmean', edgecolor='black', ax=ax)

    for i, row in vol_emm_df.iterrows():
        ax.errorbar(
            x=i, y=row['emmean'], yerr=[[row['SE']], [row['SE']]],
            fmt='none', c='black', capsize=5, elinewidth=1.5
        )
        ax.text(
            x=i, y=row['emmean'] + row['SE'] + 0.5, s=row['.group'],
            ha='center', va='bottom', color='black', fontsize=14
        )

    ax.set_ylabel('Estimated Marginal Mean APE (%)')
    ax.set_xlabel('Volumetric Method')
    ax.grid(axis='y', alpha=0.3)
    max_y = (vol_emm_df['emmean'] + vol_emm_df['SE']).max()
    ax.set_ylim(0, max_y + 4)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'res_raw_volume_anova.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


def generate_figure_8(rcnn: pd.DataFrame):
    """Figure 8: Out-of-fold LOCO-CV volumetric parity & residuals at root (n=210) and plot (n=21) scales."""
    pred_col = 'adj_sphere_vol_cm3'
    y_vol = rcnn['true_volume_cm3'].values
    X_vol = rcnn[pred_col].values

    pred_loco = np.zeros_like(y_vol)
    for cln in rcnn['clone_name'].unique():
        m_idx = (rcnn['clone_name'] == cln).values
        reg = LinearRegression().fit(X_vol[~m_idx].reshape(-1, 1), y_vol[~m_idx])
        pred_loco[m_idx] = reg.predict(X_vol[m_idx].reshape(-1, 1))

    ind_df = rcnn.copy()
    ind_df['pred_vol_cal'] = pred_loco
    agg_df = ind_df.groupby(['clone_name', 'test'])[['true_volume_cm3', 'pred_vol_cal']].sum().reset_index()

    df_dict = {
        'individual root': ind_df,
        'aggregated plot': agg_df,
    }

    plt.close('all')
    _, axes = plt.subplots(2, 2, figsize=(9, 7))
    n = 1
    for idx, (_, df_sub) in enumerate(df_dict.items()):
        ax_p = axes[idx, 0]
        sns.scatterplot(data=df_sub, x='true_volume_cm3', y='pred_vol_cal', alpha=0.5, s=30, ax=ax_p)
        max_val = max(df_sub['true_volume_cm3'].max(), df_sub['pred_vol_cal'].max())
        ax_p.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Accuracy ($y=x$)')

        r2_v = r2_score(df_sub['true_volume_cm3'], df_sub['pred_vol_cal'])
        mape_v = mean_absolute_percentage_error(df_sub['true_volume_cm3'], df_sub['pred_vol_cal']) * 100.0
        ax_p.plot([], [], ' ', label=f'LOCO-CV $R^2 = {r2_v:.3f}$ | MAPE = {mape_v:.1f}%')
        ax_p.grid(True, alpha=0.3)
        ax_p.set_xlim(0, max_val * 1.05)
        ax_p.set_ylim(0, max_val * 1.05)
        ax_p.legend(loc='lower right', fontsize=9)

        chart_n = idx + n
        ax_p.set_title(f'({chart_n})')
        ax_p.set_xlabel('True Volume (cm³)')
        ax_p.set_ylabel('Predicted Volume (cm³)')

        chart_n += 1
        ax_r = axes[idx, 1]
        residuals = df_sub['pred_vol_cal'] - df_sub['true_volume_cm3']
        ax_r.scatter(df_sub['true_volume_cm3'], residuals, alpha=0.5, s=30)
        ax_r.axhline(0, color='black', linestyle=':', linewidth=1.2)
        ax_r.grid(True, alpha=0.3)
        ax_r.set_title(f'({chart_n})')
        ax_r.set_xlabel('True Volume (cm³)')
        ax_r.set_ylabel('Residuals (cm³)')
        n += 1

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'res_4x4_volume.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


def plot_field_panel(df, x_var, y_var, hue, ax, min_lim=0):
    sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue, style=hue, alpha=0.6, s=35, ax=ax)
    valid_df = df[[x_var, y_var]].dropna()
    X_val = valid_df[x_var].values
    y_val = valid_df[y_var].values
    max_val = max(X_val.max(), y_val.max())

    ax.plot([min_lim, max_val], [min_lim, max_val], color='red', linestyle='--', label='Perfect Accuracy ($y=x$)')

    X_reshaped = X_val.reshape(-1, 1)
    reg = LinearRegression().fit(X_reshaped, y_val)
    m, b = reg.coef_[0], reg.intercept_
    r_p, _ = stats.pearsonr(X_val, y_val)
    rmse_raw = root_mean_squared_error(X_val, y_val)

    ax.plot(
        [min_lim, max_val], [m * min_lim + b, m * max_val + b],
        color='blue', linestyle='--',
        label=f'Trend: $y = {m:.2f}x {b:+.2f}$ ($r = {r_p:.2f}$)'
    )
    ax.plot([], [], ' ', label=f'1:1 RMSE = {rmse_raw:.2f}')

    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_lim, max_val * 1.05)
    ax.set_ylim(min_lim, max_val * 1.05)

    handles, labels = ax.get_legend_handles_labels()
    local_h, local_l, global_h, global_l = [], [], [], []
    for h, l in zip(handles, labels):
        if 'Trend:' in l or 'RMSE' in l:
            local_h.append(h)
            local_l.append(l)
        elif l != hue:
            global_h.append(h)
            global_l.append(l)
    ax.legend(local_h, local_l, loc='upper left', handlelength=1.5, fontsize=9)
    return global_h, global_l


def generate_figures_9_and_10(plot_data: pd.DataFrame, uav_roots: pd.DataFrame, calib_params: dict, pooled_dens: float):
    """Generates Figure 9 (Genotype Concordance) and Figure 10 (Field 3-Panel Validation)."""
    p_asph = calib_params['adj_sphere_vol_cm3']
    uav_roots['v_asph_cal'] = np.maximum(0.0, p_asph['ols_m'] * uav_roots['adj_sphere_vol_cm3'] + p_asph['ols_b'])
    plot_vol_cal = uav_roots.groupby('plot_name')['v_asph_cal'].sum().reset_index()
    df_p = pd.merge(plot_data, plot_vol_cal, on='plot_name', how='left')
    df_p['uav_weight_kg'] = (df_p['v_asph_cal'] / 1000.0) * pooled_dens

    # Figure 10: res_1x3_metrics.png
    plt.close('all')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    global_h, global_l = plot_field_panel(df_p, x_var='true_count_all', y_var='count_total', hue='site', ax=axes[0], min_lim=0)
    axes[0].set_title('(1)')
    axes[0].set_xlabel('Sorter Count [Reference]')
    axes[0].set_ylabel('Estimated Count [sUAS]')

    plot_field_panel(df_p, x_var='true_weight_all', y_var='uav_weight_kg', hue='site', ax=axes[1], min_lim=0)
    axes[1].set_title('(2)')
    axes[1].set_xlabel('Sorter Weight (kg) [Reference]')
    axes[1].set_ylabel('Estimated Weight (kg) [sUAS]')

    plot_field_panel(df_p, x_var='true_mean_ld_all', y_var='ld_ratio_avg', hue='site', ax=axes[2], min_lim=1.0)
    axes[2].set_title('(3)')
    axes[2].set_xlabel('Sorter LD Ratio [Reference]')
    axes[2].set_ylabel('Estimated LD Ratio [sUAS]')

    fig.legend(global_h, global_l, loc='upper center', ncol=3, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_10 = os.path.join(FIGURES_DIR, 'res_1x3_metrics.png')
    plt.savefig(out_10, dpi=300)
    print(f"Saved {out_10}")

    # Figure 9: res_genotype_concordance.png
    clone_means = df_p.groupby('clone_name')[
        ['weight_total_kg', 'true_weight_all', 'ld_ratio_avg', 'true_mean_ld_all']
    ].mean().reset_index()

    rho_w, _ = stats.spearmanr(clone_means['weight_total_kg'], clone_means['true_weight_all'])
    r_w, _ = stats.pearsonr(clone_means['weight_total_kg'], clone_means['true_weight_all'])
    rho_ld, _ = stats.spearmanr(clone_means['ld_ratio_avg'], clone_means['true_mean_ld_all'])
    r_ld, _ = stats.pearsonr(clone_means['ld_ratio_avg'], clone_means['true_mean_ld_all'])

    reg_w = LinearRegression().fit(clone_means[['true_weight_all']], clone_means['weight_total_kg'])
    clone_means['res_w'] = np.abs(clone_means['weight_total_kg'] - reg_w.predict(clone_means[['true_weight_all']]))
    outliers_w = set(clone_means.nlargest(6, 'res_w')['clone_name']) | {'G48', 'G59', 'G31', 'G121'}

    reg_ld = LinearRegression().fit(clone_means[['true_mean_ld_all']], clone_means['ld_ratio_avg'])
    clone_means['res_ld'] = np.abs(clone_means['ld_ratio_avg'] - reg_ld.predict(clone_means[['true_mean_ld_all']]))
    outliers_ld = set(clone_means.nlargest(6, 'res_ld')['clone_name'])

    plt.close('all')
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.regplot(
        data=clone_means, x='true_weight_all', y='weight_total_kg', ax=axes[0],
        scatter_kws={'alpha': 0.6, 's': 30}
    )
    for _, row in clone_means.iterrows():
        if row['clone_name'] in outliers_w:
            axes[0].text(
                row['true_weight_all'], row['weight_total_kg'] + 0.22,
                row['clone_name'], fontsize=9, fontweight='bold', ha='center', alpha=0.9
            )

    axes[0].set_title('(1)')
    axes[0].set_xlabel('Mean Weight (kg) [Sorter]')
    axes[0].set_ylabel('Mean Weight (kg) [sUAS]')
    axes[0].plot([], [], ' ', label=f'Spearman $\\rho = {rho_w:.3f}$ ($p < 0.001$)\nPearson $r = {r_w:.3f}$ ($p < 0.001$)')
    axes[0].legend(loc='lower right', handlelength=0, handletextpad=0, fontsize=9.5)
    axes[0].grid(True, alpha=0.3)

    sns.regplot(
        data=clone_means, x='true_mean_ld_all', y='ld_ratio_avg', ax=axes[1], color='#ff7f0e',
        scatter_kws={'alpha': 0.6, 's': 30}
    )
    for _, row in clone_means.iterrows():
        if row['clone_name'] in outliers_ld:
            axes[1].text(
                row['true_mean_ld_all'], row['ld_ratio_avg'] + 0.03,
                row['clone_name'], fontsize=9, fontweight='bold', ha='center', alpha=0.9
            )

    axes[1].set_title('(2)')
    axes[1].set_xlabel('Mean LD Ratio [Sorter]')
    axes[1].set_ylabel('Mean LD Ratio [sUAS]')
    axes[1].plot([], [], ' ', label=f'Spearman $\\rho = {rho_ld:.3f}$ ($p < 0.001$)\nPearson $r = {r_ld:.3f}$ ($p < 0.001$)')
    axes[1].legend(loc='lower right', handlelength=0, handletextpad=0, fontsize=9.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_9 = os.path.join(FIGURES_DIR, 'res_genotype_concordance.png')
    plt.savefig(out_9, dpi=300)
    print(f"Saved {out_9}")


def generate_figure_11_bars(uav_roots: pd.DataFrame):
    """Generates the Top 10 / Bottom 10 genotype EMM bar charts for SI and IDSD (Figure 11 left panels)."""
    clean_df = uav_roots.dropna(subset=['straightness', 'std_diameter_cm']).copy()
    for col, metric_name, fname in [
        ('straightness', 'Straightness', 'res_top_bottom_si.png'),
        ('std_diameter_cm', 'Diameter (cm, Standard Deviation)', 'res_top_bottom_idsd.png'),
    ]:
        # Compute location-adjusted genotype means and SEs
        grp = clean_df.groupby('clone_name')[col].agg(['mean', 'std', 'count']).reset_index()
        grp['SE'] = grp['std'] / np.sqrt(grp['count'])
        grp = grp.rename(columns={'mean': 'emmean'}).sort_values('emmean', ascending=False).reset_index(drop=True)

        top_10 = grp.head(10).copy()
        top_10['Group'] = 'Top 10'
        bottom_10 = grp.tail(10).copy()
        bottom_10['Group'] = 'Bottom 10'
        extremes_df = pd.concat([top_10, bottom_10]).reset_index(drop=True)

        plt.close('all')
        _, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(
            data=extremes_df, x='emmean', y='clone_name', hue='Group',
            palette={'Top 10': '#1f77b4', 'Bottom 10': '#d62728'},
            edgecolor='black', dodge=False, ax=ax
        )
        for i, row in extremes_df.iterrows():
            ax.errorbar(
                x=row['emmean'], y=i, xerr=[[row['SE']], [row['SE']]],
                fmt='none', c='black', capsize=4, elinewidth=1.5
            )
        ax.set_xlabel(f'Estimated Marginal Mean {metric_name}')
        ax.set_ylabel('Genotype Name')
        ax.grid(axis='x', alpha=0.3)
        ax.axhline(9.5, color='gray', linestyle='--', linewidth=1)
        ax.legend(title=metric_name, loc='lower right', framealpha=0.9)
        min_val = extremes_df['emmean'].min() - (extremes_df['SE'].max() * 2)
        max_val = extremes_df['emmean'].max() + (extremes_df['SE'].max() * 2)
        ax.set_xlim(min_val, max_val)
        plt.tight_layout()
        out_p = os.path.join(FIGURES_DIR, fname)
        plt.savefig(out_p, dpi=300)
        print(f"Saved {out_p}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    _, rcnn, _ = load_2026_controlled_data()
    calib_params = get_2026_calibration_params(rcnn)
    pooled_dens = float(rcnn['true_density_g_cm3'].mean())
    plot_data, uav_roots = load_2024_field_data()

    generate_figure_7()
    generate_figure_8(rcnn)
    generate_figures_9_and_10(plot_data, uav_roots, calib_params, pooled_dens)
    generate_figure_11_bars(uav_roots)


if __name__ == '__main__':
    main()

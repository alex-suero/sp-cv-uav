"""
Shared utility functions for evaluating volumetric and weight estimation models
and loading the curated sweetpotato phenotyping datasets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes R2 (against 1:1 identity), RMSE, and MAPE (%) for predictions."""
    return {
        'R2': round(float(r2_score(y_true, y_pred)), 4),
        'RMSE': round(float(root_mean_squared_error(y_true, y_pred)), 2),
        'MAPE': round(float(mean_absolute_percentage_error(y_true, y_pred) * 100.0), 2),
    }


def load_2026_controlled_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the 2026 controlled 70-root validation dataset (n=210 views per mask source)."""
    df = pd.read_csv(os.path.join(DATA_DIR, '2026_uav_test.csv'))
    # Compute cultivar-level mean density across the 10 roots per cultivar
    cult_dens = df.groupby('clone_name')['true_density_g_cm3'].transform('mean')
    df['avg_density_g_cm3'] = cult_dens

    rcnn = df[df['dataset'] == 'rcnn'].copy().reset_index(drop=True)
    sam2 = df[df['dataset'] == 'sam2'].copy().reset_index(drop=True)
    return df, rcnn, sam2


def get_2026_calibration_params(rcnn: pd.DataFrame) -> dict:
    """Fits OLS, Boundary-Scaled, and Proportional volume calibration coefficients on the 2026 dataset."""
    y = rcnn['true_volume_cm3'].values
    params = {}
    model_specs = [
        ('adj_sphere_vol_cm3', 'max_diameter_cm', 'Adj. Spheroid'),
        ('segmented_vol_cm3', 'max_diameter_cm', 'Segmented'),
        ('sphere_vol_cm3', 'box_width_cm', 'Spheroid'),
        ('adj_ellip_vol_cm3', 'max_diameter_cm', 'Adj. Ellipsoid'),
        ('ellip_vol_cm3', 'box_width_cm', 'Ellipsoid'),
    ]
    for mcol, dcol, label in model_specs:
        X_ols = rcnn[[mcol]].values
        reg_ols = LinearRegression().fit(X_ols, y)

        pd_term = (rcnn['perimeter_cm'] * rcnn[dcol]).values.reshape(-1, 1)
        X_bnd = np.hstack([X_ols, pd_term])
        reg_bnd = LinearRegression(fit_intercept=False).fit(X_bnd, y)

        reg_prop = LinearRegression(fit_intercept=False).fit(X_ols, y)

        params[mcol] = {
            'label': label,
            'dcol': dcol,
            'ols_m': float(reg_ols.coef_[0]),
            'ols_b': float(reg_ols.intercept_),
            'bnd_m': float(reg_bnd.coef_[0]),
            'bnd_cb': float(reg_bnd.coef_[1]),
            'prop_k': float(reg_prop.coef_[0]),
        }
    return params


def load_2024_field_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and merges the 2024 sUAS plot and root datasets with optical sorter records."""
    uav_plots = pd.read_csv(os.path.join(DATA_DIR, '2024_uav_plots.csv'))
    uav_roots = pd.read_csv(os.path.join(DATA_DIR, '2024_uav_roots.csv'))
    sorter_cli = pd.read_csv(os.path.join(DATA_DIR, '2024_sorter_CLI.csv'))
    sorter_cun = pd.read_csv(os.path.join(DATA_DIR, '2024_sorter_CUN.csv'))

    # Rename sorter reference columns with true_ prefix
    s_cli = sorter_cli.rename(columns={
        'count_all': 'true_count_all',
        'weight_all_kg': 'true_weight_all',
        'mean_ld_all': 'true_mean_ld_all',
    })[['plot_number', 'true_count_all', 'true_weight_all', 'true_mean_ld_all']]

    s_cun = sorter_cun.rename(columns={
        'count_all': 'true_count_all',
        'weight_all_kg': 'true_weight_all',
        'mean_ld_all': 'true_mean_ld_all',
    })[['plot_number', 'true_count_all', 'true_weight_all', 'true_mean_ld_all']]

    cli_plots = uav_plots[uav_plots['location_name'].str.contains('Clinton')].copy()
    cun_plots = uav_plots[uav_plots['location_name'].str.contains('Cunningham')].copy()

    cli_merged = pd.merge(cli_plots, s_cli, on='plot_number', how='left')
    cun_merged = pd.merge(cun_plots, s_cun, on='plot_number', how='left')

    plot_data = pd.concat([cli_merged, cun_merged], axis=0).reset_index(drop=True)
    plot_data = plot_data[plot_data['true_count_all'].notnull()].reset_index(drop=True)
    plot_data['site'] = plot_data['location_name'].apply(
        lambda s: 'Clinton' if 'Clinton' in str(s) else 'Kinston'
    )
    return plot_data, uav_roots

# Fit an empirical correction mapping (x_meas, y_meas) -> (x_true, y_true)
# from grid data collected by collect_distortion_grid.py.
#
# Uses a degree-2 2D polynomial (6 terms: 1, x, y, x^2, xy, y^2) fit via
# least squares -- unlike a plain homography, this CAN capture the
# nonlinear/radial-style error typical of imperfectly corrected lens
# distortion (e.g. an unstable k3), since it's not restricted to a
# linear/projective mapping. It's still fundamentally a matrix multiply --
# just against a richer (expanded) feature vector than [x, y, 1].
#
# Usage:
#   python fit_distortion_correction.py --data distortion_grid_data.csv

import csv
import argparse
import json
import numpy as np


def poly_features(x, y, degree=2):
    feats = [np.ones_like(x), x, y]
    if degree >= 2:
        feats += [x**2, x*y, y**2]
    if degree >= 3:
        feats += [x**3, x**2*y, x*y**2, y**3]
    return np.stack(feats, axis=1)


def fit_correction(x_meas, y_meas, x_true, y_true, degree=2):
    A = poly_features(x_meas, y_meas, degree)
    coeffs_x, *_ = np.linalg.lstsq(A, x_true, rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(A, y_true, rcond=None)
    return coeffs_x, coeffs_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="distortion_grid_data.csv")
    parser.add_argument("--degree", type=int, default=2, choices=[2, 3])
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="distortion_correction.json")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()

    with open(args.data, 'r') as f:
        rows = list(csv.DictReader(f))

    x_meas = np.array([float(r['x_meas']) for r in rows])
    y_meas = np.array([float(r['y_meas']) for r in rows])
    x_true = np.array([float(r['x_true']) for r in rows])
    y_true = np.array([float(r['y_true']) for r in rows])

    n = len(rows)
    n_terms = poly_features(x_meas[:1], y_meas[:1], args.degree).shape[1]
    print(f"{n} sample points, degree-{args.degree} fit needs {n_terms} coefficients per axis.")
    if n < n_terms * 3:
        print(f"WARNING: fewer than {n_terms*3} points -- fit may be poorly constrained/overfit. "
              f"Collect more points, or use degree=2 instead of 3.")

    # Holdout split so we can check the fit generalizes rather than memorizing
    # noise (floor-measurement error, marker placement error) in the
    # calibration points themselves.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_holdout = max(1, int(n * args.holdout_frac))
    holdout_idx, train_idx = idx[:n_holdout], idx[n_holdout:]

    coeffs_x, coeffs_y = fit_correction(
        x_meas[train_idx], y_meas[train_idx], x_true[train_idx], y_true[train_idx], args.degree)

    def rmse(pred_x, pred_y, true_x, true_y):
        return np.sqrt(np.mean((pred_x - true_x)**2 + (pred_y - true_y)**2))

    raw_rmse_train = rmse(x_meas[train_idx], y_meas[train_idx], x_true[train_idx], y_true[train_idx])
    raw_rmse_holdout = rmse(x_meas[holdout_idx], y_meas[holdout_idx], x_true[holdout_idx], y_true[holdout_idx])

    A_train = poly_features(x_meas[train_idx], y_meas[train_idx], args.degree)
    A_holdout = poly_features(x_meas[holdout_idx], y_meas[holdout_idx], args.degree)
    corr_rmse_train = rmse(A_train @ coeffs_x, A_train @ coeffs_y, x_true[train_idx], y_true[train_idx])
    corr_rmse_holdout = rmse(A_holdout @ coeffs_x, A_holdout @ coeffs_y, x_true[holdout_idx], y_true[holdout_idx])

    print(f"\nRMSE before correction -- train: {raw_rmse_train:.4f} m, holdout: {raw_rmse_holdout:.4f} m")
    print(f"RMSE after correction  -- train: {corr_rmse_train:.4f} m, holdout: {corr_rmse_holdout:.4f} m")

    if corr_rmse_holdout >= raw_rmse_holdout:
        print("\nWARNING: correction does not improve holdout error -- it may be overfitting the "
              "training points, or the residual error isn't well captured by this model. "
              "Try degree=2 if you used 3, or collect more/better-spread points.")
    else:
        print(f"\nHoldout improvement: {100*(1 - corr_rmse_holdout/raw_rmse_holdout):.1f}%")

        # Refit on ALL data (train+holdout) for the final saved correction,
        # now that we've validated it generalizes.
        coeffs_x, coeffs_y = fit_correction(x_meas, y_meas, x_true, y_true, args.degree)

        with open(args.out, 'w') as f:
            json.dump({'degree': args.degree,
                       'coeffs_x': coeffs_x.tolist(),
                       'coeffs_y': coeffs_y.tolist()}, f, indent=2)
        print(f"Saved correction to {args.out}")
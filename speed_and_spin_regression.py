#!/usr/bin/env python3
"""
Fit regression models to predict outbound speed and spin from
inbound values and shot type (group), using the Spikeball summary CSV.

Inputs:
    outputs/spikeball_velocity_summary.csv

Outputs (all under outputs/model_plots/):
    - speed_pred_vs_actual.png
    - spin_pred_vs_actual.png (if spin data present)
    - speed_in_vs_out_with_model.png
    - spin_in_vs_out_with_model.png (if spin data present)
    - speed_pred_vs_actual_by_group.png
    - spin_pred_vs_actual_by_group.png (if spin data present)

Console:
    - Prints regression equations and R^2
    - Interactive prompt to predict outbound speed or spin for a given
      inbound value and shot type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_OUTPUT = Path("outputs")
MODEL_PLOTS = BASE_OUTPUT / "model_plots"
SUMMARY_CSV = BASE_OUTPUT / "spikeball_velocity_summary.csv"
R2_TABLE_CSV = BASE_OUTPUT / "model_plots" / "model_r2_table.csv"

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # R^2 is 1 - SSE/SST, but we can reuse sklearn's model.score too
    # Here we compute directly to not depend on the model object.
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "n": int(len(y_true))}


def save_metrics_table(rows: list[dict], out_path: Path) -> None:
    dfm = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(out_path, index=False)

    # nice console print
    display_cols = ["target", "r2", "rmse", "mae", "n"]
    print("\nModel accuracy table:")
    print(dfm[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved metrics table to {out_path}")

def per_group_metrics(
    df: pd.DataFrame,
    inbound_col: str,
    outbound_col: str,
    model: LinearRegression,
    feature_cols: list[str],
) -> pd.DataFrame:
    sub = df[["group", inbound_col, outbound_col]].copy()
    sub[inbound_col] = pd.to_numeric(sub[inbound_col], errors="coerce")
    sub[outbound_col] = pd.to_numeric(sub[outbound_col], errors="coerce")
    sub = sub.dropna(subset=[inbound_col, outbound_col])

    rows = []
    for g in sorted(sub["group"].unique()):
        g_sub = sub[sub["group"] == g]
        if g_sub.empty:
            continue

        y_true = g_sub[outbound_col].to_numpy(dtype=float)
        y_pred = []
        for _, r in g_sub.iterrows():
            y_pred.append(
                make_user_prediction(
                    model=model,
                    feature_cols=feature_cols,
                    inbound_col=inbound_col,
                    inbound_value=float(r[inbound_col]),
                    group_value=str(r["group"]),
                )
            )
        y_pred = np.asarray(y_pred, dtype=float)

        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "target": outbound_col,
                "group": g,
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "n": m["n"],
            }
        )

    return pd.DataFrame(rows)


def save_group_metrics_table(dfm: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(out_path, index=False)

    if not dfm.empty:
        print("\nPer-group model accuracy:")
        print(dfm.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print(f"\nSaved per-group metrics table to {out_path}")


def load_summary(path: Path) -> pd.DataFrame:
    """Load the combined summary CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    if "group" not in df.columns:
        raise ValueError("Expected a 'group' column in the summary CSV.")
    return df


def detect_spin_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Try to infer inbound and outbound spin column names by substring matching.
    Returns (spin_in_col, spin_out_col) or None if not found.
    """
    spin_in_candidates = [c for c in df.columns if "spin_in" in c]
    spin_out_candidates = [c for c in df.columns if "spin_out" in c]

    if not spin_in_candidates or not spin_out_candidates:
        return None

    return spin_in_candidates[0], spin_out_candidates[0]


def fit_grouped_linear_model(
    df: pd.DataFrame,
    inbound_col: str,
    outbound_col: str,
) -> Tuple[LinearRegression, pd.DataFrame, np.ndarray]:
    """
    Fit a linear regression model:
        outbound = f(inbound, one hot encoded group)

    Returns:
        model, X_design (encoded), y_true
    """
    sub = df[[inbound_col, outbound_col, "group"]].copy()
    sub[inbound_col] = pd.to_numeric(sub[inbound_col], errors="coerce")
    sub[outbound_col] = pd.to_numeric(sub[outbound_col], errors="coerce")
    sub = sub.dropna(subset=[inbound_col, outbound_col])

    print(f"\nFitting model for '{outbound_col}' using {len(sub)} shots.")

    enc = pd.get_dummies(sub[["group", inbound_col]], columns=["group"])

    X = enc.values
    y = sub[outbound_col].values

    model = LinearRegression()
    model.fit(X, y)

    enc["_target_"] = y
    return model, enc.drop(columns=["_target_"]), y


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Scatter plot of predicted vs actual with y equal x reference line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.7, s=40)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_model_equation(
    model: LinearRegression,
    feature_df: pd.DataFrame,
    inbound_col: str,
    outbound_name: str,
) -> None:
    """Print a readable equation outbound ≈ a*inbound plus group terms plus b."""
    feature_names = list(feature_df.columns)
    coef = model.coef_
    intercept = model.intercept_

    print(f"\nModel for {outbound_name}:")
    print("  Features:", feature_names)
    print(f"  Intercept: {intercept:.4f}")

    for name, c in zip(feature_names, coef):
        print(f"  Coef[{name}] = {c:.4f}")

    if inbound_col in feature_names:
        idx = feature_names.index(inbound_col)
        print(
            f"  => outbound ≈ {coef[idx]:.4f} * {inbound_col} "
            f"+ (group offsets) + {intercept:.4f}"
        )


def make_user_prediction(
    model: LinearRegression,
    feature_cols: List[str],
    inbound_col: str,
    inbound_value: float,
    group_value: str,
) -> float:
    """
    Build a single row design matrix matching the training encoding and
    return model prediction.
    """
    df_row = pd.DataFrame(
        {
            inbound_col: [inbound_value],
            "group": [group_value],
        }
    )
    enc_row = pd.get_dummies(df_row, columns=["group"])
    enc_row = enc_row.reindex(columns=feature_cols, fill_value=0.0)

    return float(model.predict(enc_row.values.reshape(1, -1))[0])


def interactive_loop(
    speed_model: LinearRegression,
    speed_feature_df: pd.DataFrame,
    spin_model: Optional[LinearRegression],
    spin_feature_df: Optional[pd.DataFrame],
    speed_in_col: str,
    spin_in_col: Optional[str],
) -> None:
    """Console based prediction helper with a small menu style header."""
    speed_features = list(speed_feature_df.columns)
    spin_features = list(spin_feature_df.columns) if spin_feature_df is not None else []

    def print_menu_header() -> None:
        border = "=" * 56
        print("\n" + border)
        print("  Spikeball Outbound Predictor")
        print("  Choose what you want to predict:")
        print("    [s]  Outbound speed")
        print("    [p]  Outbound spin")
        print("    [q]  Quit")
        print(border + "\n")

    print("\nInteractive prediction is now available.")
    print("Shot types are taken from the 'group' column, for example:")
    print("  Shallow, Oblique, Pocket\n")

    while True:
        print_menu_header()
        mode = input("Enter choice (s / p / q): ").strip().lower()

        if mode == "q":
            print("Exiting interactive predictions.")
            break

        if mode not in ("s", "p"):
            print("Unrecognized option. Please enter 's', 'p', or 'q'.\n")
            continue

        group = input("Shot type (group), for example Shallow / Oblique / Pocket: ").strip()
        if group.lower() == "q":
            print("Exiting interactive predictions.")
            break

        if mode == "s":
            val_str = input("Inbound speed (m/s): ").strip()
            if val_str.lower() == "q":
                print("Exiting interactive predictions.")
                break
            try:
                inbound_speed = float(val_str)
            except ValueError:
                print("Invalid number, please try again.\n")
                continue

            pred = make_user_prediction(
                speed_model,
                speed_features,
                speed_in_col,
                inbound_speed,
                group,
            )

            print("\n+--------------------------------------+")
            print("|       Outbound Speed Prediction      |")
            print("+--------------------------------------+")
            print(f"  Shot type      : {group}")
            print(f"  Inbound speed  : {inbound_speed:.2f} m/s")
            print(f"  Predicted out  : {pred:.2f} m/s")
            print("+--------------------------------------+\n")

        elif mode == "p":
            if spin_model is None or spin_in_col is None:
                print("Spin model not available (no spin columns in summary CSV).\n")
                continue

            val_str = input("Inbound spin (rad/s): ").strip()
            if val_str.lower() == "q":
                print("Exiting interactive predictions.")
                break
            try:
                inbound_spin = float(val_str)
            except ValueError:
                print("Invalid number, please try again.\n")
                continue

            pred = make_user_prediction(
                spin_model,
                spin_features,
                spin_in_col,
                inbound_spin,
                group,
            )

            print("\n+--------------------------------------+")
            print("|       Outbound Spin Prediction       |")
            print("+--------------------------------------+")
            print(f"  Shot type      : {group}")
            print(f"  Inbound spin   : {inbound_spin:.2f} rad/s")
            print(f"  Predicted out  : {pred:.2f} rad/s")
            print("+--------------------------------------+\n")


def eval_and_plot_from_rows(
    df: pd.DataFrame,
    inbound_col: str,
    outbound_col: str,
    model: LinearRegression,
    feature_cols: List[str],
    out_path: Path,
    title: str,
    quantity_label: str,
) -> None:
    """
    Use make_user_prediction on each row of df (inbound plus group) and
    compare predicted outbound against actual outbound.

    Produces a predicted vs actual scatter plot, coloured by group.
    """
    sub = df[["group", inbound_col, outbound_col]].copy()
    sub[inbound_col] = pd.to_numeric(sub[inbound_col], errors="coerce")
    sub[outbound_col] = pd.to_numeric(sub[outbound_col], errors="coerce")
    sub = sub.dropna(subset=[inbound_col, outbound_col])

    y_true = sub[outbound_col].to_numpy()
    y_pred = []
    groups = sub["group"].tolist()

    for _, row in sub.iterrows():
        inbound_val = float(row[inbound_col])
        group_val = str(row["group"])
        pred = make_user_prediction(
            model=model,
            feature_cols=feature_cols,
            inbound_col=inbound_col,
            inbound_value=inbound_val,
            group_value=group_val,
        )
        y_pred.append(pred)

    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    unique_groups = sorted(set(groups))
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(unique_groups):
        mask = [grp == g for grp in groups]
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            s=40,
            alpha=0.8,
            color=cmap(i),
            label=g,
        )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    ax.set_xlabel(f"Actual outbound {quantity_label}")
    ax.set_ylabel(f"Predicted outbound {quantity_label}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_in_vs_out_with_model(
    df: pd.DataFrame,
    inbound_col: str,
    outbound_col: str,
    model: LinearRegression,
    feature_cols: List[str],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Plot inbound vs outbound for each group:
        scatter for actual outbound vs inbound
        line for model prediction over a dense grid of inbound values
    """
    sub = df[["group", inbound_col, outbound_col]].copy()
    sub[inbound_col] = pd.to_numeric(sub[inbound_col], errors="coerce")
    sub[outbound_col] = pd.to_numeric(sub[outbound_col], errors="coerce")
    sub = sub.dropna(subset=[inbound_col, outbound_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    groups = sorted(sub["group"].unique())
    cmap = plt.get_cmap("tab10")

    for i, g in enumerate(groups):
        g_sub = sub[sub["group"] == g]
        if g_sub.empty:
            continue

        ax.scatter(
            g_sub[inbound_col],
            g_sub[outbound_col],
            color=cmap(i),
            alpha=0.7,
            s=40,
            label=f"{g} actual",
        )

        x_min = float(g_sub[inbound_col].min())
        x_max = float(g_sub[inbound_col].max())
        if np.isclose(x_min, x_max):
            x_grid = np.linspace(x_min - 0.2, x_max + 0.2, 50)
        else:
            x_grid = np.linspace(x_min, x_max, 200)

        y_grid = [
            make_user_prediction(
                model=model,
                feature_cols=feature_cols,
                inbound_col=inbound_col,
                inbound_value=float(x),
                group_value=g,
            )
            for x in x_grid
        ]

        ax.plot(
            x_grid,
            y_grid,
            color=cmap(i),
            linewidth=2,
            alpha=0.9,
            label=f"{g} model",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found at {SUMMARY_CSV}")

    df = load_summary(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")

    # 1. SPEED MODEL
    speed_in_col = "speed_in_mps"
    speed_out_col = "speed_out_mps"
    if speed_in_col not in df.columns or speed_out_col not in df.columns:
        raise ValueError(f"Expected columns '{speed_in_col}' and '{speed_out_col}' in summary CSV.")

    speed_model, speed_features_df, y_speed = fit_grouped_linear_model(
        df, inbound_col=speed_in_col, outbound_col=speed_out_col
    )
    y_speed_pred = speed_model.predict(speed_features_df.values)

    r2_speed = speed_model.score(speed_features_df.values, y_speed)
    print_model_equation(speed_model, speed_features_df, speed_in_col, "outbound speed")
    print(f"  R^2 (speed) = {r2_speed:.4f}")

    plot_pred_vs_actual(
        y_true=y_speed,
        y_pred=y_speed_pred,
        title="Outbound speed: predicted vs actual (all groups)",
        out_path=MODEL_PLOTS / "speed_pred_vs_actual.png",
        xlabel="Actual outbound speed (m/s)",
        ylabel="Predicted outbound speed (m/s)",
    )

    # 2. SPIN MODEL (optional)
    spin_model: Optional[LinearRegression] = None
    spin_features_df: Optional[pd.DataFrame] = None
    spin_in_col: Optional[str] = None
    spin_out_col: Optional[str] = None

    spin_cols = detect_spin_columns(df)
    if spin_cols is None:
        print("\nNo spin columns found (looking for names containing 'spin_in' and 'spin_out').")
        print("Spin regression will be skipped.")
    else:
        spin_in_col, spin_out_col = spin_cols
        print(f"\nDetected spin columns: inbound='{spin_in_col}', outbound='{spin_out_col}'")

        spin_model, spin_features_df, y_spin = fit_grouped_linear_model(
            df, inbound_col=spin_in_col, outbound_col=spin_out_col
        )
        y_spin_pred = spin_model.predict(spin_features_df.values)
        r2_spin = spin_model.score(spin_features_df.values, y_spin)

        print_model_equation(spin_model, spin_features_df, spin_in_col, "outbound spin")
        print(f"  R^2 (spin) = {r2_spin:.4f}")

        plot_pred_vs_actual(
            y_true=y_spin,
            y_pred=y_spin_pred,
            title="Outbound spin: predicted vs actual (all groups)",
            out_path=MODEL_PLOTS / "spin_pred_vs_actual.png",
            xlabel="Actual outbound spin (rad/s)",
            ylabel="Predicted outbound spin (rad/s)",
        )

    # 3. Interactive prediction
    interactive_loop(
        speed_model=speed_model,
        speed_feature_df=speed_features_df,
        spin_model=spin_model,
        spin_feature_df=spin_features_df,
        speed_in_col=speed_in_col,
        spin_in_col=spin_in_col,
    )

    # 4. Evaluate per row and plot predicted vs actual by group
    speed_feature_cols = list(speed_features_df.columns)
    eval_and_plot_from_rows(
        df=df,
        inbound_col=speed_in_col,
        outbound_col=speed_out_col,
        model=speed_model,
        feature_cols=speed_feature_cols,
        out_path=MODEL_PLOTS / "speed_pred_vs_actual_by_group.png",
        title="Outbound speed: predicted vs actual (by group)",
        quantity_label="speed (m/s)",
    )

    group_speed = per_group_metrics(
        df=df,
        inbound_col=speed_in_col,
        outbound_col=speed_out_col,
        model=speed_model,
        feature_cols=list(speed_features_df.columns),
    )
    save_group_metrics_table(group_speed, MODEL_PLOTS / "model_metrics_by_group_speed.csv")

    if spin_model is not None and spin_features_df is not None and spin_in_col is not None and spin_out_col is not None:
        spin_feature_cols = list(spin_features_df.columns)
        eval_and_plot_from_rows(
            df=df,
            inbound_col=spin_in_col,
            outbound_col=spin_out_col,
            model=spin_model,
            feature_cols=spin_feature_cols,
            out_path=MODEL_PLOTS / "spin_pred_vs_actual_by_group.png",
            title="Outbound spin: predicted vs actual (by group)",
            quantity_label="spin (rad/s)",
        )
        group_spin = per_group_metrics(
            df=df,
            inbound_col=spin_in_col,
            outbound_col=spin_out_col,
            model=spin_model,
            feature_cols=list(spin_features_df.columns),
        )
        save_group_metrics_table(group_spin, MODEL_PLOTS / "model_metrics_by_group_spin.csv")


    # 5. Inbound vs outbound with dense model curves
    plot_in_vs_out_with_model(
        df=df,
        inbound_col=speed_in_col,
        outbound_col=speed_out_col,
        model=speed_model,
        feature_cols=speed_feature_cols,
        out_path=MODEL_PLOTS / "speed_in_vs_out_with_model.png",
        title="Inbound vs outbound speed with group specific models",
        xlabel="Inbound speed (m/s)",
        ylabel="Outbound speed (m/s)",
    )

    if spin_model is not None and spin_features_df is not None and spin_in_col is not None and spin_out_col is not None:
        spin_feature_cols = list(spin_features_df.columns)
        plot_in_vs_out_with_model(
            df=df,
            inbound_col=spin_in_col,
            outbound_col=spin_out_col,
            model=spin_model,
            feature_cols=spin_feature_cols,
            out_path=MODEL_PLOTS / "spin_in_vs_out_with_model.png",
            title="Inbound vs outbound spin with group specific models",
            xlabel="Inbound spin (rad/s)",
            ylabel="Outbound spin (rad/s)",
        )

    print(f"\nSaved all regression plots to {MODEL_PLOTS.resolve()}.")

    metrics_rows = []

    # after speed model fit + y_speed_pred
    m_speed = regression_metrics(y_speed, y_speed_pred)
    metrics_rows.append({"target": "speed_out_mps", **m_speed})

    # after spin model fit + y_spin_pred (only if spin model exists)
    if spin_model is not None and spin_features_df is not None:
        m_spin = regression_metrics(y_spin, y_spin_pred)
        metrics_rows.append({"target": str(spin_out_col), **m_spin})

    # at the end of main (after plots, before final print)
    save_metrics_table(metrics_rows, R2_TABLE_CSV)


if __name__ == "__main__":
    main()

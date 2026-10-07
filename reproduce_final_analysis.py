# -*- coding: utf-8 -*-
"""
Reproduce the final analysis and figures for the HfOx-AlOx-PVP MOBO study.

Input
-----
One Excel file with the exact columns used in the public dataset:
    Muestras
    AlOx
    HfOx
    PVP
    C (nF/cm2)
    C_std
    -log J
    I_std
    Class
    Feasibility

Coding used in the dataset
--------------------------
Clasificacion:
    0 = Opaque
    1 = Precipitated
    2 = Suitable

Feasibility:
    0 = Infeasible
    1 = Feasible

Samples 1-30 are the optimization/training set.
Samples 31-33 are independent validation samples and are NOT used for training.

Analyses reproduced
-------------------
1) Independent fixed-noise GPR models for capacitance and -log(J)
2) Final ternary response maps
3) Predictive uncertainty maps
4) Three-class GP processability classifier
5) Binary GP feasibility classifier and P_feas constraint
6) Feasibility-constrained final Pareto front and hypervolume
7) Leave-one-out cross-validation
8) SHAP summary and dependence plots
9) Independent validation plot for samples 31-33

The script does NOT rerun the historical experimental campaign or regenerate
Rounds 1-4. It reconstructs the final models from the complete training dataset,
which is sufficient to reproduce the final model-based analyses and figures.

Recommended environment matching the manuscript:
    BoTorch 0.8.5
    GPyTorch 1.10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gpytorch
import shap

from sklearn.metrics import r2_score, mean_squared_error

from botorch.models.gp_regression import FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal


# ---------------------------------------------------------------------------
# Reproducibility settings
# ---------------------------------------------------------------------------

torch.set_default_dtype(torch.float64)
torch.manual_seed(13)
np.random.seed(13)

INPUT_NAMES = ["AlOx", "HfOx", "PVP"]
OUTPUT_NAMES = ["C (nF/cm2)", "-log J"]
STD_NAMES = ["C_std", "I_std"]

TRAINING_LAST_SAMPLE = 30
VALIDATION_FIRST_SAMPLE = 31

COMPONENT_MIN = 0.10
COMPONENT_MAX = 0.80
MESH_STEP = 0.01
P_FEAS_THRESHOLD = 0.60

CLASSIFIER_TRAINING_ITER = 500
CLASSIFIER_LR = 0.02

CLASS_NAMES = {
    0: "Opaque",
    1: "Precipitated",
    2: "Suitable",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(excel_file: str | Path):
    """Load and validate the public Excel dataset."""
    df = pd.read_excel(excel_file)

    required = [
        "Sample", *INPUT_NAMES, *OUTPUT_NAMES, *STD_NAMES,
        "Class", "Feasibility"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna(subset=["Sample"]).copy()
    df["Sample"] = df["Sample"].astype(int)

    # Check ternary constraint.
    composition_sum = df[INPUT_NAMES].sum(axis=1)
    if not np.allclose(composition_sum, 1.0, atol=2e-4):
        bad = df.loc[np.abs(composition_sum - 1.0) > 2e-4,
                     ["Sample", *INPUT_NAMES]]
        raise ValueError(
            "Some compositions do not satisfy AlOx + HfOx + PVP = 1:\n"
            + bad.to_string(index=False)
        )

    # Training = samples 1-30. Validation = samples 31-33.
    training = df[df["Sample"] <= TRAINING_LAST_SAMPLE].copy()
    validation = df[df["Sample"] >= VALIDATION_FIRST_SAMPLE].copy()

    # GPR uses only experimentally feasible/Suitable formulations.
    regression = training[
        (training["Feasibility"] == 1) &
        (training["Class"] == 2)
    ].copy()

    if len(regression) != 23:
        print(
            f"Warning: expected 23 feasible regression samples, found {len(regression)}."
        )

    return df, training, regression, validation
# ---------------------------------------------------------------------------
# Fixed-noise Gaussian-process regression
# ---------------------------------------------------------------------------

def fit_single_gp(X, y, y_std):
    """
    Fit one fixed-noise GPR model using the same workflow as the original code:
    - normalize X using the observed feasible regression-data bounds
    - standardize y
    - FixedNoiseGP
    - maximize exact marginal log likelihood
    """
    X_bounds = torch.stack((X.min(dim=0).values, X.max(dim=0).values))
    X_norm = normalize(X, bounds=X_bounds)

    y_mean = y.mean()
    y_scale = y.std()
    y_standardized = ((y - y_mean) / y_scale).unsqueeze(-1)

    variance_standardized = ((y_std ** 2) / (y_scale ** 2)).unsqueeze(-1)

    model = FixedNoiseGP(
        X_norm,
        y_standardized,
        variance_standardized
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()

    return {
        "model": model,
        "X_bounds": X_bounds,
        "y_mean": y_mean,
        "y_std": y_scale,
    }

def fit_regression_models(regression_df):
    X = torch.tensor(
        regression_df[INPUT_NAMES].to_numpy(),
        dtype=torch.float64
    )

    models = []
    for output_name, std_name in zip(OUTPUT_NAMES, STD_NAMES):
        y = torch.tensor(
            regression_df[output_name].to_numpy(),
            dtype=torch.float64
        )
        y_std = torch.tensor(
            regression_df[std_name].to_numpy(),
            dtype=torch.float64
        )
        models.append(fit_single_gp(X, y, y_std))

    return X, models

def predict_gp(model_info, X, batch_size=256):
    mean_chunks = []
    uncertainty_chunks = []

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        X_norm = normalize(X_batch, bounds=model_info["X_bounds"])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model_info["model"].posterior(X_norm)
            mean_std = posterior.mean.squeeze(-1)
            variance_std = posterior.variance.squeeze(-1)

        mean = (
            mean_std * model_info["y_std"]
            + model_info["y_mean"]
        )
        uncertainty = (
            torch.sqrt(variance_std.clamp_min(0.0))
            * model_info["y_std"]
        )

        mean_chunks.append(mean.detach().cpu())
        uncertainty_chunks.append(uncertainty.detach().cpu())

    return torch.cat(mean_chunks), torch.cat(uncertainty_chunks)

def predict_all_regression(models, X):
    means = []
    uncertainties = []

    for model_info in models:
        mean, uncertainty = predict_gp(model_info, X)
        means.append(mean)
        uncertainties.append(uncertainty)

    return (
        torch.stack(means, dim=1),
        torch.stack(uncertainties, dim=1)
    )

# ---------------------------------------------------------------------------
# Gaussian-process classification
# ---------------------------------------------------------------------------

class GPClassifier(ApproximateGP):
    def __init__(self, inducing_points, num_classes):
        self.num_classes = num_classes

        inducing_points = inducing_points.unsqueeze(0).repeat(
            num_classes, 1, 1
        )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_classes])
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(
            batch_shape=torch.Size([num_classes])
        )

        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=inducing_points.shape[-1],
                batch_shape=torch.Size([num_classes])
            ),
            batch_shape=torch.Size([num_classes])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def fit_gp_classifier(X, y, num_classes):
    X_bounds = torch.stack((X.min(dim=0).values, X.max(dim=0).values))
    X_norm = normalize(X, bounds=X_bounds)

    model = GPClassifier(X_norm, num_classes).double()

    likelihood = SoftmaxLikelihood(
        num_classes=num_classes,
        num_features=num_classes
    ).double()

    mll = VariationalELBO(
        likelihood,
        model,
        num_data=len(y)
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=CLASSIFIER_LR
    )

    model.train()
    likelihood.train()

    for _ in range(CLASSIFIER_TRAINING_ITER):
        optimizer.zero_grad()
        loss = -mll(model(X_norm), y).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return {
        "model": model,
        "likelihood": likelihood,
        "X_bounds": X_bounds,
        "num_classes": num_classes,
    }

def fit_classification_models(training_df):
    X = torch.tensor(
        training_df[INPUT_NAMES].to_numpy(),
        dtype=torch.float64
    )

    y_class = torch.tensor(
        training_df["Class"].to_numpy(),
        dtype=torch.long
    )

    y_feas = torch.tensor(
        training_df["Feasibility"].to_numpy(),
        dtype=torch.long
    )

    class_model = fit_gp_classifier(X, y_class, num_classes=3)
    feasibility_model = fit_gp_classifier(X, y_feas, num_classes=2)

    return X, class_model, feasibility_model

def predict_classifier(model_info, X, batch_size=2000):
    probability_chunks = []

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        X_norm = normalize(X_batch, bounds=model_info["X_bounds"])

        with torch.inference_mode():
            latent = model_info["model"](X_norm)
            prediction = model_info["likelihood"](latent)
            probabilities = prediction.probs.mean(dim=0)

        probability_chunks.append(probabilities.detach().cpu())

    probabilities = torch.cat(probability_chunks, dim=0)
    predicted_class = probabilities.argmax(dim=-1)

    return predicted_class, probabilities

# ---------------------------------------------------------------------------
# Ternary mesh and plotting utilities
# ---------------------------------------------------------------------------

def ternary_mesh(step=MESH_STEP, restrict_search_domain=False):
    tol = 1e-12
    xs = torch.arange(0.0, 1.0 + tol, step, dtype=torch.float64)
    a, b = torch.meshgrid(xs, xs, indexing="ij")
    c = 1.0 - a - b
    mask = c >= -tol

    X_mesh = torch.stack([a[mask], b[mask], c[mask].clamp_min(0.0)], dim=1)

    if restrict_search_domain:
        domain_mask = torch.all((X_mesh >= COMPONENT_MIN - tol) & (X_mesh <= COMPONENT_MAX + tol), dim=1)
        X_mesh = X_mesh[domain_mask]

    return X_mesh

def ternary_to_xy(X):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    b = X[:, 1]
    c = X[:, 2]

    x = b + 0.5 * c
    y = (np.sqrt(3) / 2) * c

    return x, y

def draw_ternary_axes(ax, tick_step=0.2, grid_step=0.1):
    h = np.sqrt(3) / 2

    triangle = np.array([
        [0, 0],
        [1, 0],
        [0.5, h],
        [0, 0]
    ])
    ax.plot(
        triangle[:, 0],
        triangle[:, 1],
        color="black",
        linewidth=1.5
    )

    for t in np.arange(grid_step, 1.0, grid_step):
        u = np.linspace(0, 1 - t, 80)

        lines = [
            np.stack(
                [np.full_like(u, t), u, 1 - t - u],
                axis=1
            ),
            np.stack(
                [u, np.full_like(u, t), 1 - t - u],
                axis=1
            ),
            np.stack(
                [u, 1 - t - u, np.full_like(u, t)],
                axis=1
            )
        ]

        for line in lines:
            xx, yy = ternary_to_xy(line)
            ax.plot(
                xx,
                yy,
                color="black",
                alpha=0.15,
                linewidth=0.6
            )

    ax.text(
        -0.03, -0.07, INPUT_NAMES[0],
        ha="left", va="top", fontsize=15
    )
    ax.text(
        1.03, -0.07, INPUT_NAMES[1],
        ha="right", va="top", fontsize=15
    )
    ax.text(
        0.5, h + 0.05, INPUT_NAMES[2],
        ha="center", va="bottom", fontsize=15
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def plot_ternary_map(
    X_mesh,
    z,
    experimental_X,
    title,
    colorbar_label,
    filename,
    output_dir,
    levels=20,
    fixed_range=None
):
    xg, yg = ternary_to_xy(X_mesh)
    xp, yp = ternary_to_xy(experimental_X)

    fig, ax = plt.subplots(figsize=(8, 7))

    if fixed_range is None:
        contour = ax.tricontourf(
            xg, yg, np.asarray(z),
            levels=levels
        )
    else:
        contour = ax.tricontourf(
            xg, yg, np.asarray(z),
            levels=np.linspace(
                fixed_range[0],
                fixed_range[1],
                levels + 1
            ),
            vmin=fixed_range[0],
            vmax=fixed_range[1]
        )

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label, fontsize=13)

    ax.scatter(
        xp, yp,
        s=55,
        edgecolors="white",
        linewidths=0.8
    )

    draw_ternary_axes(ax)
    ax.set_title(title, fontsize=15, pad=20)

    plt.tight_layout()
    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# ---------------------------------------------------------------------------
# Final response and uncertainty maps
# ---------------------------------------------------------------------------

def final_response_maps(
    regression_X,
    regression_models,
    output_dir
):
    X_mesh = ternary_mesh(
        step=MESH_STEP,
        restrict_search_domain=False
    )

    means, uncertainty = predict_all_regression(
        regression_models,
        X_mesh
    )

    for j, output_name in enumerate(OUTPUT_NAMES):
        short = "C" if j == 0 else "logJ"

        plot_ternary_map(
            X_mesh,
            means[:, j].cpu().numpy(),
            regression_X,
            title=f"GPR-predicted {output_name}",
            colorbar_label=output_name,
            filename=f"Final_GPR_{short}.png",
            output_dir=output_dir,
            levels=20
        )

        plot_ternary_map(
            X_mesh,
            uncertainty[:, j].cpu().numpy(),
            regression_X,
            title=f"Predictive uncertainty - {output_name}",
            colorbar_label="Predictive uncertainty",
            filename=f"Uncertainty_{short}.png",
            output_dir=output_dir,
            levels=20
        )

    return X_mesh, means, uncertainty

# ---------------------------------------------------------------------------
# Processability classification and feasibility constraint
# ---------------------------------------------------------------------------

def classification_maps(
    classification_X,
    class_model,
    feasibility_model,
    output_dir
):
    X_mesh = ternary_mesh(
        step=MESH_STEP,
        restrict_search_domain=False
    )

    predicted_class, class_probability = predict_classifier(
        class_model,
        X_mesh
    )

    _, feasibility_probability = predict_classifier(
        feasibility_model,
        X_mesh
    )

    plot_ternary_map(
        X_mesh,
        predicted_class.cpu().numpy(),
        classification_X,
        title="GP processability classification",
        colorbar_label="Predicted class",
        filename="Processability_Classification.png",
        output_dir=output_dir,
        levels=3,
        fixed_range=(-0.5, 2.5)
    )

    plot_ternary_map(
        X_mesh,
        feasibility_probability[:, 1].cpu().numpy(),
        classification_X,
        title="Probability of feasibility",
        colorbar_label="Pfeas",
        filename="Feasibility_Probability.png",
        output_dir=output_dir,
        levels=20,
        fixed_range=(0.0, 1.0)
    )

    return (
        X_mesh,
        predicted_class,
        class_probability,
        feasibility_probability
    )

# ---------------------------------------------------------------------------
# Pareto front and hypervolume
# ---------------------------------------------------------------------------

def pareto_analysis(
    regression_models,
    regression_df,
    feasibility_model,
    output_dir
):
    # Candidate domain used in the manuscript: each fraction 0.10-0.80.
    X_domain = ternary_mesh(
        step=MESH_STEP,
        restrict_search_domain=False
    )

    means, _ = predict_all_regression(
        regression_models,
        X_domain
    )

    _, feasibility_probability = predict_classifier(
        feasibility_model,
        X_domain
    )

    feasible_mask = (
        feasibility_probability[:, 1]
        >= P_FEAS_THRESHOLD
    )

    X_feasible = X_domain[feasible_mask]
    Y_feasible = means[feasible_mask]

    model_pareto_mask = is_non_dominated(Y_feasible)
    X_model_pareto = X_feasible[model_pareto_mask]
    Y_model_pareto = Y_feasible[model_pareto_mask]

    Y_exp = torch.tensor(
        regression_df[OUTPUT_NAMES].to_numpy(),
        dtype=torch.float64
    )

    exp_pareto_mask = is_non_dominated(Y_exp)
    Y_exp_pareto = Y_exp[exp_pareto_mask]

    reference_point = torch.tensor(
        [0.0, 0.0],
        dtype=torch.float64
    )

    experimental_hv = DominatedPartitioning(
        ref_point=reference_point,
        Y=Y_exp
    ).compute_hypervolume().item()

    predicted_hv = DominatedPartitioning(
        ref_point=reference_point,
        Y=Y_model_pareto
    ).compute_hypervolume().item()

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        Y_model_pareto[:, 0],
        Y_model_pareto[:, 1],
        s=40,
        label="Predicted Pareto front"
    )

    ax.scatter(
        Y_exp[:, 0],
        Y_exp[:, 1],
        marker="x",
        s=55,
        label="Experimental data"
    )

    ax.scatter(
        Y_exp_pareto[:, 0],
        Y_exp_pareto[:, 1],
        s=80,
        edgecolor="black",
        label="Experimental Pareto points"
    )

    ax.set_xlabel(OUTPUT_NAMES[0], fontsize=14)
    ax.set_ylabel(OUTPUT_NAMES[1], fontsize=14)

    ax.text(
        0.04,
        0.04,
        (
            f"Experimental HV = {experimental_hv:.1f}\n"
            f"Predicted HV = {predicted_hv:.1f}"
        ),
        transform=ax.transAxes,
        fontsize=11
    )

    ax.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "Final_Pareto_Front.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    pareto_df = pd.DataFrame(
        X_model_pareto.cpu().numpy(),
        columns=INPUT_NAMES
    )
    pareto_df["Pred_C"] = (
        Y_model_pareto[:, 0]
        .cpu()
        .numpy()
    )
    pareto_df["Pred_logJ"] = (
        Y_model_pareto[:, 1]
        .cpu()
        .numpy()
    )

    pareto_df.to_csv(
        output_dir / "Predicted_Pareto_Front.csv",
        index=False
    )

    return experimental_hv, predicted_hv

# ---------------------------------------------------------------------------
# Leave-one-out validation
# ---------------------------------------------------------------------------

def leave_one_out(regression_df, output_dir):
    measured = []
    predicted = []

    for leave_out in range(len(regression_df)):
        train_fold = regression_df.drop(
            regression_df.index[leave_out]
        )

        X_fold, models_fold = fit_regression_models(
            train_fold
        )

        test_row = regression_df.iloc[[leave_out]]
        X_test = torch.tensor(
            test_row[INPUT_NAMES].to_numpy(),
            dtype=torch.float64
        )

        prediction, _ = predict_all_regression(
            models_fold,
            X_test
        )

        measured.append(
            test_row[OUTPUT_NAMES]
            .to_numpy()
            .flatten()
        )

        predicted.append(
            prediction
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

    measured = np.asarray(measured)
    predicted = np.asarray(predicted)

    metrics = []

    results = regression_df[
        ["Sample", *INPUT_NAMES]
    ].reset_index(drop=True)

    for j, output_name in enumerate(OUTPUT_NAMES):
        y_true = measured[:, j]
        y_pred = predicted[:, j]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        metrics.append({
            "Objective": output_name,
            "R2": r2,
            "RMSE": rmse
        })

        results[
            f"{output_name}_measured"
        ] = y_true

        results[
            f"{output_name}_predicted"
        ] = y_pred

        lim_min = min(
            y_true.min(),
            y_pred.min()
        )
        lim_max = max(
            y_true.max(),
            y_pred.max()
        )

        margin = 0.05 * (
            lim_max - lim_min
        )
        lim_min -= margin
        lim_max += margin

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.scatter(
            y_true,
            y_pred,
            s=80,
            edgecolor="black"
        )

        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            "--",
            color="black",
            linewidth=1.5
        )

        ax.set_xlim(
            lim_min,
            lim_max
        )
        ax.set_ylim(
            lim_min,
            lim_max
        )

        ax.set_xlabel(
            f"Measured {output_name}",
            fontsize=14
        )
        ax.set_ylabel(
            f"LOO predicted {output_name}",
            fontsize=14
        )

        ax.text(
            0.05,
            0.95,
            (
                f"$R^2$ = {r2:.2f}\n"
                f"RMSE = {rmse:.2f}"
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=12
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        plt.tight_layout()

        short = (
            "C"
            if j == 0
            else "logJ"
        )

        plt.savefig(
            output_dir / f"LOO_{short}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    results.to_csv(
        output_dir / "LOO_predictions.csv",
        index=False
    )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        output_dir / "LOO_metrics.csv",
        index=False
    )

    print("\nLOO metrics")
    print(metrics_df.to_string(index=False))

    return metrics_df

# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def shap_analysis(
    regression_X,
    regression_models,
    output_dir
):
    input_data = (
        regression_X
        .detach()
        .cpu()
        .numpy()
    )

    X_df = pd.DataFrame(
        input_data,
        columns=INPUT_NAMES
    )

    for objective_index, output_name in enumerate(
        OUTPUT_NAMES
    ):
        model_info = regression_models[
            objective_index
        ]

        def predictor(input_array):
            X_predict = torch.tensor(
                np.asarray(input_array),
                dtype=torch.float64
            )

            prediction, _ = predict_gp(
                model_info,
                X_predict
            )

            return (
                prediction
                .detach()
                .cpu()
                .numpy()
            )

        n_background = min(
            10,
            len(input_data)
        )

        background = shap.kmeans(
            input_data,
            n_background
        )

        explainer = shap.KernelExplainer(
            predictor,
            background
        )

        shap_values = explainer.shap_values(
            input_data,
            nsamples=100
        )

        short = (
            "C"
            if objective_index == 0
            else "logJ"
        )

        shap.summary_plot(
            shap_values,
            input_data,
            feature_names=INPUT_NAMES,
            show=False
        )

        plt.title(
            f"SHAP - {output_name}",
            fontsize=14
        )

        plt.tight_layout()

        plt.savefig(
            output_dir / f"SHAP_Summary_{short}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        for variable_index, variable_name in enumerate(
            INPUT_NAMES
        ):
            shap.dependence_plot(
                variable_index,
                shap_values,
                X_df,
                interaction_index=variable_index,
                show=False
            )

            plt.title(
                f"{output_name} - {variable_name}"
            )

            plt.tight_layout()

            plt.savefig(
                output_dir
                / (
                    f"SHAP_Dependence_"
                    f"{short}_{variable_name}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

# ---------------------------------------------------------------------------
# Independent validation samples 31-33
# ---------------------------------------------------------------------------

def validation_analysis(
    regression_models,
    regression_df,
    validation_df,
    output_dir
):
    X_validation = torch.tensor(
        validation_df[INPUT_NAMES].to_numpy(),
        dtype=torch.float64
    )

    prediction, _ = predict_all_regression(
        regression_models,
        X_validation
    )

    predicted = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )

    experimental = validation_df[
        OUTPUT_NAMES
    ].to_numpy()

    validation_output = validation_df[
        ["Sample", *INPUT_NAMES, *OUTPUT_NAMES]
    ].copy()

    validation_output["Pred_C"] = predicted[:, 0]
    validation_output[
        "Pred_-logJ"
    ] = predicted[:, 1]

    validation_output.to_csv(
        output_dir / "Validation_predictions.csv",
        index=False
    )

    Y_training = regression_df[
        OUTPUT_NAMES
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        Y_training[:, 0],
        Y_training[:, 1],
        alpha=0.35,
        label="Training data"
    )

    for i, sample in enumerate(
        validation_df["Sample"].to_numpy()
    ):
        ax.scatter(
            predicted[i, 0],
            predicted[i, 1],
            marker="D",
            facecolors="none",
            edgecolors="black",
            s=90
        )

        ax.scatter(
            experimental[i, 0],
            experimental[i, 1],
            s=80
        )

        ax.plot(
            [
                predicted[i, 0],
                experimental[i, 0]
            ],
            [
                predicted[i, 1],
                experimental[i, 1]
            ],
            "--",
            linewidth=1
        )

        ax.annotate(
            str(sample),
            (
                experimental[i, 0],
                experimental[i, 1]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel(
        OUTPUT_NAMES[0],
        fontsize=14
    )
    ax.set_ylabel(
        OUTPUT_NAMES[1],
        fontsize=14
    )

    ax.set_title(
        "Independent validation samples"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "Validation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# ---------------------------------------------------------------------------
# Save final mesh predictions
# ---------------------------------------------------------------------------

def save_final_mesh(
    regression_models,
    class_model,
    feasibility_model,
    output_dir
):
    X_domain = ternary_mesh(
        step=MESH_STEP,
        restrict_search_domain=True
    )

    regression_mean, regression_uncertainty = (
        predict_all_regression(
            regression_models,
            X_domain
        )
    )

    predicted_class, class_probability = (
        predict_classifier(
            class_model,
            X_domain
        )
    )

    _, feasibility_probability = (
        predict_classifier(
            feasibility_model,
            X_domain
        )
    )

    mesh_df = pd.DataFrame(
        X_domain.cpu().numpy(),
        columns=INPUT_NAMES
    )

    mesh_df["Pred_C"] = (
        regression_mean[:, 0]
        .cpu()
        .numpy()
    )

    mesh_df["Pred_logJ"] = (
        regression_mean[:, 1]
        .cpu()
        .numpy()
    )

    mesh_df["Uncertainty_C"] = (
        regression_uncertainty[:, 0]
        .cpu()
        .numpy()
    )

    mesh_df["Uncertainty_logJ"] = (
        regression_uncertainty[:, 1]
        .cpu()
        .numpy()
    )

    mesh_df["Predicted_Class"] = (
        predicted_class
        .cpu()
        .numpy()
    )

    for class_id in range(3):
        mesh_df[
            f"P_Class_{class_id}"
        ] = (
            class_probability[:, class_id]
            .cpu()
            .numpy()
        )

    mesh_df["P_feas"] = (
        feasibility_probability[:, 1]
        .cpu()
        .numpy()
    )

    mesh_df["Feasible_constraint"] = (
        mesh_df["P_feas"]
        >= P_FEAS_THRESHOLD
    ).astype(int)

    mesh_df.to_csv(
        output_dir / "Final_mesh_predictions.csv",
        index=False
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the final HfOx-AlOx-PVP "
            "GPR, classification, feasibility, "
            "LOO, SHAP, Pareto, and validation analyses."
        )
    )

    parser.add_argument(
        "excel",
        nargs="?",
        default="Data.xlsx",
        help="Input Excel file (default: Data.xlsx)"
    )

    parser.add_argument(
        "--output",
        default="results",
        help="Output folder (default: results)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        full_df,
        training_df,
        regression_df,
        validation_df
    ) = load_data(args.excel)

    print(
        f"Total samples: {len(full_df)}"
    )
    print(
        f"Optimization/training samples: "
        f"{len(training_df)}"
    )
    print(
        f"Feasible GPR samples: "
        f"{len(regression_df)}"
    )
    print(
        f"Independent validation samples: "
        f"{len(validation_df)}"
    )

    # Regression
    regression_X, regression_models = (
        fit_regression_models(
            regression_df
        )
    )

    # Classification and feasibility
    (
        classification_X,
        class_model,
        feasibility_model
    ) = fit_classification_models(
        training_df
    )

    # Final response + uncertainty maps
    final_response_maps(
        regression_X,
        regression_models,
        output_dir
    )

    # Processability + feasibility maps
    classification_maps(
        classification_X,
        class_model,
        feasibility_model,
        output_dir
    )

    # Final Pareto analysis
    experimental_hv, predicted_hv = (
        pareto_analysis(
            regression_models,
            regression_df,
            feasibility_model,
            output_dir
        )
    )

    print(
        f"\nExperimental hypervolume: "
        f"{experimental_hv:.2f}"
    )
    print(
        f"Predicted hypervolume: "
        f"{predicted_hv:.2f}"
    )

    # LOO
    leave_one_out(
        regression_df,
        output_dir
    )

    # SHAP
    shap_analysis(
        regression_X,
        regression_models,
        output_dir
    )

    # Independent samples 31-33
    validation_analysis(
        regression_models,
        regression_df,
        validation_df,
        output_dir
    )

    # Numerical predictions for the accessible domain
    save_final_mesh(
        regression_models,
        class_model,
        feasibility_model,
        output_dir
    )

    print(
        f"\nAnalysis complete. "
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

if __name__ == "__main__":
    main()

# HfOx-AlOx-PVP MOBO: final-analysis reproduction code

This repository reproduces the **final model-based analyses and figures** used in the manuscript:

**Multi-Objective Bayesian Optimization of HfAlOx-PVP Hybrid Gate Dielectrics for Thin-Film Transistor Applications**

## Files

- `Data.xlsx` — experimental dataset
- `reproduce_final_analysis.py` — complete reproduction script
- `requirements.txt` — Python dependencies

## Dataset

The Excel file contains 33 formulations.

- Samples **1-30**: optimization/training data
- Samples **31-33**: independent validation data, excluded from all model training

The relevant columns are:

| Column | Meaning |
|---|---|
| `Sample` | Sample number |
| `AlOx`, `HfOx`, `PVP` | Ternary precursor fractions |
| `C (nF/cm2)` | Areal capacitance at 1 kHz |
| `C_std` | Experimental SD of capacitance |
| `-log J` | Leakage-current objective |
| `I_std` | Experimental SD of `-log J` |
| `Class` | Processability class |
| `Feasibility` | Binary feasibility label |

### Processability labels

- `0` = Opaque
- `1` = Precipitated
- `2` = Suitable

### Feasibility labels

- `0` = Infeasible
- `1` = Feasible

Only samples labeled `Suitable` and `Factibilidad = 1` are included in the GPR models.

## Analyses reproduced

The script performs:

1. Independent fixed-noise Gaussian-process regression (GPR) for capacitance and `-log J`
2. Final ternary GPR response maps
3. Predictive uncertainty maps
4. Three-class Gaussian-process processability classification
5. Binary GP feasibility classification
6. Feasibility constraint using `P_feas >= 0.60`
7. Final Pareto front and dominated hypervolume
8. Leave-one-out (LOO) cross-validation
9. SHAP summary plots and individual dependence plots
10. Independent validation using samples 31-33

The candidate domain is defined by:

- `AlOx`, `HfOx`, `PVP` fractions between **0.10 and 0.80**
- ternary sum = 1
- mesh spacing = **0.01**

This gives **2556 candidate compositions** in the accessible discretized search space.

## Running the analysis

Place `Data.xlsx` in the same directory as the Python script and run:

```bash
python reproduce_final_analysis.py Data.xlsx
```

or simply:

```bash
python reproduce_final_analysis.py
```

The second command assumes the input file is named `Data.xlsx`.

All generated files are saved in:

```text
results/
```

## Main outputs

The output folder includes:

- `Final_GPR_C.png`
- `Final_GPR_logJ.png`
- `Uncertainty_C.png`
- `Uncertainty_logJ.png`
- `Processability_Classification.png`
- `Feasibility_Probability.png`
- `Final_Pareto_Front.png`
- `LOO_C.png`
- `LOO_logJ.png`
- `SHAP_Summary_C.png`
- `SHAP_Summary_logJ.png`
- SHAP dependence plots
- `Validation.png`
- `LOO_metrics.csv`
- `LOO_predictions.csv`
- `Validation_predictions.csv`
- `Predicted_Pareto_Front.csv`
- `Final_mesh_predictions.csv`

## Scope

The purpose of this script is to reproduce the **final analysis** from the completed experimental dataset.

The historical sequential selection of Rounds 1-4 is not rerun here because those experiments were performed sequentially during the campaign. The public dataset preserves the experimentally selected formulations, while this script reconstructs the final surrogate models, feasibility model, cross-validation, model interpretation, Pareto analysis, and validation reported in the manuscript.

## Software

The manuscript analysis used:

- Python
- BoTorch 0.8.5
- GPyTorch 1.10

For maximum reproducibility, use the versions listed in `requirements.txt`.

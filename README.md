# ENPM667 Project 2 – Double-Pendulum Crane Control

This repository contains the source code, scripts, and figures used to solve ENPM667 Project 2. The project studies a double-pendulum gantry crane model and walks through controllability, LQR design, observability, Luenberger observer synthesis, and LQG output feedback on both the linearized and nonlinear dynamics.

## Environment setup

### Requirements

- Python **3.11** (per `pyproject.toml`)
- `uv` 0.4+ **or** plain `pip`/`venv`
- OS X / Linux with BLAS/LAPACK available (for SciPy) and a working LaTeX-free Matplotlib backend

### Quick start with `uv` (recommended)

```bash
uv venv                       # creates .venv tied to Python 3.11
source .venv/bin/activate     # or .venv\\Scripts\\activate on Windows
uv sync                       # installs runtime deps from pyproject.toml / uv.lock
```

`uv sync` honors the locked versions found in `uv.lock`. Re-run it whenever dependencies change.

#### Daily uv workflow

- Run scripts through the managed env without activating manually:

  ```bash
  uv run python src/component1_C.py
  uv run python src/component1_D.py
  ```

- Add extra packages (dev or runtime) and update `pyproject.toml`/`uv.lock` in one shot:

  ```bash
  uv add black ruff
  uv add --dev pytest
  ```

- Upgrade existing dependencies as needed:

  ```bash
  uv lock --upgrade sympy
  uv sync
  ```

### Alternative: vanilla `pip`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "matplotlib<3.9" "numpy<2" "pandas>=2.3.3" \
            "roboticstoolbox-python>=1.1.1" "scipy<1.13" \
            "spatialmath-python>=1.1.15" "sympy>=1.14.0"
```

Developer tooling (optional):

```bash
pip install "black>=25.9.0" "ruff>=0.13.2" "pytest>=8.4.2"
```

## Running the components

All scripts assume the environment above is active and are invoked from the project root.

### Component 1 – C) Controllability

```bash
python src/component1_C.py
```

Outputs the symbolic controllability matrix, its rank, determinant, and an optional numeric rank check. Edit the `subs_example` dictionary to try alternate physical parameters.

### Component 1 – D) LQR search and simulations

```bash
python src/component1_D.py
```

- Sweeps candidate diagonal `Q` and scalar `R`, computes LQR gains, and simulates both the linearized model (`A - B K`) and the nonlinear crane dynamics.
- Saves a comparison table (`component1_D_comparison_table.png`) plus best-design state/control plots (`component1_D_best_{x,theta1,theta2,u}.png`). Copies of these artifacts live under `fig/component1_D/`.
- Adjust the `Q_candidates`, `R_candidates`, initial condition `x0`, or time span `t_span` to explore different cases.

### Component 2 – E) Observability checks

```bash
python src/component2_E_observability.py
```

Runs the dual controllability test for four output selections, prints symbolic ranks, and repeats the test numerically for the project’s nominal parameters. Modify `C_list` to try new sensor groupings.

### Component 2 – F) Luenberger observers

```bash
python src/component2_F_observer_design.py
```

- Reuses Design #2’s LQR gain, builds observers for each observable output option, and places poles using `scipy.signal.place_poles`.
- Simulates the coupled plant-observer system for both the linearized and nonlinear models, reporting integral squared error, peak estimation errors, and control magnitudes.
- Saves a CSV-like table image (`component2_F_observer_comparison_table.png`) and best-case time-series figures (`component2_F_case{n}_{lin,nonlin}.png`). Curated copies are stored under `fig/component2_F/`.
- Tweak `speed_factors`, the initial states `x0`, or simulation length `t_final` to examine different observer aggressiveness levels.

### Component 2 – G) LQG controller

```bash
python src/component2_G_LQG.py
```

Constructs an LQG controller (Design #2 LQR + continuous-time Kalman filter for `y = (x, θ1, θ2)`), injects process/measurement noise, and simulates both the linearized and nonlinear plants via RK4 integration. Generates four figures showing state estimates and control effort:

- `component2_G_LQG_y_xth1th2_linear_true_vs_hat.png`
- `component2_G_LQG_y_xth1th2_linear_states.png`
- `component2_G_LQG_y_xth1th2_nonlinear_true_vs_hat.png`
- `component2_G_LQG_y_xth1th2_nonlinear_states.png`

Copies of the figures intended for the report live under `fig/component2_G/`. Adjust noise levels (`sigma_w`, `sigma_x_m`, `sigma_th1_deg`, `sigma_th2_deg`), actuator saturation (`U_MAX`), or seeds to study robustness.

## Tips for extending or grading

- Each script is self-contained; no package install is required beyond dependencies listed in `pyproject.toml`.
- Numerical experiments that produce PNGs write to the project root. Move them into the appropriate `fig/` subfolder before committing, or update the scripts’ `plt.savefig` paths if you want them to land there directly.
- There is no automated test suite. To verify results, re-run the scripts after making parameter/code changes and inspect the console outputs and regenerated figures.
- Use `ruff` and `black` from the optional dev dependencies to keep code formatting consistent when modifying the scripts.

## License

The project is distributed under the [Apache License 2.0](LICENSE). Refer to that file for details on reuse and attribution.

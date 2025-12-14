# component2_F_observer_design.py
#
# Luenberger observers for the crane system (Component 2 – Part F)
# - Uses LQR K from Component 1 D (Design #2)
# - Designs observers for each observable output using pole placement
# - Simulates closed-loop (plant + LQR + observer) for both
#   linearized and nonlinear systems
# - Compares designs using simple metrics and saves summary figures

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
# 1. Parameters and linearized model (same structure as before)
# ------------------------------------------------------------

M = 1000.0   # cart mass [kg]
m1 = 100.0   # pendulum 1 mass [kg]
m2 = 100.0   # pendulum 2 mass [kg]
l1 = 20.0    # pendulum 1 length [m]
l2 = 10.0    # pendulum 2 length [m]
g  = 9.81    # gravity [m/s^2]

# State: x = [ x, xdot, theta1, theta1dot, theta2, theta2dot ]'
A = np.array([
    [0.0, 1.0,             0.0,                               0.0,             0.0,                               0.0],
    [0.0, 0.0,   g*m1/M,           0.0,             g*m2/M,           0.0],
    [0.0, 0.0,             0.0,                               1.0,             0.0,                               0.0],
    [0.0, 0.0, -g*(M + m1)/(M*l1), 0.0,          -g*m2/(M*l1),        0.0],
    [0.0, 0.0,             0.0,                               0.0,             0.0,                               1.0],
    [0.0, 0.0,   -g*m1/(M*l2),     0.0, -g*(M + m2)/(M*l2),   0.0]
], dtype=float)

B = np.array([
    [0.0],
    [1.0/M],
    [0.0],
    [-1.0/(M*l1)],
    [0.0],
    [-1.0/(M*l2)]
], dtype=float)

n = A.shape[0]


# ------------------------------------------------------------
# 2. LQR state-feedback controller (Design #2 from Component 1 D)
# ------------------------------------------------------------

Qdiag = np.array([1.0, 0.1, 10.0, 0.5, 10.0, 0.5])
Q = np.diag(Qdiag)
R = np.array([[0.01]])

P = la.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

eig_cl = la.eigvals(A - B @ K)
print("Closed-loop eigenvalues (A - B K):")
print(eig_cl)
print()


# ------------------------------------------------------------
# 3. Nonlinear dynamics
# ------------------------------------------------------------

params = dict(M=M, m1=m1, m2=m2, l1=l1, l2=l2, g=g)

def crane_nonlinear_rhs(t, x, u, p):
    """
    Nonlinear continuous-time dynamics of the crane system.

    x = [x, xdot, theta1, theta1dot, theta2, theta2dot]
    u = horizontal force [N]
    """
    M = p["M"]; m1 = p["m1"]; m2 = p["m2"]
    l1 = p["l1"]; l2 = p["l2"]; g = p["g"]

    x_pos, xdot, th1, th1dot, th2, th2dot = x

    D = M + m1*np.sin(th1)**2 + m2*np.sin(th2)**2

    num2 = (u
            + m1*(g*np.cos(th1) + l1*th1dot**2)*np.sin(th1)
            + m2*(g*np.cos(th2) + l2*th2dot**2)*np.sin(th2))

    x2dot = num2 / D
    x4dot = -(x2dot*np.cos(th1) + g*np.sin(th1))/l1
    x6dot = -(x2dot*np.cos(th2) + g*np.sin(th2))/l2

    return np.array([xdot, x2dot, th1dot, x4dot, th2dot, x6dot])


# ------------------------------------------------------------
# 4. Output matrices for observable cases (from Part E)
# ------------------------------------------------------------

C_cases = {
    "y = x": np.array([[1, 0, 0, 0, 0, 0]], float),
    "y = (x, theta2)": np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0]], float),
    "y = (x, theta1, theta2)": np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0]], float),
}


# ------------------------------------------------------------
# 5. Closed-loop (plant + LQR + observer) simulation helpers
# ------------------------------------------------------------

def simulate_closed_loop_observer_linear(A, B, C, K, L,
                                         x0, xhat0, t_eval,
                                         step_amp=1.0):
    """
    Linear plant with LQR + Luenberger observer.

    Plant:      x_dot     = A x + B u
    Observer:   xhat_dot  = A xhat + B u + L (y - yhat)
    Control:    u         = -K xhat + step_amp
    """
    n = A.shape[0]

    def rhs(t, z):
        x = z[:n]
        xhat = z[n:]
        u = float(-K @ xhat + step_amp)
        dx = A @ x + B.flatten() * u
        y = C @ x
        yhat = C @ xhat
        dxhat = A @ xhat + B.flatten() * u + L @ (y - yhat)
        return np.concatenate([dx, dxhat])

    z0 = np.concatenate([x0, xhat0])
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), z0,
                    t_eval=t_eval, rtol=1e-7, atol=1e-9)

    x = sol.y[:n, :].T
    xhat = sol.y[n:, :].T
    u_hist = np.array([-K @ xhat[i] + step_amp for i in range(len(t_eval))])

    return sol.t, x, xhat, u_hist


def simulate_closed_loop_observer_nonlinear(A, B, C, K, L,
                                            x0, xhat0, t_eval,
                                            params, step_amp=1.0):
    """
    Nonlinear plant with the SAME LQR + observer as above.

    Plant:      x_dot     = f_nonlinear(x,u)
    Observer:   xhat_dot  = A xhat + B u + L (y - yhat)
    Control:    u         = -K xhat + step_amp
    """
    n = A.shape[0]

    def rhs(t, z):
        x = z[:n]
        xhat = z[n:]
        u = float(-K @ xhat + step_amp)
        dx = crane_nonlinear_rhs(t, x, u, params)
        y = C @ x
        yhat = C @ xhat
        dxhat = A @ xhat + B.flatten() * u + L @ (y - yhat)
        return np.concatenate([dx, dxhat])

    z0 = np.concatenate([x0, xhat0])
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), z0,
                    t_eval=t_eval, rtol=1e-7, atol=1e-9)

    x = sol.y[:n, :].T
    xhat = sol.y[n:, :].T
    u_hist = np.array([-K @ xhat[i] + step_amp for i in range(len(t_eval))])

    return sol.t, x, xhat, u_hist


def compute_metrics(t, x, xhat, u):
    """
    Simple metrics to compare observers:
    - ISE of estimation error: int ||x - xhat||^2 dt
    - max abs error in x, theta1, theta2
    - max control effort |u|
    """
    err = x - xhat
    e2 = np.sum(err**2, axis=1)
    ise = np.trapz(e2, t)

    max_err = np.max(np.abs(err), axis=0)
    max_u = np.max(np.abs(u))

    return ise, max_err, max_u


# ------------------------------------------------------------
# 6. Sweep observer pole speeds and collect results
# ------------------------------------------------------------

t_final = 40.0
t_eval = np.linspace(0.0, t_final, 2001)

# Initial conditions: some cart offset + small pendulum angles
x0 = np.array([
    0.5,                        # x [m]
    0.0,                        # xdot
    np.deg2rad(5.0),            # theta1 [rad]
    0.0,
    np.deg2rad(-5.0),           # theta2 [rad]
    0.0
])
xhat0 = np.zeros(6)             # observer initial guess = 0

speed_factors = [4.0, 8.0, 12.0, 20,0]      # observer poles' real parts are sf * real(closed-loop poles)

results = []

for out_name, C in C_cases.items():
    for sf in speed_factors:
        # Desired observer poles: make them faster (more negative real parts)
        desired_poles = sf * np.real(eig_cl) + 1j * np.imag(eig_cl)

        # Observer gain L via duality: place poles of (A - L C)
        pp = place_poles(A.T, C.T, desired_poles)
        L = pp.gain_matrix.T

        # Check observer eigenvalues (should match desired_poles)
        eig_obs = la.eigvals(A - L @ C)
        print(f"Output '{out_name}', speed_factor = {sf}")
        print(" Observer eigenvalues(A - L C):")
        print(eig_obs)
        print()

        # Simulate linear and nonlinear closed-loop
        t_lin, x_lin, xhat_lin, u_lin = simulate_closed_loop_observer_linear(
            A, B, C, K, L, x0, xhat0, t_eval
        )
        t_non, x_non, xhat_non, u_non = simulate_closed_loop_observer_nonlinear(
            A, B, C, K, L, x0, xhat0, t_eval, params
        )

        # Metrics
        ise_lin, maxerr_lin, maxu_lin = compute_metrics(t_lin, x_lin, xhat_lin, u_lin)
        ise_non, maxerr_non, maxu_non = compute_metrics(t_non, x_non, xhat_non, u_non)

        results.append({
            "output": out_name,
            "speed_factor": sf,
            "ISE_lin": ise_lin,
            "ISE_non": ise_non,
            "max_err_x_lin": maxerr_lin[0],
            "max_err_theta1_lin": maxerr_lin[2],
            "max_err_theta2_lin": maxerr_lin[4],
            "max_err_x_non": maxerr_non[0],
            "max_err_theta1_non": maxerr_non[2],
            "max_err_theta2_non": maxerr_non[4],
            "max_u_lin": maxu_lin,
            "max_u_non": maxu_non,
        })

print("Finished simulations for all observer designs.\n")

df = pd.DataFrame(results)
print(df)

# Shorter table for the report
cols = [
    "output", "speed_factor",
    "ISE_lin", "ISE_non",
    "max_err_x_non", "max_err_theta1_non", "max_err_theta2_non",
    "max_u_non",
]
df_short = df[cols]

# Save as an image (similar style as Component 1 D)
fig, ax = plt.subplots(figsize=(11, 3))
ax.axis("off")
tbl = ax.table(cellText=df_short.values,
               colLabels=df_short.columns,
               loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1.2, 1.4)
fig.tight_layout()
fig.savefig("component2_F_observer_comparison_table.png", dpi=200)
plt.close(fig)
print("Saved comparison table to component2_F_observer_comparison_table.png\n")


# ------------------------------------------------------------
# 7. For each output, pick the "best" observer and plot trajectories
#    (here: smallest nonlinear ISE)
# ------------------------------------------------------------

best = {}
for row in results:
    key = row["output"]
    if key not in best or row["ISE_non"] < best[key]["ISE_non"]:
        best[key] = row

print("Best designs per output (based on smallest nonlinear ISE):")
for out_name, row in best.items():
    print(f"  {out_name}: speed_factor = {row['speed_factor']}")
print()

def plot_best_case(case_name, C, speed_factor, filename_prefix):
    desired_poles = speed_factor * np.real(eig_cl) + 1j * np.imag(eig_cl)
    L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

    t_lin, x_lin, xhat_lin, u_lin = simulate_closed_loop_observer_linear(
        A, B, C, K, L, x0, xhat0, t_eval
    )
    t_non, x_non, xhat_non, u_non = simulate_closed_loop_observer_nonlinear(
        A, B, C, K, L, x0, xhat0, t_eval, params
    )

    # ----- Linear plant -----
    fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(t_lin, x_lin[:, 0], label="x true")
    axs[0].plot(t_lin, xhat_lin[:, 0], "--", label="x_hat")
    axs[0].set_ylabel("x [m]")
    axs[0].legend()

    axs[1].plot(t_lin, np.rad2deg(x_lin[:, 2]), label="theta1 true")
    axs[1].plot(t_lin, np.rad2deg(xhat_lin[:, 2]), "--", label="theta1_hat")
    axs[1].set_ylabel("theta1 [deg]")
    axs[1].legend()

    axs[2].plot(t_lin, np.rad2deg(x_lin[:, 4]), label="theta2 true")
    axs[2].plot(t_lin, np.rad2deg(xhat_lin[:, 4]), "--", label="theta2_hat")
    axs[2].set_ylabel("theta2 [deg]")
    axs[2].legend()

    axs[3].plot(t_lin, u_lin)
    axs[3].set_ylabel("u [N]")
    axs[3].set_xlabel("time [s]")

    fig.suptitle(f"{case_name} – linear plant")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_lin.png", dpi=200)
    plt.close(fig)

    # ----- Nonlinear plant -----
    fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(t_non, x_non[:, 0], label="x true")
    axs[0].plot(t_non, xhat_non[:, 0], "--", label="x_hat")
    axs[0].set_ylabel("x [m]")
    axs[0].legend()

    axs[1].plot(t_non, np.rad2deg(x_non[:, 2]), label="theta1 true")
    axs[1].plot(t_non, np.rad2deg(xhat_non[:, 2]), "--", label="theta1_hat")
    axs[1].set_ylabel("theta1 [deg]")
    axs[1].legend()

    axs[2].plot(t_non, np.rad2deg(x_non[:, 4]), label="theta2 true")
    axs[2].plot(t_non, np.rad2deg(xhat_non[:, 4]), "--", label="theta2_hat")
    axs[2].set_ylabel("theta2 [deg]")
    axs[2].legend()

    axs[3].plot(t_non, u_non)
    axs[3].set_ylabel("u [N]")
    axs[3].set_xlabel("time [s]")

    fig.suptitle(f"{case_name} – nonlinear plant")
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}_nonlin.png", dpi=200)
    plt.close(fig)


for idx, (out_name, row) in enumerate(best.items(), start=1):
    C = C_cases[out_name]
    sf = row["speed_factor"]
    prefix = f"component2_F_case{idx}"
    print(f"Plotting best observer for {out_name} (speed_factor={sf}) ...")
    plot_best_case(out_name, C, sf, prefix)

print("Done. Generated plots for best observers in Component 2 F.")

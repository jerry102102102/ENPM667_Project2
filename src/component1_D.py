#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENPM667 Project 2 - Component 1 D)
LQR design, linear vs nonlinear simulation, and comparison table.

This script:
  1. Builds the linearized model (A,B) for the double-pendulum crane
     with the specified numerical parameters.
  2. Runs a small grid search over Q,R for continuous-time LQR,
     computing K and the closed-loop eigenvalues.
  3. For each (Q,R,K), simulates:
        - Linear closed-loop:   x_dot = (A - B K) x
        - Nonlinear closed-loop: x_dot = f(x, -K x)
     from a given initial condition.
  4. Computes simple performance metrics and saves a comparison table
     as an image.
  5. Selects a "best" design based on eigenvalues and control effort
     and plots time responses only for that design.

Author: (your name)
"""

import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Parameters and linearized (A,B)
# ---------------------------------------------------------------------
g = 9.81
M  = 1000.0           # cart mass
m1 = 100.0            # first pendulum mass
m2 = 100.0            # second pendulum mass
l1 = 20.0             # first link length
l2 = 10.0             # second link length

# Linearized A, B from Part B (numerical)
A = np.array([
    [0.0, 1.0,                     0.0, 0.0,                     0.0, 0.0],
    [0.0, 0.0,               g*m1/M, 0.0,               g*m2/M, 0.0],
    [0.0, 0.0,                     0.0, 1.0,                     0.0, 0.0],
    [0.0, 0.0, -g*(M + m1)/(M*l1), 0.0,       -g*m2/(M*l1), 0.0],
    [0.0, 0.0,                     0.0, 0.0,                     0.0, 1.0],
    [0.0, 0.0,       -g*m1/(M*l2), 0.0, -g*(M + m2)/(M*l2), 0.0],
])

B = np.array([
    [0.0],
    [1.0/M],
    [0.0],
    [-1.0/(M*l1)],
    [0.0],
    [-1.0/(M*l2)],
])

def controllability_rank(A, B):
    n = A.shape[0]
    C = B
    for k in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, k) @ B))
    return np.linalg.matrix_rank(C)

print("Controllability rank (should be 6):", controllability_rank(A, B))


# ---------------------------------------------------------------------
# 2. LQR solver and candidate Q,R
# ---------------------------------------------------------------------
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    eig_cl = eigvals(A - B @ K)
    return K, P, eig_cl

Q_candidates = [
    np.diag([1.0, 0.1, 10.0, 0.5, 10.0, 0.5]),
    np.diag([10.0, 0.1, 50.0, 1.0, 50.0, 1.0]),
    np.diag([1.0, 0.1, 100.0, 1.0, 100.0, 1.0]),
]

R_candidates = [
    np.array([[0.001]]),
    np.array([[0.01]]),
    np.array([[0.1]]),
]


# ---------------------------------------------------------------------
# 3. Nonlinear dynamics
# ---------------------------------------------------------------------
def crane_nonlinear_dynamics(t, x, u_func):
    """
    Nonlinear dynamics of the double-pendulum crane.

    State x = [ x, xdot, theta1, theta1_dot, theta2, theta2_dot ].

    u_func: function u(t, x) -> scalar control force.
    """
    x1, x2, x3, x4, x5, x6 = x
    u = u_func(t, x)  # scalar

    D = M + m1 * np.sin(x3)**2 + m2 * np.sin(x5)**2

    x_ddot = (
        u
        + m1 * (g*np.cos(x3) + l1 * x4**2) * np.sin(x3)
        + m2 * (g*np.cos(x5) + l2 * x6**2) * np.sin(x5)
    ) / D

    theta1_ddot = -(x_ddot * np.cos(x3) + g * np.sin(x3)) / l1
    theta2_ddot = -(x_ddot * np.cos(x5) + g * np.sin(x5)) / l2

    return np.array([
        x2,
        x_ddot,
        x4,
        theta1_ddot,
        x6,
        theta2_ddot,
    ])


def make_linear_feedback(K):
    """Return u_func(t, x) = -K x (scalar)."""
    K = np.asarray(K).reshape(1, -1)
    def u_func(t, x):
        return float(-(K @ x)[0])
    return u_func


# ---------------------------------------------------------------------
# 4. Simulation helpers
# ---------------------------------------------------------------------

def simulate_linear(A, B, K, x0, t_span, num_steps=2000):
    Acl = A - B @ K
    def lin_cl(t, x):
        return (Acl @ x).flatten()
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    sol = solve_ivp(lin_cl, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T

def simulate_nonlinear(K, x0, t_span, num_steps=2000):
    u_func = make_linear_feedback(K)
    def nonlin_cl(t, x):
        return crane_nonlinear_dynamics(t, x, u_func)
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    sol = solve_ivp(nonlin_cl, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T

def compute_control_trace(K, t, x_traj):
    K = np.asarray(K).reshape(1, -1)
    u = np.zeros_like(t)
    for i, x in enumerate(x_traj):
        u[i] = -(K @ x)[0]
    return u

def max_abs_angle(x_traj, idx):
    return float(np.max(np.abs(x_traj[:, idx])))

def max_abs_u(K, t, x_traj):
    u = compute_control_trace(K, t, x_traj)
    return float(np.max(np.abs(u)))

def settling_time(t, signal, tol):
    idx = np.where(np.abs(signal) > tol)[0]
    if len(idx) == 0:
        return 0.0
    return float(t[idx[-1]])


# --- plotting helpers (only for BEST design later) -------------------
def plot_best_linear_vs_nonlinear(t_lin, x_lin, t_non, x_non):
    # Cart position
    plt.figure()
    plt.plot(t_lin, x_lin[:, 0], label='linear')
    plt.plot(t_non, x_non[:, 0], '--', label='nonlinear')
    plt.xlabel('Time [s]')
    plt.ylabel('Cart position x [m]')
    plt.title('Best design: cart position response')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("component1_D_best_x.png", dpi=300)
    plt.close()

    # Theta1
    plt.figure()
    plt.plot(t_lin, x_lin[:, 2] * 180/np.pi, label='linear')
    plt.plot(t_non, x_non[:, 2] * 180/np.pi, '--', label='nonlinear')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta1 [deg]')
    plt.title('Best design: first pendulum angle')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("component1_D_best_theta1.png", dpi=300)
    plt.close()

    # Theta2
    plt.figure()
    plt.plot(t_lin, x_lin[:, 4] * 180/np.pi, label='linear')
    plt.plot(t_non, x_non[:, 4] * 180/np.pi, '--', label='nonlinear')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta2 [deg]')
    plt.title('Best design: second pendulum angle')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("component1_D_best_theta2.png", dpi=300)
    plt.close()

def plot_best_control(t_lin, x_lin, t_non, x_non, K):
    u_lin = compute_control_trace(K, t_lin, x_lin)
    u_non = compute_control_trace(K, t_non, x_non)
    plt.figure()
    plt.plot(t_lin, u_lin, label='linear u')
    plt.plot(t_non, u_non, '--', label='nonlinear u')
    plt.xlabel('Time [s]')
    plt.ylabel('Input force F [N]')
    plt.title('Best design: control input')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("component1_D_best_u.png", dpi=300)
    plt.close()


# --- table helper ----------------------------------------------------
def make_table_image(headers, rows, filename):
    fig, ax = plt.subplots(figsize=(len(headers)*1.4, 0.4*len(rows) + 1.5))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(headers))))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. Main grid search + simulation
# ---------------------------------------------------------------------
if __name__ == "__main__":

    x0 = np.array([
        0.5,
        0.0,
        5.0 * np.pi/180.0,
        0.0,
        -5.0 * np.pi/180.0,
        0.0,
    ])

    t_span = (0.0, 40.0)

    designs_results = []   # for table + best-design selection
    design_params = []     # store (Q,R,K,eig_cl,t_lin,x_lin,t_non,x_non)

    design_index = 0
    for Qi in Q_candidates:
        for Ri in R_candidates:
            design_index += 1

            K, P, eig_cl = lqr(A, B, Qi, Ri)
            print("-----------------------------------------------------")
            print(f"Design #{design_index}: Qdiag = {np.diag(Qi)}, R = {Ri[0,0]}")
            print("Closed-loop eigenvalues (A - B K):")
            print(eig_cl)
            print()

            t_lin, x_lin = simulate_linear(A, B, K, x0, t_span)
            t_non, x_non = simulate_nonlinear(K, x0, t_span)

            ts_x_lin = settling_time(t_lin, x_lin[:, 0], tol=0.01)
            ts_x_non = settling_time(t_non, x_non[:, 0], tol=0.01)

            max_th1_lin = max_abs_angle(x_lin, 2) * 180/np.pi
            max_th1_non = max_abs_angle(x_non, 2) * 180/np.pi

            max_th2_lin = max_abs_angle(x_lin, 4) * 180/np.pi
            max_th2_non = max_abs_angle(x_non, 4) * 180/np.pi

            max_u_lin = max_abs_u(K, t_lin, x_lin)
            max_u_non = max_abs_u(K, t_non, x_non)

            max_real_eig = float(np.max(np.real(eig_cl)))

            designs_results.append([
                design_index,
                f"{np.diag(Qi)}",
                float(Ri[0,0]),
                ts_x_lin,
                ts_x_non,
                max_th1_lin,
                max_th1_non,
                max_th2_lin,
                max_th2_non,
                max_u_lin,
                max_u_non,
                max_real_eig,
            ])

            design_params.append({
                "index": design_index,
                "Q": Qi,
                "R": Ri,
                "K": K,
                "eig_cl": eig_cl,
                "t_lin": t_lin,
                "x_lin": x_lin,
                "t_non": t_non,
                "x_non": x_non,
                "max_u_non": max_u_non,
                "max_real_eig": max_real_eig,
            })

    # ---- build comparison table image (still all designs) ------------
    headers = [
        "Design",
        "Q diag",
        "R",
        "ts_x_lin [s]",
        "ts_x_non [s]",
        "max|θ1|_lin [deg]",
        "max|θ1|_non [deg]",
        "max|θ2|_lin [deg]",
        "max|θ2|_non [deg]",
        "max|u|_lin [N]",
        "max|u|_non [N]",
        "max Re(λ_cl)",
    ]
    table_filename = "component1_D_comparison_table.png"
    make_table_image(headers, designs_results, table_filename)
    print("\nSaved comparison table image to:", table_filename)

    # ---- select "best" design ---------------------------------------
    # 条件：max Re(λ_cl) < -1e-3, 之中挑 max|u|_non 最小
    candidates = [d for d in design_params if d["max_real_eig"] < -1e-3]
    if len(candidates) == 0:
        # 如果沒有符合條件，就直接選 max|u|_non 最小那個
        candidates = design_params

    best_design = min(candidates, key=lambda d: d["max_u_non"])

    print("\nBest design selected:")
    print("  index =", best_design["index"])
    print("  Q diag =", np.diag(best_design["Q"]))
    print("  R =", float(best_design["R"][0,0]))
    print("  max|u|_non =", best_design["max_u_non"])
    print("  max Re(λ_cl) =", best_design["max_real_eig"])

    # ---- plot only for best design ----------------------------------
    plot_best_linear_vs_nonlinear(
        best_design["t_lin"], best_design["x_lin"],
        best_design["t_non"], best_design["x_non"]
    )
    plot_best_control(
        best_design["t_lin"], best_design["x_lin"],
        best_design["t_non"], best_design["x_non"],
        best_design["K"]
    )

    print("Saved best-design plots: component1_D_best_*.png")
    print("Done.")

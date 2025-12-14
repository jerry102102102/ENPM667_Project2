#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ENPM667 Project 2 - Component 2 G)
LQG output feedback for smallest output vector, applied to:
  (1) linearized plant
  (2) original nonlinear plant

This version:
- Uses y = (x, theta1, theta2)
- LQR uses Design 2 from Component 1 table
- Measurement noise increased (sigma_x, sigma_th1, sigma_th2)
- Kalman Rv is set with separate [sigma_x, sigma_th1, sigma_th2]
- Fixes plotting layout (no subplot overlap)
- Avoids numpy scalar deprecation warnings
"""

import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 0) Parameters
# ------------------------------------------------------------
g = 9.81
M  = 1000.0
m1 = 100.0
m2 = 100.0
l1 = 20.0
l2 = 10.0

T_END = 40.0
DT = 0.01
N = int(T_END / DT) + 1
t = np.linspace(0.0, T_END, N)

# Initial condition (same as your Component 1)
x0 = np.array([0.5, 0.0, 5.0*np.pi/180.0, 0.0, -5.0*np.pi/180.0, 0.0])

# Optional actuator saturation (set None to disable)
U_MAX = None   # e.g. 80.0  (N)


# ------------------------------------------------------------
# 1) Linearized model (A,B)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 2) Output vector: y = (x, theta1, theta2)
# ------------------------------------------------------------
C = np.array([
    [1., 0., 0., 0., 0., 0.],   # x
    [0., 0., 1., 0., 0., 0.],   # theta1
    [0., 0., 0., 0., 1., 0.],   # theta2
])


# ------------------------------------------------------------
# 3) LQR (Design 2 from your table)
# ------------------------------------------------------------
# Design 2 row: Q diag [1, 0.1, 10, 0.5, 10, 0.5], R=0.01
Q = np.diag([1.0, 0.1, 10.0, 0.5, 10.0, 0.5])
R = np.array([[0.01]])

P_lqr = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ (B.T @ P_lqr))  # (1x6)
eig_cl = eigvals(A - B @ K)


# ------------------------------------------------------------
# 4) Continuous-time Kalman gain (dual CARE)
# ------------------------------------------------------------
sigma_w = 1.0   # process noise std

# -----------------------------
# Measurement noise (INCREASED)
# -----------------------------
# Original was: 0.005 m, 0.2 deg, 0.2 deg
# Increase to make the effect obvious:
sigma_x_m = 0.02          # 2 cm
sigma_th1_deg = 1.0       # 1 deg
sigma_th2_deg = 1.0       # 1 deg

sigma_th1 = np.deg2rad(sigma_th1_deg)
sigma_th2 = np.deg2rad(sigma_th2_deg)

Rv = np.diag([sigma_x_m**2, sigma_th1**2, sigma_th2**2])
Qw = np.array([[sigma_w**2]])
G = B

P_kf = solve_continuous_are(A.T, C.T, G @ Qw @ G.T, Rv)
L = (P_kf @ C.T @ np.linalg.inv(Rv))  # (6x3)


# ------------------------------------------------------------
# 5) Nonlinear dynamics
# ------------------------------------------------------------
def crane_nonlinear_f(x, u_scalar):
    """
    State x = [x, xdot, th1, th1dot, th2, th2dot]
    """
    x1, x2, th1, th1d, th2, th2d = x
    u = float(u_scalar)

    D = M + m1 * np.sin(th1)**2 + m2 * np.sin(th2)**2

    x_ddot = (
        u
        + m1 * (g*np.cos(th1) + l1 * th1d**2) * np.sin(th1)
        + m2 * (g*np.cos(th2) + l2 * th2d**2) * np.sin(th2)
    ) / D

    th1_ddot = -(x_ddot * np.cos(th1) + g * np.sin(th1)) / l1
    th2_ddot = -(x_ddot * np.cos(th2) + g * np.sin(th2)) / l2

    return np.array([x2, x_ddot, th1d, th1_ddot, th2d, th2_ddot])


# ------------------------------------------------------------
# 6) RK4 integrator
# ------------------------------------------------------------
def rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5*dt*k1, *args)
    k3 = f(x + 0.5*dt*k2, *args)
    k4 = f(x + dt*k3, *args)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ------------------------------------------------------------
# 7) Simulation
# ------------------------------------------------------------
def simulate_lqg(plant="linear", seed=0):
    rng = np.random.default_rng(seed)

    x_true = np.zeros((N, 6))
    x_hat  = np.zeros((N, 6))
    y_meas = np.zeros((N, 3))
    u_hist = np.zeros(N)

    x_true[0] = x0.copy()

    # initialize observer with first noisy measurement
    v0 = rng.normal(0.0, [sigma_x_m, sigma_th1, sigma_th2])
    y0 = (C @ x_true[0]) + v0
    x_hat[0] = np.zeros(6)
    x_hat[0, 0] = y0[0]
    x_hat[0, 2] = y0[1]
    x_hat[0, 4] = y0[2]
    y_meas[0] = y0

    for k in range(N-1):
        v = rng.normal(0.0, [sigma_x_m, sigma_th1, sigma_th2])
        y = (C @ x_true[k]) + v
        y_meas[k] = y

        u = (-(K @ x_hat[k]).item())
        if U_MAX is not None:
            u = float(np.clip(u, -U_MAX, U_MAX))
        u_hist[k] = u

        w = float(rng.normal(0.0, sigma_w))

        if plant == "linear":
            def f_lin(x, u_scalar, w_scalar):
                return (A @ x) + (B.flatten()*u_scalar) + (B.flatten()*w_scalar)
            x_true[k+1] = rk4_step(f_lin, x_true[k], DT, u, w)

        elif plant == "nonlinear":
            def f_nonlin(x, u_scalar, w_scalar):
                return crane_nonlinear_f(x, u_scalar + w_scalar)
            x_true[k+1] = rk4_step(f_nonlin, x_true[k], DT, u, w)

        else:
            raise ValueError("plant must be 'linear' or 'nonlinear'")

        def f_hat(xh, u_scalar, y_vec):
            innov = y_vec - (C @ xh)
            return (A @ xh) + (B.flatten()*u_scalar) + (L @ innov)

        x_hat[k+1] = rk4_step(f_hat, x_hat[k], DT, u, y)

    vN = rng.normal(0.0, [sigma_x_m, sigma_th1, sigma_th2])
    y_meas[-1] = (C @ x_true[-1]) + vN

    uN = (-(K @ x_hat[-1]).item())
    if U_MAX is not None:
        uN = float(np.clip(uN, -U_MAX, U_MAX))
    u_hist[-1] = uN

    return x_true, x_hat, y_meas, u_hist


# ------------------------------------------------------------
# 8) Plotting utilities
# ------------------------------------------------------------
def save_true_vs_hat_yfig(fname, title, x_true, x_hat, u_hist):
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(t, x_true[:, 0], label="x true")
    ax[0].plot(t, x_hat[:, 0], "--", label="x hat")
    ax[0].set_ylabel("x [m]")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t, np.rad2deg(x_true[:, 2]), label="theta1 true")
    ax[1].plot(t, np.rad2deg(x_hat[:, 2]), "--", label="theta1 hat")
    ax[1].set_ylabel("theta1 [deg]")
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(t, np.rad2deg(x_true[:, 4]), label="theta2 true")
    ax[2].plot(t, np.rad2deg(x_hat[:, 4]), "--", label="theta2 hat")
    ax[2].set_ylabel("theta2 [deg]")
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(t, u_hist, label="u")
    ax[3].set_ylabel("u [N]")
    ax[3].set_xlabel("time [s]")
    ax[3].grid(True)
    ax[3].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def save_all_states_fig(fname, title, x_true, x_hat):
    labels = ["x [m]", "xdot [m/s]", "theta1 [deg]", "theta1dot [deg/s]", "theta2 [deg]", "theta2dot [deg/s]"]
    fig, ax = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

    ax[0].plot(t, x_true[:, 0], label="true")
    ax[0].plot(t, x_hat[:, 0], "--", label="hat")
    ax[0].set_ylabel(labels[0]); ax[0].grid(True); ax[0].legend()

    ax[1].plot(t, x_true[:, 1], label="true")
    ax[1].plot(t, x_hat[:, 1], "--", label="hat")
    ax[1].set_ylabel(labels[1]); ax[1].grid(True); ax[1].legend()

    ax[2].plot(t, np.rad2deg(x_true[:, 2]), label="true")
    ax[2].plot(t, np.rad2deg(x_hat[:, 2]), "--", label="hat")
    ax[2].set_ylabel(labels[2]); ax[2].grid(True); ax[2].legend()

    ax[3].plot(t, np.rad2deg(x_true[:, 3]), label="true")
    ax[3].plot(t, np.rad2deg(x_hat[:, 3]), "--", label="hat")
    ax[3].set_ylabel(labels[3]); ax[3].grid(True); ax[3].legend()

    ax[4].plot(t, np.rad2deg(x_true[:, 4]), label="true")
    ax[4].plot(t, np.rad2deg(x_hat[:, 4]), "--", label="hat")
    ax[4].set_ylabel(labels[4]); ax[4].grid(True); ax[4].legend()

    ax[5].plot(t, np.rad2deg(x_true[:, 5]), label="true")
    ax[5].plot(t, np.rad2deg(x_hat[:, 5]), "--", label="hat")
    ax[5].set_ylabel(labels[5]); ax[5].set_xlabel("time [s]")
    ax[5].grid(True); ax[5].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------
# 9) Main
# ------------------------------------------------------------
if __name__ == "__main__":

    print("==== LQG Summary ====")
    print("Output vector: y = (x, theta1, theta2)")
    print("C =\n", C)
    print("LQR (Design 2): Q diag =", np.diag(Q), " R =", float(R[0,0]))
    print("Closed-loop eigenvalues of (A-BK):")
    print(eig_cl)
    print()
    print("Kalman (continuous):")
    print(f"sigma_w = {sigma_w}")
    print(f"sigma_v = [sigma_x={sigma_x_m} m, sigma_th1={sigma_th1_deg} deg, sigma_th2={sigma_th2_deg} deg]")
    print("Rv =\n", Rv)
    print()
    print("K =\n", K)
    print()
    print("L =\n", L)
    print()

    x_true_lin, x_hat_lin, _, u_lin = simulate_lqg(plant="linear", seed=1)
    save_true_vs_hat_yfig(
        "component2_G_LQG_y_xth1th2_linear_true_vs_hat.png",
        "LQG (y = x, theta1, theta2) - Linear plant (true vs hat)",
        x_true_lin, x_hat_lin, u_lin
    )
    save_all_states_fig(
        "component2_G_LQG_y_xth1th2_linear_states.png",
        "LQG (y = x, theta1, theta2) - Linear plant (all states true vs hat)",
        x_true_lin, x_hat_lin
    )

    x_true_non, x_hat_non, _, u_non = simulate_lqg(plant="nonlinear", seed=2)
    save_true_vs_hat_yfig(
        "component2_G_LQG_y_xth1th2_nonlinear_true_vs_hat.png",
        "LQG (y = x, theta1, theta2) - Nonlinear plant (true vs hat)",
        x_true_non, x_hat_non, u_non
    )
    save_all_states_fig(
        "component2_G_LQG_y_xth1th2_nonlinear_states.png",
        "LQG (y = x, theta1, theta2) - Nonlinear plant (all states true vs hat)",
        x_true_non, x_hat_non
    )

    print("Saved figures:")
    print(" - component2_G_LQG_y_xth1th2_linear_true_vs_hat.png")
    print(" - component2_G_LQG_y_xth1th2_linear_states.png")
    print(" - component2_G_LQG_y_xth1th2_nonlinear_true_vs_hat.png")
    print(" - component2_G_LQG_y_xth1th2_nonlinear_states.png")
    print("Done.")

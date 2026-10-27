"""
Classical hybrid controller: energy-based swing-up + LQR balance.

Far from upright, the energy controller pumps energy into the pendulum
(like pushing a swing) until it has just enough to reach the top. Once the
pendulum is within `switch_angle_deg` of upright, LQR takes over and
balances it. Starting near the top, LQR is in charge from the first step.

Both parts respect the same torque limit as the RL agents.

Physics (matches Pendulum-v1, rod pendulum, theta = 0 is upright):
    theta_ddot = (3g / 2l) * sin(theta) + (3 / m l^2) * u
    Energy     E = 1/2 * I * theta_dot^2 + m * g * (l/2) * cos(theta),  I = m l^2 / 3
    Upright at rest has E = m * g * l / 2
"""

import numpy as np
from scipy.linalg import solve_discrete_are


def wrap_angle(theta):
    """Map any angle to [-pi, pi], so 'distance from upright' is just abs()."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


class HybridController:

    def __init__(self, env, pump_gain=5.0, switch_angle_deg=20.0, q_scale=1.0):
        """
        env              : a TorqueLimitedPendulum (physics and torque limit are read from it)
        pump_gain        : how aggressively the swing-up adds energy
        switch_angle_deg : hand off to LQR when within this many degrees of upright
        q_scale          : multiplies LQR's state cost Q; higher = tighter, more aggressive balancing
        """
        self.m, self.g, self.l = env.m, env.g, env.l
        self.max_torque = env.max_torque
        self.dt = env.dt
        self.pump_gain = pump_gain
        self.switch_angle = np.deg2rad(switch_angle_deg)

        self.inertia = self.m * self.l**2 / 3
        self.target_energy = self.m * self.g * self.l / 2

        self.K = self._compute_lqr_gain(q_scale)

    def _compute_lqr_gain(self, q_scale):
        # Linearized dynamics around upright (sin(theta) ~ theta), state x = [theta, theta_dot]:
        #   theta_ddot = a * theta + b * u
        a = 3 * self.g / (2 * self.l)
        b = 3 / (self.m * self.l**2)

        # The simulator steps in discrete time (dt = 0.05 s) with semi-implicit Euler:
        #   theta_dot' = theta_dot + (a*theta + b*u) * dt
        #   theta'     = theta + theta_dot' * dt
        # Designing LQR on this exact discrete model keeps it stable even with a
        # strong motor, where a continuous-time design would overreact between steps.
        dt = self.dt
        A = np.array([[1 + a * dt**2, dt],
                      [a * dt,        1.0]])
        B = np.array([[b * dt**2],
                      [b * dt]])

        # Bryson's rule: weight each term by 1 / (largest acceptable value)^2
        max_angle = np.deg2rad(10)   # acceptable angle error
        max_speed = 1.0              # acceptable angular velocity (rad/s)
        Q = q_scale * np.diag([1 / max_angle**2, 1 / max_speed**2])
        R = np.array([[1 / self.max_torque**2]])

        # Solve the discrete Riccati equation, then K = (R + B^T P B)^-1 B^T P A
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K   # shape (1, 2)

    def energy(self, theta, theta_dot):
        return 0.5 * self.inertia * theta_dot**2 + self.m * self.g * (self.l / 2) * np.cos(theta)

    def act(self, theta, theta_dot):
        """Return the torque to apply, as an array of shape (1,) like env.step expects."""
        theta = wrap_angle(theta)

        if abs(theta) < self.switch_angle:
            # Balance: LQR
            u = -(self.K @ np.array([theta, theta_dot]))[0]
        else:
            # Swing-up: push in the direction of motion when energy is too low,
            # against it when too high. Energy changes at rate theta_dot * u.
            energy_error = self.target_energy - self.energy(theta, theta_dot)
            direction = np.sign(theta_dot) if abs(theta_dot) > 1e-3 else 1.0
            u = self.pump_gain * energy_error * direction

        u = np.clip(u, -self.max_torque, self.max_torque)
        return np.array([u], dtype=np.float32)
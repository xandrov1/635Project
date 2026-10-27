"""
Runs a controller over the evaluation grid and records per-episode metrics.

Grid:   torque limit x starting angle x episodes (one seed per episode)
Output: a CSV with one row per episode:
        torque_fraction, start_angle_deg, seed, success, settling_time, effort

Metrics:
  success        |theta| < 10 deg and |theta_dot| < 0.5 rad/s at every step of the final 2 s
  settling_time  first time after which the pendulum stays in that band until the end
                 (only meaningful for successful episodes; NaN otherwise)
  effort         sum of u^2 * dt over the episode

A controller is passed in as a factory: make_policy(env) -> act(obs, theta, theta_dot) -> action.
The factory is called once per torque limit, so it can build a controller tuned for that limit
(classical) or load the policy trained for that limit (RL).

Usage (classical controller, default settings):
    python evaluate.py --out results/classical_default.csv
    python evaluate.py --episode-seconds 20 --out results/classical_default_20s.csv
"""

import argparse
import csv
import os

import numpy as np

from pendulum import TorqueLimitedPendulum
from classical import HybridController, wrap_angle


TORQUE_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
START_ANGLES_DEG = list(range(0, 181, 15))
EVAL_SEEDS = list(range(0, 20))          # evaluation episodes
TUNING_SEEDS = list(range(1000, 1005))   # tuning episodes, never used for evaluation

SUCCESS_ANGLE_DEG = 10.0
SUCCESS_SPEED = 0.5      # rad/s
SUCCESS_WINDOW_S = 2.0


def in_band(theta, theta_dot):
    return (abs(np.rad2deg(wrap_angle(theta))) < SUCCESS_ANGLE_DEG
            and abs(theta_dot) < SUCCESS_SPEED)


def run_episode(env, act, start_angle_deg, seed):
    """Run one episode and return (success, settling_time, effort)."""
    obs, _ = env.reset(seed=seed, options={"start_angle_deg": start_angle_deg})
    theta, theta_dot = env.state

    band = []        # whether each step ended inside the success band
    effort = 0.0
    done = False
    while not done:
        action = act(obs, theta, theta_dot)
        u = float(np.clip(action[0], -env.max_torque, env.max_torque))
        effort += u**2 * env.dt

        obs, _, terminated, truncated, info = env.step(action)
        theta, theta_dot = info["theta"], info["theta_dot"]
        band.append(in_band(theta, theta_dot))
        done = terminated or truncated

    window = int(round(SUCCESS_WINDOW_S / env.dt))
    success = all(band[-window:])

    settling_time = float("nan")
    if success:
        # Walk back from the end to find where the final in-band stretch began
        k = len(band)
        while k > 0 and band[k - 1]:
            k -= 1
        settling_time = k * env.dt

    return success, settling_time, effort


def evaluate(make_policy, out_path, torque_fractions=TORQUE_FRACTIONS,
             start_angles=START_ANGLES_DEG, seeds=EVAL_SEEDS,
             episode_seconds=10.0, label=""):
    """Evaluate a controller over the grid and write one CSV row per episode."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["label", "torque_fraction", "start_angle_deg", "seed",
              "success", "settling_time", "effort"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for frac in torque_fractions:
            env = TorqueLimitedPendulum(torque_fraction=frac,
                                        episode_seconds=episode_seconds)
            act = make_policy(env)
            successes = []

            for angle in start_angles:
                for seed in seeds:
                    success, t_settle, effort = run_episode(env, act, angle, seed)
                    successes.append(success)
                    writer.writerow({
                        "label": label,
                        "torque_fraction": frac,
                        "start_angle_deg": angle,
                        "seed": seed,
                        "success": int(success),
                        "settling_time": t_settle,
                        "effort": effort,
                    })

            print(f"torque {frac:>4}x: success rate {np.mean(successes):.2f}")

    print(f"Saved {out_path}")


def classical_policy(pump_gain=5.0, switch_angle_deg=20.0, q_scale=1.0):
    """Factory for the hybrid controller. The classical side ignores obs and uses theta directly."""
    def make_policy(env):
        controller = HybridController(env, pump_gain=pump_gain,
                                      switch_angle_deg=switch_angle_deg,
                                      q_scale=q_scale)
        return lambda obs, theta, theta_dot: controller.act(theta, theta_dot)
    return make_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the classical controller over the grid.")
    parser.add_argument("--out", default="results/classical_default.csv")
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--pump-gain", type=float, default=5.0)
    parser.add_argument("--switch-angle", type=float, default=20.0)
    parser.add_argument("--q-scale", type=float, default=1.0)
    args = parser.parse_args()

    evaluate(
        classical_policy(args.pump_gain, args.switch_angle, args.q_scale),
        out_path=args.out,
        episode_seconds=args.episode_seconds,
        label="classical",
    )
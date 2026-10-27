"""
Custom pendulum environment for the RL vs. classical control project.

Built on Gymnasium's Pendulum-v1, with three changes:
  1. Torque limit is set as a FRACTION of the torque needed to hold the
     pendulum horizontal (m*g*l/2 = 5 N*m with the default physics).
  2. Speed cap raised from 8 to 17 rad/s, so swing-up (~7.75 rad/s needed)
     is never clipped by the simulator.
  3. We control exactly where the pendulum starts (for the evaluation grid)
     and how long an episode lasts.

Angle convention (same as Pendulum-v1):
  theta = 0      -> pointing straight UP (the goal)
  theta = +/- pi -> hanging straight DOWN
"""

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv


class TorqueLimitedPendulum(PendulumEnv):

    def __init__(self, torque_fraction=1.0, max_speed=17.0,
                 episode_seconds=10.0, render_mode=None):
        # Let Pendulum-v1 set up its default physics (g=10, m=1, l=1, dt=0.05)
        super().__init__(render_mode=render_mode)

        # --- Change 1: torque limit as a fraction of "hold horizontal" torque ---
        self.horizontal_torque = self.m * self.g * self.l / 2   # = 5.0 N*m
        self.torque_fraction = torque_fraction
        self.max_torque = torque_fraction * self.horizontal_torque

        # --- Change 2: raise the speed cap ---
        self.max_speed = max_speed

        # --- Change 3: episode length (we handle the time limit ourselves) ---
        self.max_steps = int(round(episode_seconds / self.dt))   # 10 s -> 200 steps
        self.steps = 0

        # The action/observation spaces were built from the OLD limits in
        # super().__init__, so rebuild them with the new ones.
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Two ways to start:
          - options={"start_angle_deg": 45}  -> evaluation: start at 45 deg from
            upright, plus a small random nudge (+/-2 deg, +/-0.1 rad/s) so
            repeated episodes in the same grid cell aren't identical.
          - no options                         -> training: random angle anywhere
            in the full circle, small random velocity.
        """
        # Seeds the random number generator (self.np_random)
        super().reset(seed=seed)
        options = options or {}

        if "start_angle_deg" in options:
            angle = np.deg2rad(options["start_angle_deg"])
            angle += self.np_random.uniform(-np.deg2rad(2), np.deg2rad(2))
            velocity = self.np_random.uniform(-0.1, 0.1)
        else:
            angle = self.np_random.uniform(-np.pi, np.pi)
            velocity = self.np_random.uniform(-0.1, 0.1)

        self.state = np.array([angle, velocity])
        self.last_u = None
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # Pendulum-v1 does the physics, the torque clipping, and the reward.
        # Because we changed self.max_torque and self.max_speed, it uses ours.
        obs, reward, terminated, _, info = super().step(action)

        self.steps += 1
        truncated = self.steps >= self.max_steps   # time's up

        # Expose the raw state so the evaluator and classical controller can use it
        info["theta"] = float(self.state[0])
        info["theta_dot"] = float(self.state[1])
        return obs, reward, terminated, truncated, info
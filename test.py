import numpy as np
from pendulum import TorqueLimitedPendulum

# Torque limit and speed cap are set correctly
env = TorqueLimitedPendulum(torque_fraction=0.4)
print("max torque:", env.max_torque)            # expect 2.0
print("max speed:", env.max_speed)              # expect 17.0
print("steps per episode:", env.max_steps)      # expect 200

# Evaluation start lands near the requested angle
env.reset(seed=0, options={"start_angle_deg": 45})
print("start angle (deg):", round(np.rad2deg(env.state[0]), 2))   # expect ~43-47

# Speed cap: with a strong motor spinning it, speed should top out at 17, not 8
env = TorqueLimitedPendulum(torque_fraction=2.0)
env.reset(seed=0, options={"start_angle_deg": 1})
peak = 0.0
for _ in range(200):
    _, _, _, _, info = env.step(np.array([10.0]))
    peak = max(peak, abs(info["theta_dot"]))
print("peak speed:", round(peak, 2))            # expect 17.0

# Episode ends after 200 steps
env.reset(seed=1)
steps, truncated = 0, False
while not truncated:
    _, _, _, truncated, _ = env.step(np.array([0.0]))
    steps += 1
print("episode length:", steps)                 # expect 200
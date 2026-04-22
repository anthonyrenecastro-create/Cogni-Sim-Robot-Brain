import numpy as np


def build_observation_vector(
    imu_accel,
    imu_gyro,
    proximity,
    battery_level,
    cpu_temp,
    input_dim=32,
):
    obs = np.zeros(input_dim, dtype=np.float32)

    obs[0:3] = np.asarray(imu_accel, dtype=np.float32)[:3]
    obs[3:6] = np.asarray(imu_gyro, dtype=np.float32)[:3]

    prox = np.asarray(proximity, dtype=np.float32)
    n = min(len(prox), 8)
    obs[6:6 + n] = prox[:n]

    obs[14] = float(battery_level)
    obs[15] = float(cpu_temp)

    return obs

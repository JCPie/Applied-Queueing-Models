"""
Chris Petri & Benjamin Meelhuijsen

MSc Applied Mathematics
AQM

Tired Server

Q4 - 2023/2024
"""

import numpy as np


def get_r(A_0: np.ndarray, A_1: np.ndarray, A_2: np.ndarray, gamma: float = 1, iterations: int = 10) -> np.ndarray:
    # Redef A
    A_0 = gamma * A_0
    A_1 = np.eye(A_1.shape[0]) + gamma * A_1
    A_2 = gamma * A_2

    R_old = np.zeros(A_2.shape)
    R_new = R_old
    for _ in range(iterations):
        R_new = -(A_0 + R_old**2 @ A_2) @ np.linalg.inv(A_1)
        R_old = R_new
    return R_new


def def_a():
    arrival = 1
    departure = 2/5
    tired = 1/2
    exhausted = 1/4

    A_0 = arrival * np.eye(3)
    A_1 = np.array([[-(arrival + departure + tired), tired, 0], [0, -(arrival + departure*3/4 + tired), tired], [exhausted, 0, -(arrival + exhausted)]])
    A_2 = np.array([[departure, 0, 0], [0, departure * 3/4, 0], [0, 0, 0]])

    return A_0, A_1, A_2


def def_b():
    arrival = 1
    departure = 2/5
    b_00 = -arrival * np.eye(1)
    b_01 = np.array([arrival, 0, 0])
    b_10 = np.array([[departure],[departure*3/4],[0]])
    return b_00, b_01, b_10


def get_gamma(A_1):
    gamma_list: list = []
    for i in range(A_1.shape[0]):
        gamma_list.append(abs(A_1[i][i]))
    max_gamma = np.max(gamma_list)
    return max_gamma


def get_pi01_T(B00, B10, B01, B11, A2, R):
    # Reshape arrays
    B10 = B10.reshape(-1, 1)
    B01 = B01.reshape(1, -1)

    # Change B01, B11 + R @ A2
    RB = B01
    RO = B11 + R @ A2

    # Substitute kth column with [e (I - R)^-1 @ e]^T
    RB[-1] = 1
    RO[:, -1] = (np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0])).T

    # Calc A*
    A_star = -B10 @ np.linalg.inv(B00) @ RB + RO
    pi1 = np.linalg.solve(A_star.T, np.matrix([0, 0, 1]).T)
    print(A_star.T @ pi1)
    print(pi1.T @ (-B10 @ np.linalg.inv(B00) @ B01 + B11 + R @ A2))
    return 0, 0


def get_pi01(b_00, b_10, A_1, A_2, R, gamma = 1):
    # Redef A
    A_1 = np.eye(A_1.shape[0]) + gamma * A_1
    A_2 = gamma * A_2

    RB = b_01
    RO = A_1 + R @ A_2
    print("Rechtsboven (RB)\n", RB, "\n","Rechtsonder (RO)\n", RO)

    # Substitute kth column with [e (I - R)^-1 @ e]^T
    RB[-2] = 1
    RO[:, -2] = (np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0])).T
    # Check if subtitution is correctly performed
    print("Rechtsboven (RB)\n", RB, "\n","Rechtsonder (RO)\n", RO)


    temp0 = -b_10 @ np.linalg.inv(b_00) @ RB.reshape(1, -1) + RO
    print("tt", temp0)
    pi1 = np.linalg.solve(temp0.T, np.matrix([0, 1, 0]).T)
    pi1 = pi1.T
    # print(pi1.T)
    # print(pi1.T @ (-b_10 @ np.linalg.inv(b_00) @ b_01.reshape(1, -1) + A_1 + R @ A_2))
    pi0 = -(pi1 @ b_10) @ np.linalg.inv(b_00)
    print("Validity", pi0 @ b_00 + pi1 @ b_10)
    print("Validity", pi0 @ b_01.reshape(1, -1) + pi1 @ (A_1 + R @ A_2))
    print("Validity", pi0 * np.ones(3) + pi1 @ np.linalg.inv(np.eye(R.shape[0])) @ np.ones(3))
    return pi0, pi1


def get_distr(n: int, pi: np.ndarray, R: np.ndarray):
    R = np.linalg.matrix_power(R, n - 1)
    new_pi = pi @ R
    return new_pi


def mean_val_analysis():
    pass


if __name__ == "__main__":
    A_0, A_1, A_2 = def_a()
    print(def_a())
    b_00, b_01, b_10 = def_b()
    print(def_b())
    gamma = get_gamma(A_1)
    R = get_r(A_0, A_1, A_2, gamma, iterations=10)
    print(R)
    # pi0, pi1 = get_pi01(b_00, b_10, A_1, A_2, R)
    pi0, pi1 = get_pi01_T(b_00, b_10, b_01, A_1, A_2, R)
    # pin = get_distr(1, pi1, R)
    # print(R ** 1)
    print(pi0, pi1)
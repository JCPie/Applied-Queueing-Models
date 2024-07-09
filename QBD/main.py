"""
Chris Petri & Benjamin Meelhuijsen

MSc Applied Mathematics
AQM

Tired Server

Q4 - 2023/2024
"""

import numpy as np
from matplotlib import pyplot as plt


def get_r(A_0: np.ndarray, A_1: np.ndarray, A_2: np.ndarray, gamma: float = 1, iterations: int = 10) -> np.ndarray:
    R_old = np.zeros(A_2.shape)
    R_new = R_old
    for _ in range(iterations):
        R_new = -(A_0 + R_old ** 2 @ A_2) @ np.linalg.inv(A_1)
        R_old = R_new
    return R_new


def m1_a():
    arrival = 1
    departure = 2 / 5
    tired = 1 / 2
    exhausted = 1 / 4
    departure_ex = 0

    A_0 = arrival * np.eye(3)
    A_1 = np.array([[-(arrival + departure + tired), tired, 0], [0, -(arrival + departure * 3 / 4 + tired), tired],
                    [exhausted, 0, -(arrival + exhausted + departure_ex)]])
    A_2 = np.array([[departure, 0, 0], [0, departure * 3 / 4, 0], [0, 0, departure_ex]])

    return A_0, A_1, A_2


def m1_b():
    arrival = 1
    departure = 2 / 5
    departure_ex = 0
    b_00 = -arrival * np.eye(1)
    b_01 = np.array([arrival, 0, 0])
    b_10 = np.array([[departure], [departure * 3 / 4], [departure_ex]])
    return b_00, b_01, b_10


def m2a_a():
    arrival = 1
    departure_1 = 2
    departure_2 = 2

    A_0 = arrival * np.eye(4)
    A_1 = np.array(
        [[-(arrival + departure_1), departure_1, 0, 0], [0, -(arrival + departure_1 + departure_2), departure_1, 0],
         [0, 0, -(arrival + departure_1 + departure_2), departure_1], [0, 0, 0, -(arrival + departure_2)]])
    A_2 = np.array([[0, 0, 0, 0], [departure_2, 0, 0, 0], [0, departure_2, 0, 0], [0, 0, departure_2, 0]])

    return A_0, A_1, A_2


def m2a_b(A0: None, A2: None):
    arrival = 1
    departure_1 = 2
    departure_2 = 2
    b_00 = np.zeros((10, 10))
    arrival_pos = [(0, 1), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)]
    departure_1_pos = [(1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)]
    departure_2_pos = [(2, 0), (4, 1), (5, 2), (7, 3), (8, 4), (9, 5)]
    for pos in arrival_pos:
        b_00[pos] = arrival

    for pos in departure_1_pos:
        b_00[pos] = departure_1

    for pos in departure_2_pos:
        b_00[pos] = departure_2

    # Diagonal
    for i in range(b_00.shape[0]):
        b_00[i, i] = -np.sum(b_00[i, :])
    b_00[6:, 6:] += -arrival * np.eye(4)

    b_01 = np.zeros((10, 4))
    b_01[6:, :] = arrival * np.eye(4)

    b_10 = np.zeros((4, 10))
    b_10[:, 5:9] = departure_2 * np.eye(4)
    b_10[0, 5] = 0

    return b_00, b_01, b_10


def m2b_a():
    arrival = 1
    departure_1 = 4
    departure_2a = 4 + 2 * np.sqrt(2)
    departure_2b = 4 - 2 * np.sqrt(2)

    hype_prob = 1 / 2

    A_0 = arrival * np.eye(16)
    A_2_00 = np.zeros((8, 8))
    A_2_11 = np.zeros((8, 8))
    A_zero = np.zeros((8, 8))
    for i in range(A_2_00.shape[0] - 2):
        A_2_00[i + 2, i] = departure_2a
        A_2_11[i + 2, i] = departure_2b
    A_top = np.hstack((A_2_00, A_zero))
    A_bot = np.hstack((A_zero, A_2_11))
    A_2 = np.vstack((A_top, A_bot))

    A_1_00 = np.zeros((8, 8))
    for i in range(A_1_00.shape[0] - 2):
        A_1_00[i, i + 1] = departure_1

    A_1_10 = np.zeros((8, 8))
    for i in range(1, A_1_10.shape[0]):
        A_1_10[i, i] = hype_prob

    A_top = np.hstack((A_1_00, A_1_10))
    A_bot = np.hstack((A_1_10, A_1_00))
    A_1 = np.vstack((A_top, A_bot))

    for i in range(A_1.shape[0]):
        A_1[i, i] = -np.sum(A_1[i, :]) - np.sum(A_0[i, :]) - np.sum(A_2[i, :])

    print("A0: ", A_0)
    print("A1: ", A_1)
    print("A2: ", A_2)
    return A_0, A_1, A_2


def m2b_b(A0, A2):
    arrival = 1
    departure_1 = 4
    departure_2a = 4 + 2 * np.sqrt(2)
    departure_2b = 4 - 2 * np.sqrt(2)
    hype_prob = 1 / 2

    b_00_00 = np.zeros((20, 20))
    b_00_11 = np.zeros((20, 20))
    arrival_pos = [(0, 2), (1, 2), (2, 6), (3, 7), (4, 8), (5, 9), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16),
                   (11, 17)]
    departure_1_pos = [(2, 3), (3, 4), (6, 7), (7, 8), (8, 9), (9, 10), (12, 13), (13, 14), (14, 15), (15, 16),
                       (16, 17), (17, 18)]
    departure_2_pos = [(4, 0), (5, 1), (10, 4), (11, 5), (14, 6), (15, 7), (16, 8), (17, 9), (18, 10), (19, 11)]

    for pos in arrival_pos:
        b_00_00[pos] = arrival
        b_00_11[pos] = arrival

    for pos in departure_1_pos:
        b_00_11[pos] = departure_1
        b_00_00[pos] = departure_1

    for pos in departure_2_pos:
        b_00_00[pos] = departure_2a
        b_00_11[pos] = departure_2b

    b_00_01 = hype_prob * np.eye(20)
    b_00_01[0, 0] = 0
    b_00_01[1, 1] = 0

    b_00_top = np.hstack((b_00_00, b_00_01))
    b_00_bot = np.hstack((b_00_01, b_00_11))
    b_00 = np.vstack((b_00_top, b_00_bot))

    b_01 = np.zeros((40, 16))
    b_01[-16:, :] = np.transpose(A0)

    b_10 = np.zeros((16, 40))
    b_10[:, -16:] = A2


    for i in range(b_00.shape[0]):
        b_00[i, i] += -np.sum(b_00[i, :]) - np.sum(b_01[i, :])

    print("b_10: ", b_10)

    return b_00, b_01, b_10


def get_gamma(A_1):
    gamma_list: list = []
    for i in range(A_1.shape[0]):
        gamma_list.append(abs(A_1[i][i]))
    max_gamma = np.max(gamma_list)
    return max_gamma


def get_pi01_T(B00, B10, B01, B11, A2, R):
    # print("\nFind distribution\n")
    # Reshape arrays
    print("type: ", type(B10))

    # B10 = B10.reshape(-1, 1)
    # B01 = B01.reshape(1, -1)

    # Change B01, B11 + R @ A2
    RB = B01
    RO = B11 + R @ A2

    # Substitute kth column with [e (I - R)^-1 @ e]^T
    print("Rechtsboven (RB)\n", RB, "\n", "Rechtsonder (RO)\n", RO)
    RB[:, -1] = 1
    C = np.eye(R.shape[0]) - R
    RO[:, -1] = (np.linalg.inv(C) @ np.ones(R.shape[0])).T
    print("Rechtsboven (RB)\n", RB, "\n", "Rechtsonder (RO)\n", RO)

    # Calc A*
    A_star = -B10 @ np.linalg.inv(B00) @ RB + RO
    # print("e @ A^-1", np.matrix([0, 0, 1]) @ np.linalg.inv(A_star))
    e_k = np.zeros(A_star.shape[0])
    e_k[-1] = 1
    pi1 = np.linalg.solve(A_star.T, e_k.T)
    pi0 = -(pi1.T @ B10) @ np.linalg.inv(B00)
    # print("A*", A_star, type(A_star))
    # print("Found pi_1: \n", pi1, type(pi1))
    print(pi0, pi1)
    print(pi1.T @ R ** 10)
    # print(A_star.T @ pi1)
    # print(pi1.T @ (-B10 @ np.linalg.inv(B00) @ B01 + B11 + R @ A2))
    # print(pi0 * np.ones(R.shape[0]) + pi1.T @ np.linalg.inv(np.eye(3) - R) @ np.ones(R.shape[0]))

    print("Validity checks")
    print("Validity", pi0 @ B00 + pi1.T @ B10)
    print("Validity", pi0 @ B01 + pi1.T @ B11 + pi1.T @ R @ A2)
    print("Validity", pi0 * np.ones(pi0.shape[0]) + pi1.T @ np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0]))

    return pi0, pi1


# def get_pi01(b_00, b_10, b_01, A_1, A_2, R, gamma=1):
#     # Redef A
#     A_1 = np.eye(A_1.shape[0]) + gamma * A_1
#     A_2 = gamma * A_2
#
#     RB = b_01
#     RO = A_1 + R @ A_2
#     # print("Rechtsboven (RB)\n", RB, "\n", "Rechtsonder (RO)\n", RO)
#
#     # Substitute kth column with [e (I - R)^-1 @ e]^T
#     RB[-1] = 1
#     RO[:, -1] = (np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0])).T
#     # Check if subtitution is correctly performed
#     # print("Rechtsboven (RB)\n", RB, "\n", "Rechtsonder (RO)\n", RO)
#
#     temp0 = -b_10 @ np.linalg.inv(b_00) @ RB.reshape(1, -1) + RO
#     # print("A*", type(temp0))
#     pi1 = np.linalg.solve(temp0.T, np.matrix([0, 0, 1]).T)
#     # print("Found pi_1: \n", pi1)
#     print("A*", temp0, type(temp0))
#     print("Found pi_1: \n", pi1, type(pi1))
#
#     pi1 = pi1.T
#     # print(pi1.T)
#     # print(pi1.T @ (-b_10 @ np.linalg.inv(b_00) @ b_01.reshape(1, -1) + A_1 + R @ A_2))
#     pi0 = -(pi1 @ b_10) @ np.linalg.inv(b_00)
#     print("Validity", pi0 @ b_00 + pi1 @ b_10)
#     print("Validity", pi0 @ b_01.reshape(1, -1) + pi1 @ (A_1 + R @ A_2))
#     print("Validity", pi0 * np.ones(3) + pi1 @ np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(3))
#     return pi0, pi1


def get_distr(n: int, pi: np.ndarray, R: np.ndarray):
    R = np.linalg.matrix_power(R, n - 1)
    new_pi = pi @ R
    return new_pi


def validity_check(pi0: np.ndarray, pi1: np.ndarray, R: np.ndarray):
    tot = np.sum(pi0)
    prob_list: list = [tot]
    for n in range(1, 100):
        new_R = np.linalg.matrix_power(R, n - 1)
        new_pi = pi1 @ new_R
        tot += np.sum(new_pi)
        prob_list.append(tot)
        print("Total sum: ", tot)
    return prob_list


def plot_prob_list(prob_list: list):
    plt.plot(prob_list)
    plt.xlabel("$n$")
    plt.ylabel("Probability")
    plt.grid()
    plt.title("Cumulative Probability based on number of considered $pi_n$")
    plt.savefig("Validity pi_n.png")
    plt.show()


def mean_val_analysis(pi0: np.ndarray, pi1: np.ndarray, R: np.ndarray):
    length_ex: float = 0
    length_in: float = 0
    N = 3

    # External length
    len_ex_ex = pi1.T @ np.linalg.matrix_power(np.eye(R.shape[0]) - R, -2) @ np.ones(R.shape[0])
    print("Exact calculation: ", len_ex_ex)

    # Iterative:
    for n in range(1, 1000):
        length_ex += n * (pi1.T @ np.linalg.matrix_power(R, n)) @ np.ones(R.shape[0])
    print("Iterative calculation: ", length_ex)

    # Internal length
    length_in_ex = np.sum(pi1) + 3 * pi1.T @ R @ np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0])
    print(length_in_ex)

    # Iterative
    for n in range(0, 10000):
        length_in += np.linalg.matrix_power(R, n) @ np.ones(R.shape[0])
    print("Iterative calculation: ", 3 * pi1.T @ length_in)
    print("Exact: ", 3 * np.sum(pi1) + 3 * pi1.T @ R @ np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0]))

    # n0 FOR M2 1
    n0 = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]

    # n0 FOR M2 2
    # n0 = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    print(n0 @ pi0.T + N * pi1.T @ np.linalg.inv(np.eye(R.shape[0]) - R) @ np.ones(R.shape[0]))


if __name__ == "__main__":
    A_0, A_1, A_2 = m2a_a()
    print("A0\n", A_0, "\nA1/B11\n", A_1, "\nA2\n", A_2)
    b_00, b_01, b_10 = m2a_b(A_0, A_2)
    print("B00\n", b_00, "\nB01\n", b_01, "\nB10\n", b_10)

    gamma = get_gamma(A_1)
    R_ = get_r(A_0, A_1, A_2, gamma, iterations=1000)
    print("R Validation: A_0 + R @ A_1 + R^2 @ A_2 = 0\n", A_0 + R_ @ A_1 + R_ ** 2 @ A_2)
    print("R\n", R_)
    # print("Run1")
    # pi0, pi1 = get_pi01(b_00, b_10, b_01, A_1, A_2, R_, gamma)
    print("Run2")
    pi0, pi1 = get_pi01_T(b_00, b_10, b_01, A_1, A_2, R_)
    print("\n\n\npi0\n", pi0, "\npi1", pi1)
    plist = validity_check(pi0, pi1.T, R_)
    plot_prob_list(plist)
    # TODO Check if R is calculated correctly + everything around calc pi1.
    # TODO Check if all pi's sum to 1 ^
    # TODO Implementation of EL // EW // ES
    mean_val_analysis(pi0, pi1, R_)

    # pin = get_distr(1, pi1, R)
    # print("Found pi distribution:")
    # print("pi_0:\n", pi0, "\n", "pi_1:\n", pi1)

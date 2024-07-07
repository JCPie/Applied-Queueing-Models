import numpy as np

import numpy as np

stations = ["0: ",  "1: SD", "2: Bb", "3: Bl", "4: Bf", "5: AGV", "6: ADBf", "7: A", "8: ABf", "9: D", "9: 10", "11: Pa", "12: DBb", "13: DBl", "14: DBf"]
O = 0
SD = 1
Bb = 2
Bl = 3
Bf = 4
AGV = 5
ADBf = 6
A = 7
ABf = 8
D = 9
P = 10
Pa = 11
DBb = 12
DBl = 13
DBf = 14

n = len(stations)

servers = [1, 3, 1, 1, 1, 1, 1, 2, 1, np.inf, 2, 1, 1, 1, 1]

batch_size = 5

lam_b = 6/60
lam_l = 6/60
lam_f = 3/60
beta_b = lam_b/batch_size
beta_l = lam_l/batch_size
beta_f = lam_f/batch_size


arrival_rate_b = np.array([beta_b, lam_b, lam_b, 0, 0, 2 * beta_b, 0, 0, 0, beta_b, beta_b, beta_b, beta_b, 0, 0])
arrival_rate_l = np.array([beta_l, lam_l, 0, lam_l, 0, 2 * beta_l, 0, 0, 0, 2 * beta_l, 2 * beta_l, beta_l, 0, beta_l, 0])
arrival_rate_f = np.array([beta_f, lam_f, 0, 0, lam_f, 2 * beta_f + beta_f, beta_f, lam_f, lam_f, beta_f + beta_f, beta_f + beta_f, 0, 0, 0, beta_f])

departure_rate_b = np.array([beta_b, lam_b, beta_b, 0, 0, 2 * beta_b, 0, 0, 0, beta_b, beta_b, beta_b, beta_b, 0, 0])
departure_rate_l = np.array([beta_l, lam_l, 0, beta_l, 0, 2 * beta_l, 0, 0, 0, 2 * beta_l, 2 * beta_l, beta_l, 0, beta_l, 0])
departure_rate_f = np.array([beta_f, lam_f, 0, 0, beta_f, 2 * beta_f + beta_f, lam_f, lam_f, beta_f, beta_f + beta_f, beta_f + beta_f, 0, 0, 0, beta_f])

expected_service_time_b = np.array([0, 8, (batch_size - 1) / (2*lam_b), 0, 0, 0, 0, 0, 0, 120, 18, 4, 0, 0, 0])
expected_service_time_l = np.array([0, 8, 0, (batch_size - 1) / (2*lam_l), 0, 0, 0, 0, 0, 120, 18, 4, 0, 0, 0])
expected_service_time_f = np.array([0, 8, 0, 0, (batch_size - 1) / (2*lam_f), 0, 0, 35, (batch_size - 1) / (2*lam_f), 120, 6, 0, 0, 0, 0])

SCV_b = np.array([0, 3/2, 3/40, 0, 0, 0, 0, 0, 0, 0, 13/18, 2, 0, 0, 0])
SCV_l = np.array([0, 3/2, 0, 3/40, 0, 0, 0, 0, 0, 0, 13/18, 2, 0, 0, 0])
SCV_f = np.array([0, 3/2, 0, 0, 3/40, 0, 0, 1/2, 1/2, 0, 1/2, 0, 0, 0, 0])


expected_service_time_AGV = 56/13
SCV_AGV = 463/1568


Pb = np.zeros([n,n])
Pb[SD,Bb] += 1
Pb[Bb, AGV] += 1
Pb[AGV, P] += 0.5
Pb[AGV,Pa] += 0.5
Pb[P, D] += 1
Pb[D, AGV] += 1
Pb[Pa, O] += 1
Pb[O, DBb] += 1
Pb[DBb, SD] += 1

Pl = np.zeros([n,n])
Pl[SD,Bl] += 1
Pl[Bl,AGV] += 1
Pl[AGV, P] += 0.5
Pl[AGV, Pa] += 0.5
Pl[P, D] += 1
Pl[D, P] += 0.5
Pl[D, AGV] += 0.5
Pl[Pa, O] += 1
Pl[O, DBl] += 1
Pl[DBl, SD] += 1

Pf = np.zeros([n,n])
Pf[SD, Bf] += 1
Pf[Bf, AGV] += 1
Pf[AGV, P] += (beta_f + beta_f) / (2 * beta_f + beta_f)
Pf[AGV, ADBf] += (beta_f) / (2 * beta_f + beta_f)
Pf[ADBf, A] += 1
Pf[A, ABf] += 1
Pf[ABf, AGV] += 1
Pf[P, D] += 1
Pf[D, AGV] += (beta_f) / (beta_f + beta_f)
Pf[D, O] += (beta_f) / (beta_f + beta_f)
Pf[O, DBf] += 1
Pf[DBf, SD] += 1
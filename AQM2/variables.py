import numpy as np
stations = ["0: outside", "1: SD", "2: Bb", "3: Bl", "4: Bf", "5: AGV", "6: DBf", "7: A", "8: D", "9: P", "10: Pa"]
O = 0
SD = 1
Bb = 2
Bl = 3
Bf = 4
AGV = 5
DBf = 6
A = 7
D = 8
P = 9
Pa = 10

n = len(stations)

batch_size = 5

lam_b = 6
lam_l = 6
lam_f = 3
beta_b = lam_b/batch_size
beta_l = lam_l/batch_size
beta_f = lam_f/batch_size
#arrival_rate = [lam_f + beta_b + beta_l, lam_b + lam_l + lam_f, lam_b, lam_l, lam_f, 2 * (beta_b + beta_l + beta_f) + lam_f, beta_f, lam_f, lam_f + beta_l + (beta_b + beta_l + beta_f), lam_f + beta_l + (beta_b + beta_l + beta_f), beta_b + beta_l]
arrival_rate_b = [beta_b, lam_b, lam_b, 0, 0, 2 * beta_b, 0, 0, beta_b, beta_b, beta_b]
arrival_rate_l = [beta_l, lam_l, 0, lam_l, 0, 2 * beta_l, 0, 0, 2 * beta_l, 2 * beta_l, beta_l]
arrival_rate_f = [lam_f, lam_f, 0, 0, lam_f, 2 * beta_f + lam_f, beta_f, lam_f, beta_f + lam_f, beta_f + lam_f, 0]
departure_rate_b = [lam_b, lam_b, beta_b, 0, 0, 2 * beta_b, 0, 0, beta_b, beta_b, beta_b]
departure_rate_l = [lam_l, lam_l, 0, beta_l, 0, 2 * beta_l, 0, 0, 2 * beta_l, 2 * beta_l, beta_l]
departure_rate_f = [lam_f, lam_f, 0, 0, beta_f, 2 * beta_f + lam_f, lam_f, lam_f, beta_f + lam_f, beta_f + lam_f, 0]

##### need to add arrival rate shit #################
expected_service_time_b = [0, 8, (batch_size - 1) / lam_b, 0, 0, 0, 0, 0, 120, 18, 4]
expected_service_time_l = [0, 8, 0, (batch_size - 1) / lam_l, 0, 0, 0, 0, 120, 18, 4]
expected_service_time_f = [0, 8, 0, 0, (batch_size - 1) / lam_f, 0, 0, 35, 120, 6, 0]

SCV_b = [0, 3/2, ]

Pb = np.zeros([n,n])
Pb[O,SD] += 1
Pb[SD,Bb] += 1
Pb[Bb, AGV] += 1
Pb[AGV, P] += 0.5
Pb[AGV,Pa] += 0.5
Pb[P, D] += 1
Pb[D, AGV] += 1
Pb[Pa, O] += 1

Pl = np.zeros([n,n])
Pl[O,SD] += 1
Pl[SD,Bl] += 1
Pl[Bl,AGV] += 1
Pl[AGV, P] += 0.5
Pl[AGV, Pa] += 0.5
Pl[P, D] += 1
Pl[D, P] += 0.5
Pl[D, AGV] += 0.5
Pl[Pa, O] = 1

Pf = np.zeros([n,n])
Pf[O, SD] += 1
Pf[SD, Bf] += 1
Pf[Bf, AGV] += 1
Pf[AGV, P] += (beta_f + lam_f) / (2 * beta_f + lam_f)
Pf[AGV, DBf] += (beta_f) / (2 * beta_f + lam_f)
Pf[DBf, A] += 1
Pf[A, AGV] += 1
Pf[P, D] += 1
Pf[D, AGV] += (beta_f) / (beta_f + lam_f)
Pf[D, O] += (lam_f) / (beta_f + lam_f)



print(Pb)
print(Pl)
print(Pf)

# P = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
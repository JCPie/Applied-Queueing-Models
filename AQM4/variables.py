import numpy as np

# Template for the machine/station array
stations = ["0: O", "1: SD",  "2: AGV", "3: A", "4: D", "5: P", "6: Pa",]
O = 0
SD = 1
AGV = 2
A = 3
D = 4
P = 5
Pa = 6

# Amount of stations
n = len(stations)

# Servers at each station
servers = [1, 3, 1, 2, 25, 2, 1]

# Batch size
batch_size = 5

# Rate variables
lam_b = 6/60
lam_l = 6/60
lam_f = 3/60
beta_b = lam_b/batch_size
beta_l = lam_l/batch_size
beta_f = lam_f/batch_size

# Arrival rates for all classes and all machines
arrival_rate_b = np.array([beta_b, beta_b, 2 * beta_b, 0, beta_b, beta_b, beta_b])
arrival_rate_l = np.array([beta_l, beta_l, 2 * beta_l, 0,  2 * beta_l, 2 * beta_l, beta_l])
arrival_rate_f = np.array([beta_f, beta_f, 2 * beta_f + beta_f, beta_f, beta_f + beta_f, beta_f + beta_f, 0])

# Departure rates for all classes and all machines
departure_rate_b = np.array([beta_b, beta_b, 2 * beta_b, 0, beta_b, beta_b, beta_b])
departure_rate_l = np.array([beta_l, beta_l, 2 * beta_l, 0,  2 * beta_l, 2 * beta_l, beta_l])
departure_rate_f = np.array([beta_f, beta_f, 2 * beta_f + beta_f, beta_f, beta_f + beta_f, beta_f + beta_f, 0])

# Expected service times for all classes and all machines
expected_service_time_b = np.array([1e-10, 40, 0, 0, 120, 18, 4])
expected_service_time_l = np.array([1e-10, 40, 0, 0, 120, 18, 4])
expected_service_time_f = np.array([1e-10, 40, 0, 175, 120, 18, 0])

# SCVs for all classes and all machines
SCV_b = np.array([0, 3/10, 0, 0, 0, 13/18, 2])
SCV_l = np.array([0, 3/10, 0, 0, 0, 13/18, 2])
SCV_f = np.array([0, 3/10, 0, 1/10, 0, 1/6, 0])

# Visit ratios for all classes and all machines
V_b = np.array([1, 1, 2, 0, 1, 1, 1])
V_l = np.array([1, 1, 2, 0, 2, 2, 1])
V_f = np.array([1, 1, 3, 1, 2, 2, 0])

# Values for the AGV
expected_service_time_AGV = 368/121
SCV_AGV = 16383/67712

# Transition matrices for all classes and all machines where they go to O if they would have otherwise left the system and then back to SD
Pb = np.zeros([n,n])
Pb[SD,AGV] += 1
Pb[AGV, P] += 0.5
Pb[AGV,Pa] += 0.5
Pb[P, D] += 1
Pb[D, AGV] += 1
Pb[Pa, O] += 1
Pb[O, SD] += 1

Pl = np.zeros([n,n])
Pl[SD,AGV] += 1
Pl[AGV, P] += 0.5
Pl[AGV, Pa] += 0.5
Pl[P, D] += 1
Pl[D, P] += 0.5
Pl[D, AGV] += 0.5
Pl[Pa, O] += 1
Pl[O, SD] += 1

Pf = np.zeros([n,n])
Pf[SD, AGV] += 1
Pf[AGV, P] += (beta_f + beta_f) / (2 * beta_f + beta_f)
Pf[AGV, A] += (beta_f) / (2 * beta_f + beta_f)
Pf[A, AGV] += 1
Pf[P, D] += 1
Pf[D, AGV] += (beta_f) / (beta_f + beta_f)
Pf[D, O] += (beta_f) / (beta_f + beta_f)
Pf[O, SD] += 1

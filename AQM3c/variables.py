import numpy as np

# Template for which index corresponds to which station
stations = ["0: SD", "1: Bb", "2: Bl", "3: Bf", "4: AGV", "5: DBf", "6: A", "7: D", "8: P", "9: Pa"]
SD = 0
Bb = 1
Bl = 2
Bf = 3
AGV = 4
DBf = 5
A = 6
D = 7
P = 8
Pa = 9

# Number of stations
n = len(stations)

# The amount of servers in each station
servers = [3, np.inf, np.inf, np.inf, 1, np.inf, 2, np.inf, 2, 1]

# Batch size
batch_size = 5

# Rate variables in #/minute
lam_b = 6/60
lam_l = 6/60
lam_f = 3/60
beta_b = lam_b/batch_size
beta_l = lam_l/batch_size
beta_f = lam_f/batch_size
gamma = lam_b + lam_l + lam_f

# Rates at which jobs of the different class enter each station
arrival_rate_b = np.array([lam_b, lam_b, 0, 0, 2 * beta_b, 0, 0, beta_b, beta_b, beta_b])
arrival_rate_l = np.array([lam_l, 0, lam_l, 0, 2 * beta_l, 0, 0, 2 * beta_l, 2 * beta_l, beta_l])
arrival_rate_f = np.array([lam_f, 0, 0, lam_f, 2 * beta_f + lam_f, beta_f, lam_f, beta_f + lam_f, beta_f + lam_f, 0])

# Rates at which jobs of the different class leave each station
departure_rate_b = np.array([lam_b, beta_b, 0, 0, 2 * beta_b, 0, 0, beta_b, beta_b, beta_b])
departure_rate_l = np.array([lam_l, 0, beta_l, 0, 2 * beta_l, 0, 0, 2 * beta_l, 2 * beta_l, beta_l])
departure_rate_f = np.array([lam_f, 0, 0, beta_f, 2 * beta_f + lam_f, lam_f, lam_f, beta_f + lam_f, beta_f + lam_f, 0])

# Visit ratios of each job class
V_b = np.array([1, 1, 0, 0, 1/batch_size, 0, 0, 1/batch_size, 1/batch_size, 1/batch_size])
V_l = np.array([1, 0, 1, 0, 2/batch_size, 0, 0, 2/batch_size, 2/batch_size, 1/batch_size])
V_f = np.array([1, 0, 0, 1, 1 + 1/batch_size, 1/batch_size, 1, 1 + 1/batch_size, 1 + 1/batch_size, 0])

# Expected server time for each class at each station, if the class does not go to a station, ES is 0
expected_service_time_b = np.array([8, (batch_size - 1) / (2*lam_b), 0, 0, 0, 0, 0, 120, 18, 4])
expected_service_time_l = np.array([8, 0, (batch_size - 1) / (2*lam_l), 0, 0, 0, 0, 120, 18, 4])
expected_service_time_f = np.array([8, 0, 0, (batch_size - 1) / (2*lam_f), 0, 0, 35, 120, 6, 0])

# SCV of each class at each station, if the class does not go to a station, SCV is 0
SCV_b = np.array([3/2, 22/360, 0, 0, 0, 0, 0, 0, 13/18, 2])
SCV_l = np.array([3/2, 0, 22/360, 0, 0, 0, 0, 0, 13/18, 2])
SCV_f = np.array([3/2, 0, 0, 22/360, 0, 0, 1/2, 0, 1/2, 0])

# The aggregated values calculated for the AGV
expected_service_time_AGV = 56/13
SCV_AGV = 0.69245008973

# Transition matrix for basic kits
Pb = np.zeros([n,n])
Pb[SD,Bb] += 1
Pb[Bb, AGV] += 1
Pb[AGV, P] += 0.5
Pb[AGV,Pa] += 0.5
Pb[P, D] += 1
Pb[D, AGV] += 1

# Transition matrix for luxury kits
Pl = np.zeros([n,n])
Pl[SD,Bl] += 1
Pl[Bl,AGV] += 1
Pl[AGV, P] += 0.5
Pl[AGV, Pa] += 0.5
Pl[P, D] += 1
Pl[D, P] += 0.5
Pl[D, AGV] += 0.5

# Transition matrix for furniture
Pf = np.zeros([n,n])
Pf[SD, Bf] += 1
Pf[Bf, AGV] += 1
Pf[AGV, P] += (beta_f + lam_f) / (2 * beta_f + lam_f)
Pf[AGV, DBf] += (beta_f) / (2 * beta_f + lam_f)
Pf[DBf, A] += 1
Pf[A, AGV] += 1
Pf[P, D] += 1
Pf[D, AGV] += (beta_f) / (beta_f + lam_f)


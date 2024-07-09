import variables as var
import numpy as np


####### AGGREGATION ######

# Aggregated arival rate by summing the different classes
arrival_rate_agg = var.arrival_rate_b + var.arrival_rate_l + var.arrival_rate_f

print("lambda aggregated: \n ", arrival_rate_agg)


# Aggregated service and squared service times using equations 2.37 and 2.38
expected_service_time_agg = np.zeros(var.n)
expected_service_time_agg_sq = np.zeros(var.n)

for i, s in enumerate(var.expected_service_time_b):
    expected_service_time_agg[i] += (1/arrival_rate_agg[i]) * var.arrival_rate_b[i] * s
    expected_service_time_agg_sq[i] += (1 / arrival_rate_agg[i]) * var.arrival_rate_b[i] * s**2

for i, s in enumerate(var.expected_service_time_l):
    expected_service_time_agg[i] += (1/arrival_rate_agg[i]) * var.arrival_rate_l[i] * s
    expected_service_time_agg_sq[i] += (1 / arrival_rate_agg[i]) * var.arrival_rate_l[i] * s ** 2

for i, s in enumerate(var.expected_service_time_f):
    expected_service_time_agg[i] += (1/arrival_rate_agg[i]) * var.arrival_rate_f[i] * s
    expected_service_time_agg_sq[i] += (1 / arrival_rate_agg[i]) * var.arrival_rate_f[i] * s ** 2

# Adding the aggregated service time of the AGV from exercise 3b
expected_service_time_agg[var.AGV] += var.expected_service_time_AGV
expected_service_time_agg_sq[var.AGV] += var.expected_service_time_AGV**2

print("ES aggregated: \n", expected_service_time_agg)
print("(ES)^2 aggregated: \n",expected_service_time_agg_sq)


# Aggregated SCVs using equation 2.39
SCV_agg = np.zeros(var.n)

for i, SCV in enumerate(var.SCV_b):
    if arrival_rate_agg[i] == 0 or expected_service_time_agg_sq[i] == 0:
        continue
    norm = (1/(arrival_rate_agg[i] * expected_service_time_agg_sq[i]))
    SCV_agg[i] += norm * (var.arrival_rate_b[i] * var.expected_service_time_b[i]**2 * (SCV + 1)) - 1/3

for i, SCV in enumerate(var.SCV_l):
    if arrival_rate_agg[i] == 0 or expected_service_time_agg_sq[i] == 0:
        continue
    norm = (1/(arrival_rate_agg[i] * expected_service_time_agg_sq[i]))
    SCV_agg[i] += norm * (var.arrival_rate_l[i] * var.expected_service_time_l[i]**2 * (SCV + 1)) - 1/3

for i, SCV in enumerate(var.SCV_f):
    if arrival_rate_agg[i] == 0 or expected_service_time_agg_sq[i] == 0:
        continue
    norm = (1/(arrival_rate_agg[i] * expected_service_time_agg_sq[i]))
    SCV_agg[i] += norm * (var.arrival_rate_f[i] * var.expected_service_time_f[i]**2 * (SCV + 1)) - 1/3

# Adding the aggregated SCV of the AGV from exercise 3b
SCV_agg[var.AGV] += 463/1568 + 1

print("SCV aggregated: \n", SCV_agg)


# Aggregating the transition matrix using equation 2.40
P_agg = np.zeros([var.n, var.n])

for i in range(var.n):
    for j in range(var.n):
        P_agg[i,j] += (1/arrival_rate_agg[i]) * (var.arrival_rate_b[i] * var.Pb[i,j] + var.arrival_rate_l[i] * var.Pl[i,j] + var.arrival_rate_f[i] * var.Pf[i,j])

print("P aggregated: \n", P_agg)


####### Analysis / MVA ######

# Implementing equation 2.29
Q = np.zeros([var.n + 1, var.n])
Q[0, 0] = 1
for i in range(1, var.n + 1):
    Q[i, :] = arrival_rate_agg[i - 1] * P_agg[i-1, :]
    for j in range(var.n):
        Q[i,j] = Q[i,j] / arrival_rate_agg[j]

print("Q: \n", Q)

# Implementing equation 2.28
v = np.zeros(var.n)
for i in range(var.n):
    v[i] += 1 / np.sum(Q[:, i]**2)

print("v: \n", v)

# Calculating the rho of the aggregated model, just lambda * ES / c
rho = np.zeros([var.n])
for i in range(var.n):
    rho[i] += arrival_rate_agg[i] * expected_service_time_agg[i] / var.servers[i]

print("rho: \n", rho)

# Implementing equation 2.27
w = np.zeros(var.n)
for i in range(var.n):
    w[i] = 1 / (1 + 4 * ((1 - rho[i])**2) * (v[i] - 1))

print("w: \n", w)

# Implementing equation 2.33
x = np.zeros(var.n)
for i in range(var.n):
    x[i] += 1 + (var.servers[i]**(-0.5)) * (np.max([SCV_agg[i], 0.2]) - 1)

print("x: \n", x)

# Implementing 2.32
a = np.zeros(var.n)
for j in range(var.n):
    a[j] += 1 + w[j] * (Q[0,j] * (j == var.SD) - 1)
    for i in range(var.n):
        a[j] += w[j] * (Q[i,j] * ((1 - P_agg[i,j]) + P_agg[i,j] * (rho[i]**2) * x[i]))

print("a: \n", a)

# Implementing the B matrix as outlined in the report
b_matrix = np.eye(var.n)
for i in range(var.n):
    for j in range(var.n):
        b_matrix[i,j] -= w[i] * P_agg[j,i] * Q[i,j] * (1 - (rho[j]**2))

print("b_matrix: \n", b_matrix)

# Solving for C^2_a
SCV_arrival_agg = np.linalg.solve(b_matrix, a)

print("SCV_arrival_agg: \n", SCV_arrival_agg)

# Implementing equation 2.24
SCV_departure_agg = np.zeros([var.n])
for i in range(var.n):
    SCV_departure_agg[i] += 1 + (1 - (rho[i]**2)) * (SCV_arrival_agg[i] - 1) + (rho[i]**2) / np.sqrt(var.servers[i]) * SCV_agg[i]

print("SCV_departure_agg: \n", SCV_departure_agg)

# Implementing equation 2.34
EW_Q_agg = np.zeros([var.n])
for i in range(var.n):
    EW_Q_agg[i] += ((SCV_arrival_agg[i] + SCV_agg[i])/2) * ((rho[i]**(np.sqrt(2*(var.servers[i] + 1))))/(var.servers[i]*(1 - rho[i]))) * expected_service_time_agg[i]

print("EW_Q_agg: \n", EW_Q_agg)

# IMplementing equation 2.35
EW = EW_Q_agg + expected_service_time_agg

print("EW: \n", EW)

# Implementing EL_i = Lambda_i EW_i
EL = np.zeros(var.n)
for i in range(var.n):
    EL[i] += arrival_rate_agg[i] * EW[i]

print("EL: \n", EL)



####### DISAGGREGATION ######


# Implementing EL_Q,j = EL_j - c_j rho_j
EL_Q = EL
for i in range (var.n):
    if var.servers[i] == np.inf:
        continue
    EL_Q[i] -= var.servers[i] * rho[i]

print("EL_Q: \n", EL_Q)

# Implementing equation 2.41 for all classes
EL_Q_b = np.zeros(var.n)
for i in range(var.n):
    EL_Q_b[i] += (var.arrival_rate_b[i] / arrival_rate_agg[i]) * EL_Q[i]

EL_Q_l = np.zeros(var.n)
for i in range(var.n):
    EL_Q_l[i] += (var.arrival_rate_l[i] / arrival_rate_agg[i]) * EL_Q[i]

EL_Q_f = np.zeros(var.n)
for i in range(var.n):
    EL_Q_f[i] += (var.arrival_rate_f[i] / arrival_rate_agg[i]) * EL_Q[i]

print("EL_Q_b: \n", EL_Q_b)
print("EL_Q_l: \n", EL_Q_l)
print("EL_Q_f: \n", EL_Q_f)

# Implementing equation 2.42 for all classes
EL_b = EL_Q_b
for i in range(var.n):
    EL_b[i] += var.arrival_rate_b[i] * var.expected_service_time_b[i]

EL_l = EL_Q_l
for i in range(var.n):
    EL_l[i] += var.arrival_rate_l[i] * var.expected_service_time_l[i]

EL_f = EL_Q_f
for i in range(var.n):
    EL_f[i] += var.arrival_rate_f[i] * var.expected_service_time_f[i]

print("EL_b: \n", EL_b)
print("EL_l: \n", EL_l)
print("EL_f: \n", EL_f)

# Implementing equation 2.44 for all classes
EW_b = np.zeros(var.n)
for i in range(var.n):
    if var.arrival_rate_b[i] == 0:
        continue
    EW_b[i] = var.V_b[i] * (EL_b[i] / var.arrival_rate_b[i])

EW_l = np.zeros(var.n)
for i in range(var.n):
    if var.arrival_rate_l[i] == 0:
        continue
    EW_l[i] = var.V_l[i] * (EL_l[i] / var.arrival_rate_l[i])

EW_f = np.zeros(var.n)
for i in range(var.n):
    if var.arrival_rate_f[i] == 0:
        continue
    EW_f[i] = var.V_f[i] * (EL_f[i] / var.arrival_rate_f[i])

print("EW_b_i : \n", EW_b)
print("EW_l_i : \n", EW_l)
print("EW_f_i : \n", EW_f)

print("EW_b : \n", np.sum(EW_b))
print("EW_l : \n", np.sum(EW_l))
print("EW_f : \n", np.sum(EW_f))
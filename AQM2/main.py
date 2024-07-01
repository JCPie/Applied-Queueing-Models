import variables as var
import numpy as np

arrival_rate_agg = var.arrival_rate_b + var.arrival_rate_l + var.arrival_rate_f

print("lambda aggregated: \n ", arrival_rate_agg)


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

expected_service_time_agg[var.AGV] += var.expected_service_time_AGV
expected_service_time_agg_sq[var.AGV] += var.expected_service_time_AGV**2

print("ES aggregated: \n", expected_service_time_agg)
print("(ES)^2 aggregated: \n",expected_service_time_agg_sq)

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


SCV_agg[var.AGV] += 463/1568 + 1

print("SCV aggregated: \n", SCV_agg)

P_agg = np.zeros([var.n, var.n])
for i in range(var.n):
    for j in range(var.n):
        P_agg[i,j] += (1/arrival_rate_agg[i]) * (var.arrival_rate_b[i] * var.Pb[i,j] + var.arrival_rate_l[i] * var.Pl[i,j] + var.arrival_rate_f[i] * var.Pf[i,j])

print("P aggregated: \n", P_agg)

Q = np.zeros([var.n + 1, var.n])
Q[0, 0] = 1
for i in range(1, var.n + 1):
    Q[i, :] = arrival_rate_agg[i - 1] * P_agg[i-1, :]
    for j in range(var.n):
        Q[i,j] = Q[i,j] / arrival_rate_agg[j]

print("Q: \n", Q)

v = np.zeros(var.n)
for i in range(var.n):
    v[i] += 1 / np.sum(Q[:, i]**2)

print("v: \n", v)

rho = np.zeros([var.n])
for i in range(var.n):
    rho[i] += arrival_rate_agg[i] * expected_service_time_agg[i] / (var.servers[i] * 60)

print("rho: \n", rho)

w = np.zeros(var.n)
for i in range(var.n):
    w[i] = 1 / (1 + 4 * ((1 - rho[i])**2) * (v[i] - 1))

print("w: \n", w)

x = np.zeros(var.n)
for i in range(var.n):
    x[i] += 1 + (var.servers[i]**(-0.5)) * (np.max([SCV_agg[i], 0.2]) - 1)

print("x: \n", x)

a = np.zeros(var.n)
for j in range(var.n):
    a[j] += 1 + w[j] * (Q[0,j] * SCV_agg[j] - 1)
    for i in range(var.n):
        a[j] += w[j] * (Q[i,j] * ((1 - P_agg[i,j]) + P_agg[i,j] * (rho[i]**2) * x[i]))

print("a: \n", a)

b_matrix = np.eye(var.n)
for i in range(var.n):
    for j in range(var.n):
        b_matrix[i,j] += w[j] * P_agg[i,j] * Q[i,j] * (1 - (rho[i]**2))

print("b_matrix: \n", b_matrix)

SCV_arrival_agg = np.linalg.solve(b_matrix, a)

print("SCV_arrival_agg: \n", SCV_arrival_agg)


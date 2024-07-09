import numpy as np
import variables as var

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

agg_rate_matrix = P_agg
for i in range(var.n):
    agg_rate_matrix[i,:] = arrival_rate_agg[i] * agg_rate_matrix[i,:]
print("Rate matrix: \n", agg_rate_matrix)

V = np.zeros(var.n)
for j in range(var.n):
    V[j] += np.sum(agg_rate_matrix[:,j]) / arrival_rate_agg[0]

print("V: \n", V)

N = 25
EW = np.zeros([N+1, var.n])
Qc = np.zeros([N+1, var.n])
ES_rem = np.zeros(var.n)
for j in range(var.n):
    ES_rem[j] += (((var.servers[j] - 1) / (var.servers[j] + 1)) * (expected_service_time_agg[j] / var.servers[j]) +
                  (2 / (var.servers[j] + 1)) * (1 / var.servers[j]) * (
                              expected_service_time_agg_sq[j] / (2 * expected_service_time_agg[j])))

EL_Q = np.zeros([N+1, var.n])
TH = np.zeros([N+1, var.n])
p = np.zeros([var.n, N+1, N+1])
for j in range(var.n):
    p[j, 0, 0] += 1

#Alg 5
for n in range(1, N+1):
    for j in range(var.n):
        for k in range(int(np.min([var.servers[j], n])), n):
            Qc[n-1, j] += p[j, k, n-1]

        for k in range(int(np.min([var.servers[j] + 1, n])), n):
            EL_Q[n-1, j] += (k - var.servers[j]) * p[j, k, n-1]

        EW[n, j] += Qc[n-1, j] * ES_rem[j] + EL_Q[n-1, j] * (expected_service_time_agg[j] / var.servers[j]) + expected_service_time_agg[j]

    soujourn = 0
    for j in range(var.n):
        soujourn += (V[j] * EW[n, j])

    TH[n, 0] += n / soujourn

    for j in range(1, var.n):
        TH[n, j] += V[j] * TH[n, 0]

    for j in range(var.n):
        p[j, 0, n] = 1
        for k in range(1,n+1):
            p[j, k, n] = (expected_service_time_agg[j] / np.min([var.servers[j], k])) * TH[n, j] * p[j, k-1, n-1]
            p[j, 0, n] -= p[j, k, n]

print("EW: \n", EW)
print("TH: \n", TH)
# print("p: \n", p)
print("ES_rem: \n", ES_rem)

for i, rate in enumerate(TH[:, 0]):
    if rate > arrival_rate_agg[0]:
        print(f"Minimum required cards is: {i}")
        break

THC_M1 = TH[N, 0]
print("TH^C_M+1 : \n", THC_M1)

lambda_in = arrival_rate_agg[0]
q = 1 - (lambda_in / THC_M1)
print("q: \n", q)


EW_LD = np.zeros([N+1, var.n])
Qc_LD = np.zeros([N+1, var.n])
EL_Q_LD = np.zeros([N+1, var.n])
TH_LD = np.zeros([N+1, var.n])
p_LD = np.zeros([var.n, N+1, N+1])
for j in range(var.n):
    p_LD[j, 0, 0] += 1

for n in range(1, N+1):
    EW_LD[n, 0] = (EL_Q_LD[n-1, 0] + Qc_LD[n-1, 0]) / lambda_in + p[0, 0, n-1] * q / lambda_in

    for j in range(1, var.n):
        EW_LD[n, j] += expected_service_time_agg[j]
        for k in range(int(np.min([var.servers[j], n])), n):
            EW_LD[n,j] += (k - var.servers[j] + 1) / (var.servers[j] / expected_service_time_agg[j]) * p[j, k, n-1]

    soujourn = 0
    for j in range(var.n):
        soujourn += V[j] * EW_LD[n, j]
    TH_LD[n, 0] += n / soujourn

    for j in range(1, var.n):
        TH_LD[n, j] += V[j] * TH_LD[n, 0]
    for j in range(1, var.n):
        p_LD[j, 0, n] = 1
        for k in range(1,n+1):
            p_LD[j, k, n] = (expected_service_time_agg[j] / np.min([var.servers[j], k])) * TH_LD[n, j] * p[j, k-1, n-1]
            p_LD[j, 0, n] -= p[j, k, n]

    Qc_LD[n, 0] = (TH_LD[n, 0] / lambda_in) * (Qc_LD[n-1, 0] + p[0, 0, n-1] * q)
    p[0, 0, n] = 1 - Qc_LD[n, 0]
    EL_Q_LD[n, 0] = TH_LD[n, 0] * EW_LD[n, 0]

print("EW_LD: \n", EW_LD)
print("TH_LD: \n", TH_LD)
print(Qc_LD)

##################  DISAGRAGATE ################
rho = np.zeros(var.n)
for i in range(var.n):
    rho[i] = expected_service_time_agg[i] * arrival_rate_agg[i] / var.servers[i]
print("rho: \n", rho)

EL = np.zeros(var.n)
for i in range(var.n):
    EL[i] += arrival_rate_agg[i] * EW_LD[N, i]

print("EL: \n", EL)

EL_Q = EL
for i in range (var.n):
    if var.servers[i] == np.inf:
        continue
    EL_Q[i] -= var.servers[i] * rho[i]

print("EL_Q: \n", EL_Q)

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

EW_b = 0
for i in range(var.n):
    if var.arrival_rate_b[i] == 0:
        continue
    EW_b += var.V_b[i] * (EL_b[i] / var.arrival_rate_b[i])

EW_l = 0
for i in range(var.n):
    if var.arrival_rate_l[i] == 0:
        continue
    EW_l += var.V_l[i] * (EL_l[i] / var.arrival_rate_l[i])

EW_f = 0
for i in range(var.n):
    if var.arrival_rate_f[i] == 0:
        continue
    EW_f += var.V_f[i] * (EL_f[i] / var.arrival_rate_f[i])

print("EW_b : \n", EW_b)
print("EW_l : \n", EW_l)
print("EW_f : \n", EW_f)
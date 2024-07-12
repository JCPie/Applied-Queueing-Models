import numpy as np
import variables as var
import functions as func

####### AGGRAGATE ######


# Aggregating arrival rate by summing the different classes
arrival_rate_agg = var.arrival_rate_b + var.arrival_rate_l + var.arrival_rate_f

print("lambda aggregated: \n ", arrival_rate_agg)


# Aggregating service times and squared service times using equation 2.37 and 2.38
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

# Adding the aggregated service time of the AGV seperate
expected_service_time_agg[var.AGV] += var.expected_service_time_AGV
expected_service_time_agg_sq[var.AGV] += var.expected_service_time_AGV**2

print("ES aggregated: \n", expected_service_time_agg)
print("(ES)^2 aggregated: \n",expected_service_time_agg_sq)

# Aggregating the SCVs using equation 2.39
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

# Adding the aggregated SCV of the AGV separate
SCV_agg[var.AGV] += 463/1568 + 1

print("SCV aggregated: \n", SCV_agg)

# Aggregating the transition matrix using equation 2.40
P_agg = np.zeros([var.n, var.n])
for i in range(var.n):
    for j in range(var.n):
        P_agg[i,j] += (1/arrival_rate_agg[i]) * (var.arrival_rate_b[i] * var.Pb[i,j] + var.arrival_rate_l[i] * var.Pl[i,j] + var.arrival_rate_f[i] * var.Pf[i,j])

print("P aggregated: \n", P_agg)

# Rate matrix for calculating the aggregated V
agg_rate_matrix = P_agg
for i in range(var.n):
    agg_rate_matrix[i,:] = arrival_rate_agg[i] * agg_rate_matrix[i,:]
print("Rate matrix: \n", agg_rate_matrix)

# Calculating the aggregated visiting ratios
V = np.zeros(var.n)
for j in range(var.n):
    V[j] += np.sum(agg_rate_matrix[:,j]) / arrival_rate_agg[0]

print("V: \n", V)



####### Analysis / MVA ######


####### ALGORITHM 5 ######

# Step 1 initialization of the variables
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

# Step 2 cycle through n starting at n=1
for n in range(1, N+1):

    # Step 3 Compute EW
    for j in range(var.n):
        for k in range(int(np.min([var.servers[j], n])), n):
            Qc[n-1, j] += p[j, k, n-1]

        for k in range(int(np.min([var.servers[j] + 1, n])), n):
            EL_Q[n-1, j] += (k - var.servers[j]) * p[j, k, n-1]

        EW[n, j] += Qc[n-1, j] * ES_rem[j] + EL_Q[n-1, j] * (expected_service_time_agg[j] / var.servers[j]) + expected_service_time_agg[j]

    #Step 4 compute TH_0 and subsequently TH_i
    soujourn = 0
    for j in range(var.n):
        soujourn += (V[j] * EW[n, j])

    TH[n, 0] += n / soujourn

    for j in range(1, var.n):
        TH[n, j] += V[j] * TH[n, 0]

    # Step 5 compute p_j (k|n)
    for j in range(var.n):
        p[j, 0, n] = 1
        for k in range(1,n+1):
            p[j, k, n] = (expected_service_time_agg[j] / np.min([var.servers[j], k])) * TH[n, j] * p[j, k-1, n-1]
            p[j, 0, n] -= p[j, k, n]

print("EW: \n", EW)
print("TH: \n", TH)
# print("p: \n", p)
print("ES_rem: \n", ES_rem)


# Find the minimal required cards
min_req_cards = 0

for i, rate in enumerate(TH[:, 0]):
    if rate > arrival_rate_agg[0]:
        min_req_cards = i
        break

# Set the throughput of the complement network
THC_M1 = TH[N, 0]
print("TH^C_M+1 : \n", THC_M1)

# Incoming lambda is the same as the rate into O
lambda_in = arrival_rate_agg[0]

# Calculate q from algorithm 7
q = 1 - (lambda_in / THC_M1)
print("q: \n", q)


####### ALGORITHM 5 #######

# Step 1 initialization
EW_LD = np.zeros([N+1, var.n])
Qc_LD = np.zeros([N+1, var.n])
EL_Q_LD = np.zeros([N+1, var.n])
TH_LD = np.zeros([N+1, var.n])
p_LD = np.zeros([var.n, N+1, N+1])
for j in range(var.n):
    p_LD[j, 0, 0] += 1

# Step 2 cycle through n starting at n = 1
for n in range(1, N+1):

    # Calculate EW_0 separately using equation 4.18
    EW_LD[n, 0] = (EL_Q_LD[n-1, 0] + Qc_LD[n-1, 0]) / lambda_in + p_LD[0, 0, n-1] * q / lambda_in

    # Step 3 calculating EW
    for j in range(1, var.n):
        for k in range(int(np.min([var.servers[j], n])), n):
            Qc_LD[n - 1, j] += p[j, k, n - 1]

        for k in range(int(np.min([var.servers[j] + 1, n])), n):
            EL_Q_LD[n - 1, j] += (k - var.servers[j]) * p[j, k, n - 1]

        EW_LD[n, j] += Qc_LD[n - 1, j] * ES_rem[j] + EL_Q_LD[n - 1, j] * (expected_service_time_agg[j] / var.servers[j]) + \
                    expected_service_time_agg[j]

    # Step 4 calculating TH_0 and subsequently TH_i
    soujourn = 0
    for j in range(var.n):
        soujourn += V[j] * EW_LD[n, j]
    TH_LD[n, 0] += n / soujourn

    for j in range(1, var.n):
        TH_LD[n, j] += V[j] * TH_LD[n, 0]

    # Step 5 calculate p_j(k|n)
    for j in range(1, var.n):
        p_LD[j, 0, n] = 1
        for k in range(1,n+1):
            p_LD[j, k, n] = (expected_service_time_agg[j] / np.min([var.servers[j], k])) * TH_LD[n, j] * p[j, k-1, n-1]
            p_LD[j, 0, n] -= p[j, k, n]

    # Calculate Q_0(n) using equation 4.19
    Qc_LD[n, 0] = (TH_LD[n, 0] / lambda_in) * (Qc_LD[n-1, 0] + p[0, 0, n-1] * q)

    # Calculate p_LD_0(0|n) using equation 4.20
    p_LD[0, 0, n] = 1 - Qc_LD[n, 0]
    p_LD[0, 1, n] = (q / lambda_in) / np.min([var.servers[0], k]) * TH_LD[n, 0] * p[0, 1 - 1, n - 1]
    for k in range(2, n + 1):
        p_LD[0, k, n] = 1 / (lambda_in * np.min([var.servers[0], k])) * TH_LD[n, 0] * p[0, k - 1, n - 1]
    # Calculate EL_0 using equation 4.21
    EL_Q_LD[n, 0] = TH_LD[n, 0] * EW_LD[n, 0]

print("EW_LD: \n", EW_LD)
print("TH_LD: \n", TH_LD)
print("QC_LD: \n", Qc_LD)



##################  DISAGRAGATE ################


# Find rho using ES * lambda / c
rho = np.zeros(var.n)
for i in range(var.n):
    rho[i] = expected_service_time_agg[i] * arrival_rate_agg[i] / var.servers[i]
print("rho: \n", rho)

# Find EL using little's law and EW
EL = np.zeros(var.n)
for i in range(var.n):
    EL[i] += arrival_rate_agg[i] * EW_LD[N, i]

print("EL: \n", EL)
print("sum EL: \n", np.sum(EL))

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

# Print answers
print("\n \n #### Answers for 4a ##### \n")

print(f"Minimum required cards is: {min_req_cards}")

print("EW_b : \n", EW_b)
print("EW_l : \n", EW_l)
print("EW_f : \n", EW_f)



###### 4b ######



# Variables for EW and TH for all n for both with and without LD for all classes
EW_LD_n_b = np.ones(N+1)
EW_LD_n_l = np.ones(N+1)
EW_LD_n_f = np.ones(N+1)
EW_n_b = np.ones(N+1)
EW_n_l = np.ones(N+1)
EW_n_f = np.ones(N+1)
TH_LD_n_b = np.ones(N+1)
TH_LD_n_l = np.ones(N+1)
TH_LD_n_f = np.ones(N+1)
TH_n_b = np.ones(N+1)
TH_n_l = np.ones(N+1)
TH_n_f = np.ones(N+1)

# Disaggregating the throughput for all n
for n in range(1, N+1):
    EW_LD_n_b[n], EW_LD_n_l[n], EW_LD_n_f[n] = func.disaggragate(EW_LD, arrival_rate_agg, expected_service_time_agg, n)
    EW_n_b[n], EW_n_l[n], EW_n_f[n] = func.disaggragate(EW, arrival_rate_agg, expected_service_time_agg, n)
    TH_n_b = n / EW_n_b
    TH_n_l = n / EW_n_l
    TH_n_f = n / EW_n_f
    TH_LD_n_b = n / EW_LD_n_b
    TH_LD_n_l = n / EW_LD_n_l
    TH_LD_n_f = n / EW_LD_n_f

# Stock size in batches
S = 2
# Tolerance for till what pi(m_hat) we go
tol = 1e-10

# Finding pi for both basic and luxury
pi_b = func.pi(var.arrival_rate_b[0], tol, TH_n_b, S, N)
pi_l = func.pi(var.arrival_rate_l[0], tol, TH_n_f, S, N)
pi_LD_b = func.pi(var.arrival_rate_b[0], tol, TH_LD_n_b, S, N)
pi_LD_l = func.pi(var.arrival_rate_l[0], tol, TH_LD_n_l, S, N)

print("pi_b : \n", pi_b)
print("pi_l : \n", pi_l)
print("pi_LD_b : \n", pi_LD_b)
print("pi_LD_l : \n", pi_LD_l)


print("\n \n #### Answers for 4b ##### \n")
# Summing from 1 to S for the FR
print("FR basic: \n",np.sum(pi_b[0:S-1]))
print("FR luxury: \n", np.sum(pi_l[0:S-1]))

# Finding expected response time for basic
# i = S means the stock is empty so non-zero response times
expected_response_time_b = 0
for i in range(S + 1, np.size(pi_LD_b)):
    # Expected response time for different amounts in the shop, using our naive distribution
    expected_response_time_b += pi_LD_b[i] * np.max([EW_b / np.min([i + 1, N]), EW_b - i / var.arrival_rate_b[0]])
print("Expected response time basic: \n", expected_response_time_b)

# Finding expected response time for luxury
# i = S means the stock is empty so non-zero response times
expected_response_time_l = 0
for i in range(S + 1, np.size(pi_LD_l)):
    # Expected response time for different amounts in the shop, using our naive distribution
    # expected_response_time_l += pi_LD_l[i] * EW_l / np.min([i + 1, N])
    expected_response_time_l += pi_LD_l[i] * np.max([EW_l / np.min([i + 1, N]), EW_l - i / var.arrival_rate_l[0]])
print("Expected response time luxury: \n", expected_response_time_l)

print("Chance of not having to wait, basic: \n ", np.sum(pi_LD_b[0:S]))
print("Chance of not having to wait, luxury: \n ", np.sum(pi_LD_l[0:S]))
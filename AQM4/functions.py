import numpy as np
import variables as var

def disaggragate(EW_LD, arrival_rate_agg, expected_service_time_agg, N):
    # Find rho using ES * lambda / c
    rho = np.zeros(var.n)
    for i in range(var.n):
        rho[i] = expected_service_time_agg[i] * arrival_rate_agg[i] / var.servers[i]

    # Find EL using little's law and EW
    EL = np.zeros(var.n)
    for i in range(var.n):
        EL[i] += arrival_rate_agg[i] * EW_LD[N, i]

    # Implementing EL_Q,j = EL_j - c_j rho_j
    EL_Q = EL
    for i in range(var.n):
        if var.servers[i] == np.inf:
            continue
        EL_Q[i] -= var.servers[i] * rho[i]


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

    return EW_b, EW_l, EW_f

# Function for calculating pi using method from page 91
def pi(lambda_in, tol, TH, S, N):
    # Start index at 0 and m_hat at S
    index = 0
    m_hat = S
    pi = np.array([1])
    while pi[index] > tol:
        # Add pi(m_hat-1)
        pi = np.append(pi, pi[index] * lambda_in / TH[np.min([S - m_hat + 1, S - N])])

        index += 1
        m_hat -= 1

    # Return normalized pi, the array is of the form:
    #pi(m_hat = S, S-1, S-2, ...)
    return pi/np.sum(pi)
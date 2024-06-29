## Calculate P0
def get_p0(N: int = 3, theta: float = 1, arrival_rate: float = 1, service_rate: float = 2) -> float:
    """
    :param N: Number of max. jobs (PAC)
    :param theta: probability of leaving the system after the "computer"
    :param arrival_rate: arrival rate = 1
    :param service_rate: service rate = 2
    :return: P_0
    """
    base = 0
    for n in range(N):
        base += ((n + 1) * (arrival_rate / (theta * service_rate)) ** n +
                 arrival_rate * N * (N + 1) * (arrival_rate / (theta * service_rate)) ** (N - 1) *
                 1 / (theta * service_rate * N - arrival_rate * (N + 1)))
    return 1 / base


def sojourn_time(P0: float, theta: float = 1, N: int = 3, arrival_rate: float = 1, service_rate: float = 2):
    alpha = (N + 1) / N

    temp0 = 1 / arrival_rate
    temp1 = P0 * (N + 1) * (1 / alpha ** N)
    temp2 = (alpha * arrival_rate) / (theta * service_rate)
    temp3 = 1 / ((alpha * arrival_rate) / (theta * service_rate) - 1) ** 2
    sojourn = temp0 * temp1 * temp2 * temp3

    return sojourn


def waiting_time(N: int = 3, theta: float = 1, arrival_rate: float = 1, service_rate: float = 2) -> float:
    temp0 = theta * service_rate * N * (N + 1)
    temp1 = (theta * service_rate * N - arrival_rate * (N + 1)) ** 2
    temp2 = 0
    for n in range(N):
        temp2 += (((n + 1) / (N + 1)) * (theta * service_rate / arrival_rate) ** (N + 1 - n) +
                  theta * service_rate * N * (theta * service_rate * N - arrival_rate * (N + 1)))
    return temp0 / (temp1 * temp2)


if __name__ == '__main__':
    P_0 = get_p0()
    soj_time = sojourn_time(P0=P_0)
    wait_time = waiting_time()
    print("If we assume that the rates were given in rate/hour:\n "
          "Waiting Time in External queue: {0:.3g} minutes\n".format(wait_time * 60),
          "Sojourn Time in system: {0:.3g} minutes".format(soj_time * 60))
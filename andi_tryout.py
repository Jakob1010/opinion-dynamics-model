import numpy as np
import matplotlib.pyplot as plt
import random
import math


def initialize_grid(n):
    dim = math.isqrt(n)
    np.random.seed(123)
    res = np.random.normal(0, 0.5, size=(dim, dim))
    res = np.clip(res, -1, 1)
    return res


def get_random_agent(n):
    upper = math.isqrt(n) - 1
    x = random.randint(0, upper)
    y = random.randint(0, upper)
    return x, y


def adjust_opinions(opinion1, opinion2, tau, mu):
    if abs(opinion1 - opinion2) > tau:
        return opinion1, opinion2
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2


def plot_grid(grid, timesteps, mu, tau):
    cmap = plt.get_cmap("bwr")
    img = plt.imshow(grid, interpolation='nearest', cmap=cmap, origin='lower')
    plt.colorbar(img, cmap=cmap)
    plt.title(f"Result after {timesteps=}, {tau=}, {mu=}")
    plt.show()


def update_grid(n, grid, tau, mu):
    # 1. pick two agents at random
    x1, y1 = get_random_agent(n)
    x2, y2 = get_random_agent(n)

    # redraw second agent so nobody talks with itself
    while x2 == x1 and y2 == y1:
        x2, y2 = get_random_agent(n)
    grid[x1][y1], grid[x2][y2] = adjust_opinions(grid[x1][y1], grid[x2][y2], tau, mu)


def run_simulation(n_agents, timesteps, tau, mu):
    t = timesteps
    grid = initialize_grid(n_agents)
    while t > 0:
        update_grid(n_agents, grid, tau, mu)
        t = t - 1
    plot_grid(grid, timesteps, mu, tau)


if __name__ == '__main__':
    mus = np.linspace(0, 1, num=5)
    taus = np.linspace(0, 1, num=5)
    for tau in taus:
        for mu in mus:
            print(f"{mu=}, {tau=}")
            run_simulation(
                n_agents=1000,
                timesteps=20000,
                tau=tau,
                mu=mu
            )

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from datetime import datetime

# ================= #
# define parameters #
# ================= #

n = 1000 # number of agents
t = 1000 # number of timesteps
t_init = t
taus = np.linspace(0.01, 1, num=5) # threshold > 0
mus = np.linspace(0.1, 0.5, num=4) # adjustment parameter 0 < µ ≤ 0.5
show_logs = False # whether additional log messages should be displayed
show_animation = False # whether the grid updates should be animated
update_interval = 5 # update interval in ms (only relevant if show_animation is true)
save_endresult = True # whether to store graphical representation of endresult (or endresults for each combination of tau and mu) as image

def log(message):
    if show_logs: print(message)

def initialize_grid(n,lo,hi):
    """Initializes a grid of agents.

    ### Parameters
    n: number of agents. Will be approximated if square root of n is not whole number, as a whole quadratic grid will always be filled.
    lo: lower boundary for initial opinion.
    hi: upper boundary for initial opinion.

    """
    dim = math.isqrt(n)

    # use seed to get same grid for each fn call if input params not modified
    np.random.seed(42)
    return np.random.uniform(lo, hi, size=(dim, dim))


def get_random_agent(n):
    x,y = random.sample(range(0, math.isqrt(n)), 2)
    return (x,y)


def adjust_opinions(mu, opinion1, opinion2):
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2


def update_grid(grid, img, tau, mu):
    global t
    log(f'timesteps left: {t}')
    
    # 1. pick two agents at random
    x1, y1 = get_random_agent(n)
    x2, y2 = get_random_agent(n)
    log(f'agent 1 ({x1},{y1}): {grid[x1][y1]}')
    log(f'agent 2 ({x2},{y2}): {grid[x2][y2]}')

    # 2. compare opinions
    difference = abs(grid[x1][y1] - grid[x2][y2])
    log(f'opinion difference: {difference}')

    # 3. (check if difference is smaller than threshold)
    #    (check number of timestamps)
    if (difference <= tau) and (t >= 0):
        # 4. adjust opinions 
        grid[x1][y1], grid[x2][y2] = adjust_opinions(grid[x1][y1],grid[x2][y2], mu) 
        log('adjusted opinions:')
        log(f'{grid[x1][y1]}|{grid[x2][y2]}')


    # update data
    t =  t - 1

    if t == 0 and save_endresult:
        save_gridimg(grid, img, tau, mu)
        

def save_gridimg(grid, img, tau, mu):
    print('\nsaving figure')
    print(f't: {t}')
    print(f'tau: {tau}')
    print(f'mu: {mu}\n')

    img.set_data(grid)

    filename = f't-init{t_init}_t-curr{t}_tau{str(round(tau, 2)).replace(".", "_")}_mu{str(round(mu, 2)).replace(".", "_")}__{datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}'
    plt.savefig(filename)

def draw_grid(frameNum, img, grid, tau, mu):
    # TODO: figure out how to make drawing independent of update
    update_grid(grid, img, tau, mu)
    img.set_data(grid)
    grid[:] = grid[:]
    return img

def main():
    print("start opinion dynamics model....")
    global t# TODO: learn about variable scopes in Python, rewrite this mess - pass less variables around like crazy

    # set up animation
    fig, ax = plt.subplots()
    # initialize grid with n agents
    grid = initialize_grid(n, -1, 1)
    img = ax.imshow(grid, interpolation='nearest', cmap="bwr")

    for tau in taus:
        for mu in mus:
            t = t_init
            print(f't reset to {t}')
            #reset grid
            grid = initialize_grid(n, -1, 1)

            if show_animation:
                anim = animation.FuncAnimation(fig, draw_grid, fargs=(img, grid, tau, mu),
                                        frames=60,
                                        interval=update_interval)
            else:
                while t > 0:
                    update_grid(grid, img, tau, mu)
                img.set_data(grid)
    plt.show()


if __name__ == "__main__":
    main()

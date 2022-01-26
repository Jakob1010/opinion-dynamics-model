from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# ================= #
# define parameters #
# ================= #

n = 1000 # number of agents
t = 20000 # number of timesteps
tau = 0.8 # threshold > 0
mu = 0.4 # adjustment parameter 0 < µ ≤ 0.5
show_logs = False # whether additional log messages should be displayed
show_animation = False # whether the grid updates should be animated
update_interval = 5 # update interval in ms (only relevant if show_animation is true)

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
    rng = np.random.default_rng(12345)
    return rng.uniform(lo, hi, size=(dim, dim))


def get_random_agent(n):
    x,y = random.sample(range(0, math.isqrt(n)), 2)
    return (x,y)


def adjust_opinions(opinion1, opinion2):
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2


def update_grid(grid):
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
        grid[x1][y1], grid[x2][y2] = adjust_opinions(grid[x1][y1],grid[x2][y2]) 
        log('adjusted opinions:')
        log(f'{grid[x1][y1]}|{grid[x2][y2]}')


    # update data
    t =  t - 1
        

def draw_grid(frameNum, img, grid):
    # TODO: figure out how to make drawing independent of update
    update_grid(grid)
    img.set_data(grid)
    grid[:] = grid[:]
    return img

def main():
    print("start opinion dynamics model....")
    # initialize grid with n agents
    grid = initialize_grid(n, 0, 1)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    if show_animation:
        anim = animation.FuncAnimation(fig, draw_grid, fargs=(img, grid),
                                frames=60,
                                interval=update_interval)
    else:
        while t > 0:
            update_grid(grid)
        img.set_data(grid)

    plt.show()


if __name__ == "__main__":
    main()

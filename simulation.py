from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# ================= #
# define parameters #
# ================= #

N = 1000  # number of agents
t = 20000  # number of timesteps
update_interval = 5  # update interval in ms
tau = 0.8  # threshold > 0
mu = 0.4  # adjustment parameter 0 < µ ≤ 0.5


def initialize_grid(n,lo,hi):
    # generate random numbers
    rng = np.random.default_rng(12345)
    random_numbers = []
    for i in range(0,N):
     x = round(random.uniform(lo, hi), 2)
     random_numbers.append(x)

    # return grid with n*n agents
    return np.random.choice(random_numbers, n*n).reshape(n, n)


def get_random_agent(n):
    x, y = random.sample(range(0, n), 2)
    return (x,y)


def adjust_opinions(opinion1, opinion2):
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2


def update_grid(frameNum, img, grid, n):
    global t
    print('timestemps left: ',str(t))
    
    # 1. pick two agents at random
    x1, y1 = get_random_agent(n)
    x2, y2 = get_random_agent(n)
    print('agent1 (', str(x1), '|', str(y1), ') -> ',grid[x1][y1])
    print('agent2 (', str(x2), '|', str(y2), ') -> ',grid[x2][y2])

    # 2. compare opinions
    difference = abs(grid[x1][y1] - grid[x2][y2])
    print('opinion difference: ', difference)

    # 3. (check if difference is smaller than threshold)
    #    (check number of timestamps)
    if (difference <= tau) and (t >= 0):
        # 4. adjust opinions 
        grid[x1][y1], grid[x2][y2] = adjust_opinions(grid[x1][y1],grid[x2][y2]) 
        print('adjusted opinions:')
        print(str(grid[x1][y1]),'|', str(grid[x2][y2]))


    # update data
    t =  t - 1
    img.set_data(grid)
    grid[:] = grid[:]
    return img,


def main():
    print("start opinion dynamics model....")

    # initalize grid with ~ N agents
    n = int(math.sqrt(N)) + 1
    grid = initialize_grid(n, 0, 1)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_grid, fargs=(img, grid, n),
                                  frames = 10,
                                  interval=update_interval,
                                  save_count=50)
    plt.show()



if __name__ == "__main__":
    main()

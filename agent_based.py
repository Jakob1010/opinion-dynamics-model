import matplotlib.pyplot as plt
import random

from sklearn import neighbors

from utils import *
from classes import Grid, Agent


def simulate_step(grid, agents, mu, tau, neighborhood, timestep):
    available_neighborhoods = ['Random', 'Moore', 'Moore 2', 'Von Neumann', 'Von Neumann 2']
    if neighborhood not in available_neighborhoods:
        raise ValueError(f"Invalid neighborhood: ({neighborhood}) passed, expected one of: {available_neighborhoods}")

    agent1 = random.choice(agents)
    x, y = agent1.get_coords()
    agent2 = grid.get_random_neighbour(x, y, neighborhood)
    op1 = agent1.get_opinion()
    op2 = agent2.get_opinion()
    op1, op2 = adjust_opinions(op1, op2, mu, tau)
    agent1.update_opinion(op1, timestep)
    agent2.update_opinion(op2, timestep)


def do_movement_phase(grid, agents, neighborhood):
    """
    Implements the movement phase where agents can switch places
    For now two random agents switch places, maybe there are more sophisticated ways?  --> TODO
    """
    if random.randint(1, 10) != 1:  # only swap around every nth step
        return
    agent1 = random.choice(agents)
    x1, y1 = agent1.get_coords()
    agent2 = grid.get_random_neighbour(x1, y1, neighborhood)
    x2, y2 = agent2.get_coords()
    grid.swap_positions(x1, y1, x2, y2)


def ffill_all_remaining_agents(agents, timesteps):
    """
    This step is needed at the end of the simulation as it is ffilling the temporal data
    for each agent up to the last timestep
    """
    for agent in agents:
        if len(agent.get_temporal_opinions()) < timesteps + 1: # plus one bc of initial opinion
            agent.update_opinion(agent.get_opinion(), timesteps)  # this ffills all the agents temporial data to be complete


def do_param_sweep(mus, taus, n, max_t, movement_phase):
    for r, mu in enumerate(mus):
        for c, tau in enumerate(taus):
            print(f"Simulation #{len(mus)*r+c}")
            agents, grid = do_one_run(mu, tau, n, max_t, movement_phase) 
            plt.subplot(len(mus), len(taus), len(mus)*r+c+1)
            plot_grid(grid.get_raw_opinions(), f"{mu=}, {tau=}, {neighborhood=}, movement={movement_phase}")
            # TODO add more plots if needed


def do_one_run(mu, tau, n, max_t, movement_phase):
    random.seed(1234567890)
    agents = []
    grid = Grid(n)
    # fill grid with newly created agents
    for x in range(grid.max_index + 1):  # every row
        row = []
        for y in range(grid.max_index + 1):  # every column
            new_agent = Agent(random.uniform(-1, 1), x, y, id=x*(grid.max_index+1)+y)
            agents.append(new_agent)
            row.append(new_agent)
        grid.data.append(row)
    t = 0
    while t < max_t:
        t += 1
        simulate_step(grid, agents, mu, tau, neighborhood, t)
        if movement_phase:
            do_movement_phase(grid, agents, neighborhood)
    ffill_all_remaining_agents(agents, max_t)
    return agents, grid


if __name__ == '__main__':
    # adapt parameters here
    n = 1089  # number of agents, in case of grid MUST be percect sqaure (33x33=1089)
    max_t = 20000
    neighborhood = "Moore 2"  # used for opinion adjustement as well as possible movement phase
    mu = 0.5
    tau = 1

    param_sweep = True
    mus = [0.1, 0.2, 0.3, 0.5]
    taus = [0.5, 0.75, 1, 1.5]

    movement_phase = True
    # End of parameters

    if param_sweep:
        # Code for parameter sweep:
        do_param_sweep(mus, taus, n, max_t, movement_phase)
        plt.show()
    else:
        # Code for one run:
        agents, grid = do_one_run(mu, tau, n, max_t, movement_phase)
        plot_grid(grid.get_raw_opinions(), f"{mu=}, {tau=}, {neighborhood=}, movement={movement_phase}")
        plot_temporal_opinions(grid.data, "Temporal Opinions of agents")
        plt.show()
    


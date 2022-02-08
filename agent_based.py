import matplotlib.pyplot as plt
import random

from utils import *
from classes import Grid, Agent


def simulate_step(grid, agents, mu, tau, neighborhood, timestep, concurrent_updates):
    if neighborhood not in Grid.available_neighborhoods:
        raise ValueError(f"Invalid neighborhood: ({neighborhood}) passed, expected one of: {Grid.available_neighborhoods}")
    if concurrent_updates:
        for agent in agents:
            if agent.last_interaction_at >= timestep:  # has already been updated by a neighbor
                continue
            x, y = agent.get_coords()
            neighbor = grid.get_random_neighbour(x, y, neighborhood)
            if neighbor.last_interaction_at < timestep:  # neighbor has not updated its opinion in this step
                op1 = agent.get_opinion()
                op2 = neighbor.get_opinion()
                op1, op2 = adjust_opinions(op1, op2, mu, tau)
                agent.update_opinion(op1, timestep)
                neighbor.update_opinion(op2, timestep)
    else:
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
            agent.update_opinion(agent.get_opinion(), timesteps)  # this ffills all the agents temporal data to be complete


def do_param_sweep(mus, taus, n, max_t, movement_phase, concurrent_updates):
    for r, mu in enumerate(mus):
        for c, tau in enumerate(taus):
            print(f"Simulation #{len(mus)*r+c}")
            _, grid = do_one_run(mu, tau, n, max_t, movement_phase, concurrent_updates) 
            plt.subplot(len(mus), len(taus), len(mus)*r+c+1)
            plot_grid(grid.get_raw_opinions(), f"{mu=}, {tau=}")
            plt.suptitle(f"Simulation results\nneighborhood: {neighborhood}, {'' if movement_phase else 'no '}movement, {'concurrent ' if concurrent_updates else 'sequential '} updates")
            # TODO add more plots if needed


def do_one_run(mu, tau, n, max_t, movement_phase, concurrent_updates):
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
        simulate_step(grid, agents, mu, tau, neighborhood, t, concurrent_updates)
        if movement_phase:
            do_movement_phase(grid, agents, neighborhood)
    ffill_all_remaining_agents(agents, max_t)
    return agents, grid


if __name__ == '__main__':
    # adapt parameters here
    n = 1089  # number of agents, in case of grid MUST be perfect square (33x33=1089)
    max_t = 100
    neighborhood = "Moore 2"  # defines the neighborhood from which a particular agent picks some random other agent to "discuss" with and optionally adjust opinions 
    tau = 1 # value in range [0, 2]; describes "maximum distance" between two agent's opinions so that they choose to adjust each other's opinions ("move towards each other")
    mu = 0.5 # value in range [0, 0.5]; defines how "strong" adjustment of opinion between two agents is (if it happens)

    param_sweep = False # if True, all combinations of mus and taus (provided below) are tested in separate simulations with the same initial grid 
    mus = [0.1, 0.2, 0.3, 0.5]
    taus = [0.5, 0.75, 1, 1.5]

    movement_phase = False # whether agents can also move; only relevant for grid simulations

    concurrent_updates = True  # whether to perform updates on all agents every timestep
    # if concurrent_updates is set to True, all agents are updated simultaneously in a single timestep
    # due to several updates happening each timestep rather than just a single one, the simulation advances much more quickly
    # the "concurrent updates" version of a simulation will be much much further advanced after a given number of time steps compared to the "sequential" version

    # End of parameters

    if param_sweep:
        # Code for parameter sweep:
        do_param_sweep(mus, taus, n, max_t, movement_phase, concurrent_updates)
        plt.show()
    else:
        # Code for one run:
        agents, grid = do_one_run(mu, tau, n, max_t, movement_phase, concurrent_updates)
        plot_grid(grid.get_raw_opinions(), "")
        plt.suptitle(f"Simulation results after {max_t} timesteps\n{mu=}, {tau=}, neighborhood: {neighborhood}\n{'' if movement_phase else 'no '}movement, {'concurrent ' if concurrent_updates else 'sequential '} updates")
        plot_temporal_opinions(grid.data, "Temporal Opinions of agents")
        plt.show()
    


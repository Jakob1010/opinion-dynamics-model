import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import math  
import dataclasses
from datetime import datetime

from utils import *
from classes import Grid, Agent, RunConfig


def simulate_step(grid: Grid, agents: list[Agent], mu, tau, neighborhood, timestep, concurrent_updates):
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

def simulate_step_network(agents: list[Agent], mu, tau, timestep, concurrent_updates):
    if concurrent_updates:
        for agent in agents:
            if agent.last_interaction_at >= timestep:  # has already been updated by a neighbor
                continue
            neighbor = random.choice(agent.connections)
            if neighbor is not None and neighbor.last_interaction_at < timestep:  # neighbor has not updated its opinion in this step
                op1 = agent.get_opinion()
                op2 = neighbor.get_opinion()
                op1, op2 = adjust_opinions(op1, op2, mu, tau)
                agent.update_opinion(op1, timestep)
                neighbor.update_opinion(op2, timestep)
    else:
        agent1 = random.choice(agents)
        while len(agent1.connections) == 0:
            agent1 = random.choice(agents)
        agent2 = random.choice(agent1.connections)
        op1 = agent1.get_opinion()
        op2 = agent2.get_opinion()
        op1, op2 = adjust_opinions(op1, op2, mu, tau)
        agent1.update_opinion(op1, timestep)
        agent2.update_opinion(op2, timestep)

def do_movement_phase(grid: Grid, agents: list[Agent], neighborhood):
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

def do_param_sweep(base_run_config: RunConfig, mus, taus):
    run_number = 0
    for mu in mus:
        for tau in taus:
            run_config = dataclasses.replace(base_run_config, mu=mu, tau=tau, run_number=run_number)

            print(f"Starting Simulation #{run_number}")
            _, grid = do_one_run(run_config) 
            print(f"Finished Simulation #{run_number}\n")
            run_number += 1
            plt.subplot(len(mus), len(taus), run_number)
            plot_opinion_grid(grid.get_raw_opinions(), title=f"{mu=}, {tau=}")
            # TODO add more plots if needed

    plt.suptitle(f"Simulation results\nneighborhood: {sim_config.neighborhood}, "+ 
                f"{'' if sim_config.movement_phase else 'no '}movement, "+ 
                f"{'concurrent ' if sim_config.concurrent_updates else 'sequential '} updates")


def do_one_run(run_config: RunConfig):
    # Python doesn't have object destructuring like JavaScript :(
    mu = run_config.mu
    tau = run_config.tau
    n = run_config.n
    max_t = run_config.timesteps
    movement_phase = run_config.movement_phase
    concurrent_updates = run_config.concurrent_updates
    neighborhood = run_config.neighborhood

    random.seed(1234567890)
    agents = []
    grid = Grid(n)
    # fill grid with newly created agents
    for x in range(grid.max_index + 1):  # every row
        row = []
        for y in range(grid.max_index + 1):  # every column
            new_agent = Agent(random.uniform(-1, 1), x, y, id=x*(grid.max_index+1)+y, timesteps=max_t)
            agents.append(new_agent)
            row.append(new_agent)
        grid.data.append(row)
    t = 0
    while t < max_t:
        t += 1
        simulate_step(grid, agents, mu, tau, neighborhood, t, concurrent_updates)
        if movement_phase:
            do_movement_phase(grid, agents, neighborhood)
        if (t % 500 == 0):
            print(f'simulated {t} of {max_t} timesteps')# log progress so we don't panic because we don't see anything

    log_run_summary(agents, run_config)
    return agents, grid

def do_one_run_network(run_config: RunConfig):
    mu = run_config.mu
    tau = run_config.tau
    n = run_config.n
    max_t = run_config.timesteps
    concurrent_updates = run_config.concurrent_updates
    density = run_config.density

    random.seed(1234567890)
    agents = []
    # create random agents
    agents = []
    for i in range(n):
        new_agent= Agent(random.uniform(-1, 1), 0, 0, timesteps=max_t)
        agents.append(new_agent)
    # link to random nodes
    linkamount = math.floor((n*(n-1))/2 * density)
    for i in range(linkamount):
        num1 = 0
        num2 = 0
        while num1 == num2:
            num1 = random.randrange(n)
            num2= random.randrange(n)
        agent1= agents[num1]
        agent2= agents[num2]
        agent1.add_connection(agent2)
        agent2.add_connection(agent1)
    t = 0
    while t < max_t:
        t += 1
        simulate_step_network(agents, mu, tau, t, concurrent_updates)
        
    log_run_summary(agents, run_config)
    return agents

def log_run_summary(agents: list[Agent], run_config: RunConfig):
    init_opinions = pd.Series([a.get_temporal_opinions()[0] for a in agents], name="Initial opinion")
    final_opinions = pd.Series([a.get_temporal_opinions()[run_config.timesteps] for a in agents], name="Final opinion")
    successful_conversations = np.sum([a.get_number_of_opinion_adjustments() for a in agents])
    total_conversations = np.sum([a.get_number_of_conversations() for a in agents])

    conversation_success_rate = successful_conversations/total_conversations
    average_conversation_rate = total_conversations/len(agents)/(run_config.timesteps - 1)

    params = vars(run_config)
    log_content = "\n".join([
        f'\n---Simulation run summary---',
        'Simulation parameters:',
        "\n".join(f"{param}: {value}" for param, value in params.items()),
        f'\nStats:',
        f'Total number of conversations: {total_conversations}',
        f'Total number of successful conversations: {successful_conversations}',
        f'Conversation success rate: {round(conversation_success_rate, 2)}',
        f'Average conversation rate: {round(average_conversation_rate, 2)}',
        f'Stats for opinion development:',
        pd.concat([init_opinions, final_opinions], axis=1).describe().to_string(),
        '\n',
    ])

    print(log_content)
    with open(f'logs/{run_config.id}{"_run_" + str(run_config.run_number) if run_config.run_number >= 0 else ""}.txt', "w") as f:
        f.write(log_content)

    return


if __name__ == '__main__':
    # Note: for explanation of all simulation parameters, have a look into RunConfig

    param_sweep: bool = True
    """
    if True, all combinations of mus and taus are tested in separate simulations with the same initial grid
    """

    # provide values for parameter sweep here (if param_sweep == True)
    mus: list[float] = [0.1, 0.25, 0.5]
    """
    Values to test for mu (if param_sweep == True)
    """

    taus: list[float] = [0.25, 0.5, 0.75, 1]
    """
    Values to test for tau (if param_sweep == True)
    """

    # else, we just use values for mu and tau defined here
    mu = 0.1
    tau = 0.5

    # define config for running the simulation (or simulations in case of parameter sweep) here
    # if param_sweep == true, several simulation runs with each possible combination of mu and tau from above lists (mus and taus) will be executed
    # check documentation for RunConfig for detailed explanation of what parameters mean
    sim_config = RunConfig(
        n=1089,
        timesteps=1000,
        neighborhood="Moore 2",
        mu=mu,
        tau=tau,
        movement_phase=True,
        concurrent_updates=True,
        density=0.2,
        id=f'{"param_sweep" if param_sweep else "single"}_simulation_{datetime.now().strftime("%d-%m-%Y_%H%M_%S")}'
    )

    if param_sweep:
        # Code for parameter sweep
        if sim_config.neighborhood == 'Social':
            raise ValueError('parameter sweep not yet implemented for social network')
        do_param_sweep(sim_config, mus, taus)
        plt.show()
    else:
        # Code for one run:
        if sim_config.neighborhood != 'Social':
            agents, grid = do_one_run(sim_config)
            fig, axs = plt.subplots(1, 2, figsize=(18, 7))#, gridspec_kw={'width_ratios': [0.8, 1]})

            plot_opinion_grid(grid.get_raw_opinions(), title="final grid state", ax=axs[1])
            plt.tight_layout()
            plt.subplots_adjust(top=0.87, left=0.05, bottom=0.1)
            plt.suptitle(f"Simulation results after {sim_config.timesteps} timesteps\nmu={sim_config.mu}, tau={sim_config.tau}" +
                        f", neighborhood: {sim_config.neighborhood}, {'' if sim_config.movement_phase else 'no '}movement,"+
                        f" {'concurrent ' if sim_config.concurrent_updates else 'sequential '} updates")
            plot_agent_opinions(agents, ax=axs[0])
            plt.show()
            
        else:
            agents = do_one_run_network(sim_config)
            fig, axs = plt.subplots(1, 1, figsize=(12, 7))#, gridspec_kw={'width_ratios': [0.8, 1]})
            plt.tight_layout()
            plt.subplots_adjust(top=0.87, left=0.05, bottom=0.1)
            plt.suptitle(f"Simulation results after {sim_config.timesteps} timesteps\nmue={sim_config.mu}, ta={sim_config.tau}" +
                        f", neighborhood: {sim_config.neighborhood}, {'' if sim_config.movement_phase else 'no '}movement,"+
                        f" {'concurrent ' if sim_config.concurrent_updates else 'sequential '} updates")
            plot_agent_opinions(agents, ax=axs)
            plt.show()

    


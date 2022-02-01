import matplotlib.pyplot as plt
import random
import math
import pandas as pd


class Agent:
    def __init__(self, opinion, x, y, id=-1):
        self.id = id
        self.__opinion = opinion
        self.__opinions = [opinion]
        self.__x = x
        self.__y = y
        self.last_interaction_at = 0 # this argument is especially needed for the last task when we want multiple agents to be updated in one timestep
    
    def get_opinion(self):
        return self.__opinion

    def get_temporal_opinions(self):
        return self.__opinions

    def update_opinion(self, new_opinion, timestep):
        # we want data for each timestep --> no interaction in between so we simply ffill our last known opinion until we meet the current timestep
        # this has performance reasons
        while len(self.__opinions) < timestep:
            self.__opinions.append(self.__opinions[-1])
        self.__opinions.append(new_opinion)  # append in list so every agent has temporial data of its opinion
        self.last_interaction_at = timestep
        self.__opinion = new_opinion

    def get_coords(self):
        return self.__x, self.__y


class Grid:
    def __init__(self, n):
        if n != math.isqrt(n)**2:
            raise ValueError("n must be a perfect square to get a quadratic grid")
        self.max_index = math.isqrt(n) - 1
        self.grid = []

    def get_random_neighbour(self, x, y, neighborhood):
        available_neighborhoods = ['Moore', 'Moore2', 'Von Neumann']
        if neighborhood not in available_neighborhoods:
            raise ValueError(f"Invalid neighborhood: ({neighborhood}) passed, expected one of: {available_neighborhoods}")

        if neighborhood in ['Moore', 'Moore2']:
            if neighborhood == 'Moore':
                possible_x_steps = [-1, 0, 1]
                possible_y_steps = [-1, 0, 1]
            else:  # degree two
                possible_x_steps = [-2, -1, 0, 1, 2]
                possible_y_steps = [-2, -1, 0, 1, 2]

            x_shift = random.choice(possible_x_steps)
            y_shift = random.choice(possible_y_steps)
            while x_shift == 0 and y_shift == 0:  # assert that we do not return current agent but a neighbor
                x_shift = random.choice(possible_x_steps)
                y_shift = random.choice(possible_y_steps)

            if x + x_shift > self.max_index or x +  x_shift < 0:  # check boundaries and pick other direction if on edge
                x_shift = -1 * x_shift
            if y + y_shift > self.max_index or y +  y_shift < 0:
                y_shift = -1 * y_shift

            return self.grid[x + x_shift][y + y_shift]
        
        elif neighborhood == 'Von Neumann':
            # TODO
            pass
            

    def swap_positions(self, x1, y1, x2, y2):
        self.grid[x1][y1], self.grid[x2][y2] = self.grid[x2][y2], self.grid[x1][y1]

    def get_raw_opinions(self):
        res = [[i.get_opinion() for i in row] for row in self.grid]
        return res


def adjust_opinions(opinion1, opinion2, mu, tau):
    if abs(opinion1 - opinion2) > tau:
        return opinion1, opinion2
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2


def simulate_step(grid, agents, mu, tau, neighborhood, timestep):
    available_neighborhoods = ['Random', 'Moore', 'Moore2', 'Von Neumann']
    if neighborhood not in available_neighborhoods:
        raise ValueError(f"Invalid neighborhood: ({neighborhood}) passed, expected one of: {available_neighborhoods}")

    if neighborhood == 'Random':
        agent1, agent2 = random.sample(agents, 2)
    else:
        # choose first at random
        agent1 = random.choice(agents)
        x, y = agent1.get_coords()
        agent2 = grid.get_random_neighbour(x, y, neighborhood)
    op1 = agent1.get_opinion()
    op2 = agent2.get_opinion()
    op1, op2 = adjust_opinions(op1, op2, mu, tau)
    agent1.update_opinion(op1, timestep)
    agent2.update_opinion(op2, timestep)


def ffill_all_remaining_agents(agents, timesteps):
    for agent in agents:
        if len(agent.get_temporal_opinions()) < timesteps + 1: # plus one bc of initial opinion
            agent.update_opinion(agent.get_opinion(), timesteps)  # this ffills all the agents temporial data to be complete
            

def plot_grid(grid, title):
    plt.figure()
    cmap = plt.get_cmap("bwr")
    img = plt.imshow(grid, interpolation='nearest', cmap=cmap, origin='lower')
    plt.clim(-1, 1)
    plt.colorbar(img)
    plt.title(title)


def plot_temporal_opinions(grid, title):
    res = pd.DataFrame()
    for row in grid:
        for agent in row:
            y = agent.get_temporal_opinions()
            #plt.scatter(x, y, alpha=0.01, facecolors='none', edgecolors='black', linewidths=0.25)
            res[f"agent{agent.id}"] = y
    res = res.stack().reset_index()
    # plot only a sample for performance
    res = res.sample(200_000)
    res.columns = ['time', 'agent', 'opinions']
    res.plot(x='time', y='opinions', kind="scatter", facecolors='none', edgecolors='black', alpha=0.02, linewidths=0.5)
    plt.ylim(-1, 1)
    plt.title(title)



if __name__ == '__main__':
    # adapt parameters here
    n = 1089
    max_t = 20000
    mu = 0.1
    tau = 1.5
    neighborhood = "Moore"
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
        grid.grid.append(row)
    t = 0
    while t < max_t:
        t += 1
        simulate_step(grid, agents, mu, tau, neighborhood, t)
    ffill_all_remaining_agents(agents, max_t)
        
    opinion_grid = grid.get_raw_opinions()
    plot_grid(opinion_grid, f"Result with {mu=} and {tau=} using {neighborhood=}")
    plot_temporal_opinions(grid.grid, "Temporal Opinions of agents")
    # TODO add more plots if needed
    plt.show()


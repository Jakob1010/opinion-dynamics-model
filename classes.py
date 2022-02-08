import math
import random

from utils import get_von_neumann_neighborhood, get_moore_neighborhood

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
        self.__opinions.append(new_opinion)  # append in list so every agent has temporal data of its opinion
        if len(self.__opinions) == 20002:
            print("Achtung", timestep, new_opinion)
        self.last_interaction_at = timestep
        self.__opinion = new_opinion

    def get_coords(self):
        return self.__x, self.__y

    def set_coords(self, x, y):
        self.__x = x
        self.__y = y


class Grid:
    # for 'Moore' and 'Von Neumann', ' 2' suffix means second degree instead of first degree!
    available_neighborhoods = ['Random', 'Moore', 'Moore 2', 'Von Neumann', 'Von Neumann 2']

    def __init__(self, n):
        if n != math.isqrt(n)**2:
            raise ValueError("n must be a perfect square to get a quadratic grid")
        self.max_index = math.isqrt(n) - 1
        self.data = []

    def get_random_neighbour(self, x, y, neighborhood):
        if neighborhood not in Grid.available_neighborhoods:
            raise ValueError(f"Invalid neighborhood: ({neighborhood}) passed, expected one of: {Grid.available_neighborhoods}")

        if neighborhood == 'Random':
            nx = random.choice([i for i in range(self.max_index+1)])
            ny = random.choice([i for i in range(self.max_index+1)])
            while nx == x and ny == y:
                nx = random.choice([i for i in range(self.max_index+1)])
                ny = random.choice([i for i in range(self.max_index+1)])
            neighbors = [(nx, ny)]
        elif neighborhood in ['Moore', 'Moore 2']:
            if neighborhood == 'Moore':
                neighbors = get_moore_neighborhood(1, x, y)
            else:  # degree of two
                neighbors = get_moore_neighborhood(2, x, y)

        elif neighborhood in ['Von Neumann','Von Neumann 2']:
            if neighborhood == 'Von Neumann':
                neighbors = get_von_neumann_neighborhood(1, x, y)
            else: # degree of two
                neighbors = get_von_neumann_neighborhood(2, x, y)

        valid_neighbors = []
        for n in neighbors:
            if 0 <= n[0] <= self.max_index and 0 <= n[1] <= self.max_index:
                valid_neighbors.append(self.data[n[0]][n[1]])
        return random.choice(valid_neighbors)
            

    def swap_positions(self, x1, y1, x2, y2):
        self.data[x1][y1], self.data[x2][y2] = self.data[x2][y2], self.data[x1][y1]
        # also update data in the actual agents
        self.data[x1][y1].set_coords(x1, y1)
        self.data[x2][y2].set_coords(x2, y2)


    def get_raw_opinions(self):
        res = [[i.get_opinion() for i in row] for row in self.data]
        return res
import math
import random
import numpy as np
import pandas as pd
from itertools import groupby

from utils import get_von_neumann_neighborhood, get_moore_neighborhood

class Agent:
    def __init__(self, initial_opinion, x, y, timesteps, id=-1):
        self.id = id
        self.__opinions = np.full(timesteps + 1, np.nan)# +1 because of initial opinion
        self.__opinions[0] = initial_opinion
        self.__x = x
        self.__y = y
        self.last_interaction_at = 0
        self.connections = []
    
    
    def get_opinion(self):
        '''
        returns the current opinion of the agent, encoded as number
        '''
        return self.__opinions[self.last_interaction_at]

    
    def get_temporal_opinions(self):
        '''
        returns pandas Series of agent's opinion across all timesteps
        Any missing values (in case of no communcation happening at a given timestep) are forward-filled
        '''
        return pd.Series(self.__opinions, name=self.id).fillna(method="ffill")


    def get_number_of_conversations(self):
        '''
        returns how often the agent communicated with some other agent (no matter if opinion changed or not)
        Assumption: update_opinion() called each time a conversation happened
        '''
        return np.count_nonzero(~np.isnan(self.__opinions))
        
    def get_number_of_opinion_adjustments(self):
        '''
        returns how often the agent communicated with some other agent AND also changed its opinion
        Assumption: update_opinion() called each time a conversation happened
        '''
        unique_opinions = len([k for k,g in groupby(self.__opinions) if k!=g and not np.isnan(k)])
        return unique_opinions - 1

    def get_opinion_adjustment_rate(self):
        """
        returns "success rate" of conversations, i.e. how often agent adjusted its opinion after a conversation
        Assumption: update_opinion() called each time a conversation happened
        """
        return self.get_number_of_opinion_adjustments()/self.get_number_of_conversations()

    def update_opinion(self, new_opinion, timestep):
        self.__opinions[timestep] = new_opinion
        self.last_interaction_at = timestep

    def get_coords(self):
        return self.__x, self.__y

    def set_coords(self, x, y):
        self.__x = x
        self.__y = y

    def add_connection(self, agent):
        self.connections.append(agent)

    

class Grid:
    # for 'Moore' and 'Von Neumann', ' 2' suffix means second degree instead of first degree!
    available_neighborhoods = ['Random', 'Moore', 'Moore 2', 'Von Neumann', 'Von Neumann 2']

    def __init__(self, n):
        if n != math.isqrt(n)**2:
            raise ValueError("n must be a perfect square to get a quadratic grid")
        self.max_index = math.isqrt(n) - 1
        self.data = []

    def get_random_neighbour(self, x, y, neighborhood) -> Agent:
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
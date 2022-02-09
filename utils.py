import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_moore_neighborhood(degree, x, y):
    x_steps = [i for i in range(-degree, degree+1)]
    y_steps = [i for i in range(-degree, degree+1)]
    res = []
    for x_shift in x_steps:
        for y_shift in y_steps:
            if x_shift == 0 and y_shift == 0:  # assure that we do not return passed point
                continue
            res.append((x+x_shift, y+y_shift))
    return res

def get_von_neumann_neighborhood(degree, x, y):
    res = []
    if degree == 1:
        res = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return res
    else:
        res = res + \
                get_von_neumann_neighborhood(degree-1, x, y) + \
                get_von_neumann_neighborhood(degree-1, x+1, y) + \
                get_von_neumann_neighborhood(degree-1, x-1, y) + \
                get_von_neumann_neighborhood(degree-1, x, y+1) + \
                get_von_neumann_neighborhood(degree-1, x, y-1)
        return [r for r in (set(tuple(i) for i in res))]

def adjust_opinions(opinion1, opinion2, mu, tau):
    if abs(opinion1 - opinion2) > tau:
        return opinion1, opinion2
    adj1 = opinion1 + mu * (opinion2-opinion1)
    adj2 = opinion2 + mu * (opinion1-opinion2)
    return adj1, adj2

def plot_grid_data(grid_data, ax=None, title=None):
    cmap = plt.get_cmap("bwr")
    plt.axis('off')
    img = plt.imshow(grid_data, interpolation='nearest', cmap=cmap, origin='lower')
    plt.clim(-1, 1)
    plt.colorbar(img)
    if ax:
        ax.plot()
        if title:
            ax.set_title(title)
    else:
        if title:
            plt.title(title)

def plot_agent_opinions(agents, ax):
    agents_df = pd.concat([a.get_temporal_opinions() for a in agents], axis=1)
    timesteps = len(agents_df.index)

    # we don't need data for every single timestep, would require way too much memory
    if len(agents_df.index) > 101:
        timesteps_to_keep = np.rint(np.linspace(start=0, stop=timesteps - 1, num=101))
        agents_df = agents_df.iloc[timesteps_to_keep, :]

    agents_df.index.name = 'Timestep'
    #agents_df.to_csv('agent_opinions.csv')

    agents_stacked = agents_df.stack().reset_index()
    agents_stacked.columns = ["Timestep", "Agent", "Opinion"]

    #black background for plot
    ax.set_facecolor('black')

    cmap = plt.get_cmap("bwr")
    ax.scatter(x=agents_stacked['Timestep'], y=agents_stacked['Opinion'], vmin=-1, vmax=1,
                cmap=cmap, c=agents_stacked['Opinion'], s=32.0, alpha=0.1, edgecolors="none")
    ax.set_title(f'Agent opinions across {agents_stacked.Timestep.max()} timesteps')
    ax.set_xlabel('time')
    ax.set_ylabel('opinion')

if __name__ == '__main__':
    res = get_von_neumann_neighborhood(3, 5, 5)
    for r in res:
        print(r, r[0] > 5)
import matplotlib.pyplot as plt
import pandas as pd


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

def plot_grid(grid, title):
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
            res[f"agent{agent.id}"] = y
    res = res.stack().reset_index()
    # plot only a sample for performance
    res = res.sample(min(500_000, len(res)))
    res.columns = ['time', 'agent', 'opinions']
    res.plot(x='time', y='opinions', kind="scatter", facecolors='none', edgecolors='black', alpha=0.02, linewidths=0.5)
    plt.ylim(-1, 1)
    plt.title(title)

if __name__ == '__main__':
    res = get_von_neumann_neighborhood(3, 5, 5)
    for r in res:
        print(r, r[0] > 5)
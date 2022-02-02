from classes import Agent
list = [Agent(0, 0, 0, 0), Agent(1, 1, 1, 1), Agent(2, 2, 2, 2)]

a = list[2]
for i in list:
    if i == list[1]:
        a.update_opinion(-1, 1)
    print(i.get_temporal_opinions())
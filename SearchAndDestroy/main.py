import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import OrderedDict
import sys
import argparse
import random
import csv

class SearchAndDestroy():
    def __init__(self, dimensions, visual, rule, target_type=None):
        self.dim = dimensions
        self.target = ()
        self.original_map = self.generate_map()
        self.num_trials = 0
        # Make the target
        self.target_type = target_type
        self.target = self.create_target()
        self.visual = visual
        self.rule = rule

        if self.visual:
            grid = gridspec.GridSpec(ncols=2, nrows=2)

            # Make a grid of 4 equal parts
            self.fig = plt.figure(figsize=(15,15))
            self.f_ax1 = self.fig.add_subplot(grid[0, 0])
            self.f_ax2 = self.fig.add_subplot(grid[0, 1])
            self.f_ax3 = self.fig.add_subplot(grid[1, 0])
            self.f_ax4 = self.fig.add_subplot(grid[1, 1])

    # MAPPING
    # 0 ---> "Flat"
    # 1 ---> "Hilly"
    # 2 ---> "Forest"
    # 3 ---> "Caves"
    def generate_map(self):
        mat = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                p = np.random.rand()
                if p <= 0.2:
                    mat[i][j] = 0
                elif p > 0.2 and p <= 0.5:
                    mat[i][j] = 1
                elif p > 0.5 and p <= 0.8:
                    mat[i][j] = 2
                else: 
                    mat[i][j] = 3
        
        return mat

    def create_target(self):
        if self.target_type is None:
            x = np.random.randint(self.dim)
            y = np.random.randint(self.dim)
            return (x, y)

        elif self.target_type == "flat":
            indices = np.where(self.original_map == 0)
           
        elif self.target_type == "hill":
            indices = np.where(self.original_map == 1)

        elif self.target_type == "forest":
            indices = np.where(self.original_map == 2)

        else:
            indices = np.where(self.original_map == 3)

        coordinates = list(zip(indices[0], indices[1]))
        if len(coordinates) > 1:
            choose = random.randint(1, len(coordinates)) - 1
            return coordinates[choose]
        else:
            return coordinates[0]


    def generate_layout(self, belief, confidence, heat_map, iterations):
        # Display original matrix in console
        # print("\nOriginal Map: \n", self.original_map)
        self.fig.suptitle("Number of iterations: {}".format(iterations))
        self.f_ax1.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
        self.f_ax1.set_title("Actual")
        self.f_ax2.matshow(belief, cmap=cm.get_cmap('Greys_r'))
        self.f_ax2.set_title("Belief Matrix")
        self.f_ax3.matshow(confidence, cmap=cm.get_cmap('Greys_r'))
        self.f_ax3.set_title("Confidence Matrix")
        self.f_ax4.matshow(heat_map, cmap=cm.get_cmap('Greys'))
        self.f_ax4.set_title("Agent")

        self.f_ax1.scatter(self.target[0], self.target[1], s=100, c='red', marker='x')

class Agent():
    def __init__(self, game):
        self.original_map = game.original_map
        self.dim = len(self.original_map)
        self.belief = np.full((self.dim, self.dim), 1/(self.dim**2))
        self.target_cell = game.create_target()
        self.confidence = np.full((self.dim, self.dim), 1/(self.dim**2))
        self.visual = game.visual
        self.heat_map = np.zeros((self.dim, self.dim))
    
        for i in range(self.dim):
            for j in range(self.dim):
                self.confidence[i][j] *= (1 - self.false_neg_rate(i, j)[0])
        # print("Initial Confidence Matrix: \n", self.confidence)

    def false_neg_rate(self, x, y):
        if self.original_map[x][y] == 0:
            fnr = (0.1, "Flat")
        elif self.original_map[x][y] == 1:
            fnr = (0.3, "Hill")
        elif self.original_map[x][y] == 2:
            fnr = (0.7, "Forest")
        else:
            fnr = (0.9, "Caves")

        return fnr

    def max_prob_cell(self, rule):
        if rule == "belief":
            mat = self.belief
        elif rule == "confidence":
            mat = self.confidence

        max_val = np.argmax(mat)
        first_index = int(max_val/self.dim)
        second_index = max_val%self.dim
        max_values = []
        for i in range(self.dim):
            for j in range(self.dim):
                if mat[i][j] == mat[first_index][second_index]:
                    max_values.append((i,j))
        random_from_max = random.randint(1, len(max_values)) - 1
        return max_values[random_from_max]

    def run_game(self):
        iterations = 1
        while True:
            current_cell = self.max_prob_cell(game.rule)
            self.heat_map[current_cell[0], current_cell[1]] += 1
            
            # print("Current cell: {}, Target cell: {}".format(current_cell, self.target_cell))
            
            if self.visual:
                    plt.ion()
                    plt.show()
                    plt.pause(1e-15)
                    game.generate_layout(self.belief, self.confidence, self.heat_map, iterations)

            if current_cell == self.target_cell:
                terrain_prob = self.false_neg_rate(current_cell[0], current_cell[1])[0]
                p = random.uniform(0, 1)
                # print("Terrain FNR: ", terrain_prob, " Probability: ", p)
                if p > terrain_prob:
                    return iterations
                    # print("Number of iterations: ", iterations)
                    # break
            else:
                # Update iterations
                iterations += 1
                
                # Calculate new belief of current cell
                self.belief[current_cell[0]][current_cell[1]] *= self.false_neg_rate(current_cell[0], current_cell[1])[0]
                # print("New Belief Matrix: \n", self.belief)
                
                # Sum of the belief matrix
                belief_sum = np.sum(self.belief)
                # Normalize the belief matrix
                self.belief = self.belief/belief_sum
                # print("Normalized Belief Matrix: \n", self.belief)

                # Calculate new confidence based on new belief
                for i in range(self.dim):
                    for j in range(self.dim):
                        self.confidence[i][j] = self.belief[i][j]*(1 - self.false_neg_rate(i, j)[0])
                
                # Sum of the confidence matrix
                conf_sum = np.sum(self.confidence)
                # Normalize the confidence matrix
                self.confidence = self.confidence/conf_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic models to search and destroy")
    parser.add_argument('-n', "--grid_dimension", default=10)
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-r', "--rule", default="belief")
    parser.add_argument('-q', "--question", default="basic")
    args = parser.parse_args(sys.argv[1:])

    if args.question == "basic":
        game = SearchAndDestroy(dimensions=int(args.grid_dimension), visual=args.visual, rule=args.rule, target_type=None)
        agent = Agent(game)
        agent_iters = agent.run_game()
        print("Number of iterations: ", agent_iters)

    if args.question == "q13":
        dict_q3 = {}
        save_file = "q3_analysis.csv"
        csv = open(save_file, "w")
        csv.write("Grid Size, Rule Type, Terrain Type, Iterations\n")
        
        for grid_size in range(5, 50):
            for rule in ["belief", "confidence"]:
                for terrain_target in ["flat", "hill", "forest", "cave"]:
                    agent_iters = 0
                    for iter in range(10):
                        print("Running for Grid Dimension {}, Rule {}, Terrain Type {}".format(grid_size, rule, terrain_target))
                        game = SearchAndDestroy(dimensions=grid_size, visual=args.visual, rule=rule, target_type=terrain_target)
                        agent = Agent(game)
                        agent_iters += agent.run_game()

                    agent_iters /= 10
                    if str(grid_size) not in dict_q3:
                        dict_q3[str(grid_size)] = [[rule, terrain_target, int(agent_iters)]]
                    
                    else:
                        dict_q3[str(grid_size)].append([rule, terrain_target, int(agent_iters)])
        
        for key, val in dict_q3.items():
            for v in val:
                row = key + "," + v[0] + "," + v[1] + "," + str(v[2]) + "\n"
                csv.write(row)
        
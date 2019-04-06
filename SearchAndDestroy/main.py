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

class SearchAndDestroy():
    def __init__(self, dimensions, visual):
        self.dim = dimensions
        self.target = ()
        self.original_map = []
        self.num_trials = 0
        # Make the target
        self.target = self.create_target()
        self.visual = visual

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
        x = np.random.randint(self.dim)
        y = np.random.randint(self.dim)

        return (x, y)

    def generate_layout(self):
        grid = gridspec.GridSpec(ncols=2, nrows=2)

        # Make a grid of 4 equal parts
        fig = plt.figure(figsize=(15,15))
        f_ax1 = fig.add_subplot(grid[0, 0])
        f_ax2 = fig.add_subplot(grid[0, 1])
        f_ax3 = fig.add_subplot(grid[1, 0])
        f_ax4 = fig.add_subplot(grid[1, 1])

        self.original_map = self.generate_map()
        
        # Display original matrix in console
        print("\nOriginal Map: \n", self.original_map)

        if self.visual:
            # Display matrix
            f_ax1.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
            f_ax1.set_title("Actual")
            f_ax2.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
            f_ax2.set_title("Agent 1")
            f_ax3.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
            f_ax3.set_title("Agent 2")
            f_ax4.matshow(self.original_map, cmap=cm.get_cmap('Greens', 4))
            f_ax4.set_title("Agent 3")

            f_ax1.scatter(self.target[0], self.target[1], s=100, c='red', marker='x')

            plt.show()


class Agent():
    def __init__(self, game):
        self.original_map = game.original_map
        self.dim = len(self.original_map)
        self.belief = np.full((self.dim, self.dim), 1/(self.dim**2))
        self.target_cell = game.create_target()
        self.confidence = np.full((self.dim, self.dim), 1/(self.dim**2))

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

    def max_prob_cell(self):
        max_val = np.argmax(self.belief)
        first_index = int(max_val/self.dim)
        second_index = max_val%self.dim
        max_values = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.belief[i][j] == self.belief[first_index][second_index]:
                    max_values.append((i,j))
        random_from_max = random.randint(1, len(max_values)) - 1
        # print(max_values, random_from_max)
        return max_values[random_from_max]

    def run_game(self):
        iterations = 1
        while True:
            current_cell = self.max_prob_cell()
            print(current_cell, self.target_cell)
            if current_cell == self.target_cell:
                print("Number of iterations: ", iterations)
                break
            else:
                # Update iterations
                iterations += 1
                # Calculate new belief of current cell
                self.belief[current_cell[0]][current_cell[1]] *= self.false_neg_rate(current_cell[0], current_cell[1])[0]
                # print("New Belief Matrix: \n", self.belief)
                print("Terrain: ", self.false_neg_rate(current_cell[0], current_cell[1])[1])
                # Sum of the belief matrix
                belief_sum = np.sum(self.belief)
                # Normalize the belief matrix
                self.belief = self.belief/belief_sum
                # print("Normalized Belief Matrix: \n", self.belief)

                # Update confidence matrix
                


                

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic models to search and destroy")
    parser.add_argument("-n", "--grid_dimension", default=10)
    parser.add_argument('-v', "--visual", default = False)
    args = parser.parse_args(sys.argv[1:])

    game = SearchAndDestroy(dimensions=int(args.grid_dimension), visual=args.visual)
    game.generate_layout()

    agent = Agent(game)
    print("\nInitial Belief Matrix: \n", agent.belief)
    agent.run_game()
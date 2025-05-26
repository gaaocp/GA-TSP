#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GA-TSP
# Genetic Algorithm (GA) for the Traveling Salesman Problem (TSP)
# Author: Guglielmo Cimolai
# Date: 26/05/2025

import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import time

# 1) Main configuration
# City map grid
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100
# SGA default parameters (play with them to improve convergence!)
DEFAULT_SGA_POP_SIZE = 100 # Population size
DEFAULT_SGA_GENERATIONS = 1000 # Number of generations
DEFAULT_SGA_CROSSOVER_RATE = 0.85 # Crossover rate
DEFAULT_SGA_MUTATION_RATE = 0.15 # Mutation rate
DEFAULT_SGA_ELITISM_SIZE = 5 # Elitism size
DEFAULT_SGA_TOURNAMENT_K = 3 # Tournament size
# Live plotting
LIVE_PLOT_UPDATE_FREQ = 1 # Update live plot every N generations, set 0 to disable.

# 2) Helpers for city generation and distance calculation
def generate_cities(num_cities, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Generate random cities (n = num_cities) in given space
    cities = []
    for _ in range(num_cities):
        x = random.randint(0, width)
        y = random.randint(0, height)
        cities.append((x, y))
    return np.array(cities)

def euclidean_distance(city1, city2):
    # Compute Euclidean distance between any two cities
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

def calculate_distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            dist = euclidean_distance(cities[i], cities[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix

# 3) GA core components
def calculate_tour_cost(tour, distance_matrix):
    # Calculate total cost (= total path distance) for a full tour
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance_matrix[tour[i], tour[(i+1) % num_cities]]
    return cost

class Individual:
    def __init__(self, tour):
        self.tour = list(tour)
        self.cost = float('inf')

    def calculate_cost(self, distance_matrix):
        self.cost = calculate_tour_cost(self.tour, distance_matrix)
        return self.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        tour_str = str(self.tour) if len(self.tour) < 15 else str(self.tour[:7] + ["..."] + self.tour[-7:])
        return f"Tour: {tour_str} Cost: {self.cost:.2f}"

# 4) Live plotter (plots best route + convergence history)
class TSPPlotterSGA:
    def __init__(self, cities):
        self.cities = cities
        plt.ion()
        self.fig, self.ax_array = plt.subplots(1, 2, figsize=(18, 7))
        self.route_ax = self.ax_array[0]
        self.convergence_ax = self.ax_array[1]

        self.fig.subplots_adjust(wspace=0.25)

        # Setup SGA route plot
        self.route_ax.set_title("SGA Best Route Evolution")
        self.route_ax.set_xlabel("X-coordinate");
        self.route_ax.set_ylabel("Y-coordinate")

        # Show cities + legend
        self.route_ax.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', label='Cities', zorder=5)
        for i, city_coord in enumerate(cities):
            self.route_ax.text(city_coord[0]+0.5, city_coord[1]+0.5, str(i), fontsize=9)
        self.route_ax.legend(loc='upper right')

        # Setup convergence plot
        self.convergence_ax.set_title("SGA Fitness Convergence")
        self.convergence_ax.set_xlabel("Generation");
        self.convergence_ax.set_ylabel("Best Cost (Distance)")
        self.sga_convergence_line = None
        self.sga_route_line = None

    def update_live_route_plot(self, best_tour_indices, generation, best_cost):
        is_update_time = (LIVE_PLOT_UPDATE_FREQ > 0 and
                          (generation % LIVE_PLOT_UPDATE_FREQ == 0 or generation == -1 or generation == 0))
        if not is_update_time and generation > 0:
            return

        if self.sga_route_line:
            try:
                self.sga_route_line.pop(0).remove()
            except (AttributeError, IndexError, ValueError):
                self.sga_route_line = None

        # Connect cities with tour found
        tour_coords = np.array([self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]])
        line_color = 'blue'

        line = self.route_ax.plot(tour_coords[:, 0], tour_coords[:, 1], color=line_color, linestyle='-', marker='.',
                                  label="Current Best")
        self.sga_route_line = line

        gen_display = "Final" if generation == -1 else str(generation)
        self.route_ax.set_title(f"SGA Route - Gen: {gen_display}, Cost: {best_cost:.2f}")

        handles, labels = self.route_ax.get_legend_handles_labels()
        by_label = {"Cities": handles[labels.index("Cities")]}
        if self.sga_route_line:
            by_label["Current Best"] = self.sga_route_line[0]
        self.route_ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.pause(0.01)

    def update_convergence_plot(self, history, exec_time=None):
        generations_axis = list(range(len(history)))
        label = "SGA Best Cost"
        if exec_time is not None:
            label += f" (Time: {exec_time:.2f}s)"

        if self.sga_convergence_line:
            self.sga_convergence_line.set_data(generations_axis, history)
            self.sga_convergence_line.set_label(label)
        else:
            self.sga_convergence_line, = self.convergence_ax.plot(generations_axis, history, label=label, color="blue")

        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def display_execution_time(self, sga_time, sga_history):
        self.update_convergence_plot(sga_history, exec_time=sga_time)

    def show_final_route(self, sga_best_ind):
        self.route_ax.cla()
        self.route_ax.set_title(f"SGA Final Route - Cost: {sga_best_ind.cost:.2f}")
        self.route_ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', marker='o', label='Cities', zorder=5)
        for i, city_coord in enumerate(self.cities):
            self.route_ax.text(city_coord[0]+0.5, city_coord[1]+0.5, str(i), fontsize=9)
        sga_tour_coords = np.array([self.cities[i] for i in sga_best_ind.tour + [sga_best_ind.tour[0]]])
        self.route_ax.plot(sga_tour_coords[:, 0], sga_tour_coords[:, 1], 'b-', label=f"SGA Final Path")
        self.route_ax.legend(loc='upper right')
        plt.pause(0.1)

    def keep_plot_open(self):
        self.fig.suptitle("TSP Solver: Standard Genetic Algorithm (SGA)", fontsize=16, y=0.99)
        plt.ioff()
        plt.show()

# 5) Standard GA implementation (Selection-Crossover-Mutation loop)
def sga_initialize_population(num_cities, population_size):
    # Initialize population for the current generation
    population = []
    base_tour = list(range(num_cities))
    for _ in range(population_size):
        tour = random.sample(base_tour, num_cities)
        population.append(Individual(tour))
    return population

def sga_selection_tournament(population, k=DEFAULT_SGA_TOURNAMENT_K):
    # Define simple selection operator (k is tournament size)
    selected_parents = []
    for _ in range(len(population)):
        aspirants = random.sample(population, k)
        selected_parents.append(min(aspirants, key=lambda ind: ind.cost))
    return selected_parents

def sga_crossover_ordered(parent1_ind, parent2_ind):
    # Define simple crossover operator
    parent1_tour = parent1_ind.tour
    parent2_tour = parent2_ind.tour
    size = len(parent1_tour)
    child_tour = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child_tour[start:end+1] = parent1_tour[start:end+1]
    p2_idx = 0
    for i in range(size):
        if child_tour[i] == -1:
            while parent2_tour[p2_idx] in child_tour[start:end+1]:
                p2_idx += 1
            child_tour[i] = parent2_tour[p2_idx]
            p2_idx += 1
    return Individual(child_tour)

def sga_mutate_swap(individual, mutation_prob_per_individual):
    # Define simple mutation operator
    if random.random() < mutation_prob_per_individual:
        tour = individual.tour
        idx1, idx2 = random.sample(range(len(tour)), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

def solve_tsp_sga(cities, distance_matrix, plotter,
                  population_size=DEFAULT_SGA_POP_SIZE,
                  generations=DEFAULT_SGA_GENERATIONS,
                  crossover_rate=DEFAULT_SGA_CROSSOVER_RATE,
                  mutation_rate=DEFAULT_SGA_MUTATION_RATE,
                  elitism_size=DEFAULT_SGA_ELITISM_SIZE,
                  tournament_k=DEFAULT_SGA_TOURNAMENT_K):
    num_cities = len(cities)
    population = sga_initialize_population(num_cities, population_size)
    for ind in population:
        ind.calculate_cost(distance_matrix)

    population.sort()
    best_overall_individual = copy.deepcopy(population[0])
    cost_history = [best_overall_individual.cost]

    algo_name = "SGA" # For potential future extension
    print(f"\n--- Running {algo_name} for {num_cities} cities ---")
    print(f"Initial best cost: {best_overall_individual.cost:.2f}")
    plotter.update_live_route_plot(best_overall_individual.tour, 0, best_overall_individual.cost)

    for gen in range(1, generations+1):
        new_population = []
        if elitism_size > 0:
            new_population.extend(copy.deepcopy(population[:elitism_size]))

        mating_pool = sga_selection_tournament(population, tournament_k)
        offspring_idx = 0
        while len(new_population) < population_size:
            parent1 = mating_pool[offspring_idx % len(mating_pool)]
            offspring_idx += 1
            parent2 = mating_pool[offspring_idx % len(mating_pool)]
            offspring_idx += 1
            if random.random() < crossover_rate:
                child = sga_crossover_ordered(parent1, parent2)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            sga_mutate_swap(child, mutation_rate)
            child.calculate_cost(distance_matrix)
            new_population.append(child)

        population = new_population
        population.sort()
        if population[0].cost < best_overall_individual.cost:
            best_overall_individual = copy.deepcopy(population[0])
        cost_history.append(best_overall_individual.cost)

        # Console output for progress
        if gen % 10 == 0 or gen == generations: # Print every 10 generations and the last one
            print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {best_overall_individual.cost:.2f}")

        plotter.update_live_route_plot(best_overall_individual.tour, gen, best_overall_individual.cost)
        plotter.update_convergence_plot(cost_history) # exec_time will be added at the end

    print(f"{algo_name} Final Best Tour: {best_overall_individual.tour} with Cost: {best_overall_individual.cost:.2f}")
    plotter.update_live_route_plot(best_overall_individual.tour, -1, best_overall_individual.cost) # -1 for final plot
    return best_overall_individual, cost_history

# 6) Main execution
if __name__ == "__main__":
    NUM_CITIES = 50 # Try different numbers of cities
    CITY_SEED = 1 # Seed for city positions reproducibility

    SGA_PARAMS = {
        "population_size": DEFAULT_SGA_POP_SIZE, "generations": DEFAULT_SGA_GENERATIONS,
        "crossover_rate": DEFAULT_SGA_CROSSOVER_RATE, "mutation_rate": DEFAULT_SGA_MUTATION_RATE,
        "elitism_size": DEFAULT_SGA_ELITISM_SIZE, "tournament_k": DEFAULT_SGA_TOURNAMENT_K
    }
    current_live_plot_freq = LIVE_PLOT_UPDATE_FREQ

    # Setup problem
    # Adaptive GA parameters selection based on problem dimensionality
    if NUM_CITIES <= 50:
        SGA_PARAMS.update({"generations": 750, "population_size": 100})
        current_live_plot_freq = 1
    elif NUM_CITIES <= 100:
        SGA_PARAMS.update({"generations": 1500, "population_size": 200, "elitism_size": 10})
        current_live_plot_freq = 5
    else:
        SGA_PARAMS.update({"generations": 5000, "population_size": 250, "elitism_size": 15})
        current_live_plot_freq = 10

    LIVE_PLOT_UPDATE_FREQ = current_live_plot_freq

    cities_coords = generate_cities(NUM_CITIES, seed=CITY_SEED)
    distance_mat = calculate_distance_matrix(cities_coords)

    print(f"Generated {NUM_CITIES} cities. Distance matrix calculated.")
    if NUM_CITIES > 100 and LIVE_PLOT_UPDATE_FREQ > 0 and LIVE_PLOT_UPDATE_FREQ < 10:
        print(
            f"INFO: Live plot update frequency is {LIVE_PLOT_UPDATE_FREQ} for {NUM_CITIES} cities. This might be slow.")

    tsp_plotter = TSPPlotterSGA(cities_coords) # Use simplified plotter

    # Solve with Standard GA
    print(f"\nSGA Parameters: {SGA_PARAMS}")
    start_time_sga = time.time()
    sga_best_individual, sga_cost_history = solve_tsp_sga(
        cities_coords, distance_mat, tsp_plotter, **SGA_PARAMS
    )
    end_time_sga = time.time()
    sga_exec_time = end_time_sga - start_time_sga
    print(f"SGA execution time: {sga_exec_time:.2f} seconds")

    print("\n" + "="*20 + " Final Results " + "="*20)
    print(f"Problem: {NUM_CITIES} cities (Seed: {CITY_SEED})")
    print(f"\nStandard GA (SGA):")
    print(f"  Best Cost: {sga_best_individual.cost:.2f}")
    print(f"  Execution Time: {sga_exec_time:.2f}s")

    tsp_plotter.display_execution_time(sga_exec_time, sga_cost_history) # Add time to convergence plot
    tsp_plotter.show_final_route(sga_best_individual) # Show final route
    tsp_plotter.convergence_ax.set_title("SGA Final Fitness Convergence") # Reset title after time add
    tsp_plotter.convergence_ax.legend(loc='upper right')

    print("\nCheck the plots for visual representation.")
    print("Close the plot window to end the script.")
    tsp_plotter.keep_plot_open()

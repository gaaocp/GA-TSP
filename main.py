#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GA-TSP
# Solving the Traveling Salesman Problem (TSP) by comparing a Simple Genetic Algorithm (SGA)
# with a Hybrid Genetic Algorithm - Ant Colony Optimization (HGA-ACO)

# Strategy:
# - Cities are generate randomly in a 2D grid, distance is Euclidean
# - An Individual in the GA is defined as a complete tour on the map connecting all cities once
# - SGA uses elitism, operators are defined from scratch
# - HGA-ACO uses pheromone traces (ACO strategy) to "guide" its GA part to better solution

# Author: Guglielmo Cimolai
# Date: 2/06/2025

import random
import matplotlib.pyplot as plt
import copy
import numpy as np
import time

# ------------------------------------
# 1) GA Parameters Configuration
# ------------------------------------
# City map grid
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100

# SGA default parameters
DEFAULT_SGA_POP_SIZE = 100 # Population size
DEFAULT_SGA_GENERATIONS = 1000 # Number of generations
DEFAULT_SGA_CROSSOVER_RATE = 0.85 # Crossover rate
DEFAULT_SGA_MUTATION_RATE = 0.15 # Mutation rate
DEFAULT_SGA_ELITISM_SIZE = 5 # Elitism size
DEFAULT_SGA_TOURNAMENT_K = 3 # Tournament size

# HGA-ACO default parameters
DEFAULT_HGA_POP_SIZE = 100
DEFAULT_HGA_GENERATIONS = 250
DEFAULT_HGA_GA_CROSSOVER_RATE = 0.7 # Crossover rate for the GA-generated portion of HGA population
DEFAULT_HGA_ACO_CONTRIBUTION_RATE = 0.5 # Proportion of non-elite new individuals from ACO construction
DEFAULT_HGA_MUTATION_RATE = 0.15 # Chance for new individuals (both GA and ACO derived) to be mutated
DEFAULT_HGA_ELITISM_SIZE = 5
DEFAULT_HGA_TOURNAMENT_K = 3
# Pheromone parameters for ACO
DEFAULT_HGA_ALPHA = 1.0 # Pheromone influence
DEFAULT_HGA_BETA = 3.0 # Heuristic (distance) influence
DEFAULT_HGA_EVAPORATION_RATE = 0.3 # Rho
DEFAULT_HGA_Q_PHEROMONE = 100.0 # Pheromone deposit constant
DEFAULT_HGA_INITIAL_PHEROMONE = 0.1
DEFAULT_HGA_BEST_N_DEPOSIT = 5 # How many best individuals from current pop deposit pheromone

# Live plotting
LIVE_PLOT_UPDATE_FREQ = 1  # Update live plot every N generations. Set to 0 to disable

# ------------------------------------
# 2) Helpers for city generation and distance calculation
# ------------------------------------
def generate_cities(num_cities, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, seed=None):
    # Generate random cities in given space
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    cities = []
    for _ in range(num_cities):
        x = random.randint(0, width)
        y = random.randint(0, height)
        cities.append((x, y))
    return np.array(cities)

def euclidean_distance(city1, city2):
    # Define Euclidean distance between any two cities
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

def calculate_distance_matrix(cities):
    # Compute Euclidean distance between any two cities
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i+1, num_cities): # Avoid self-distance
            dist = euclidean_distance(cities[i], cities[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def calculate_tour_cost(tour, distance_matrix):
    # Calculate total cost (= total path distance) for a full tour
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance_matrix[tour[i], tour[(i+1) % num_cities]] # Connect back to start
    return cost

# ------------------------------------
# 3) Individual definition (common to both SGA and "GA part" of HGA-ACO)
# ------------------------------------
class Individual:
    # An Individual is defined as a complete tour on the map connecting all cities once
    def __init__(self, tour):
        self.tour = list(tour) # Make a mutable copy
        self.cost = float('inf') # Initialize cost as INF

    def calculate_cost(self, distance_matrix):
        self.cost = calculate_tour_cost(self.tour, distance_matrix)
        return self.cost

    def __lt__(self, other): # For sorting by cost (lower is better)
        return self.cost < other.cost

    def __repr__(self):
        # Shorten tour representation if too long
        tour_str = str(self.tour) if len(self.tour) < 15 else str(self.tour[:7] + ["..."] + self.tour[-7:])
        return f"Tour: {tour_str} Cost: {self.cost:.2f}"

# ------------------------------------
# 4) Live plotter (plots best route + convergence history)
# ------------------------------------
class TSPPlotter:
    def __init__(self, cities):
        self.cities = cities
        plt.ion() # interactive mode: ON
        self.fig, self.ax = plt.subplots(1, 2, figsize=(18, 7))
        self.route_ax = self.ax[0]
        self.convergence_ax = self.ax[1]

        # Setup city map plots
        self.route_ax.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', label='Cities', zorder=5)
        for i, city_coord in enumerate(cities):
            self.route_ax.text(city_coord[0] + 0.5, city_coord[1] + 0.5, str(i), fontsize=9)
        self.route_ax.set_xlabel("X-coordinate")
        self.route_ax.set_ylabel("Y-coordinate")
        self.route_ax.legend(loc='upper right')

        # Setup convergence plot
        self.convergence_ax.set_xlabel("Generation")
        self.convergence_ax.set_ylabel("Best Cost (Distance)")
        self.convergence_ax.set_title("Fitness Convergence")
        self.convergence_lines = {} # Store convergence lines by algo name
        self.route_lines = {} # Store route lines by algo name

    def update_live_route_plot(self, best_tour_indices, algo_name, generation, best_cost):
        if not LIVE_PLOT_UPDATE_FREQ or generation % LIVE_PLOT_UPDATE_FREQ != 0:
            if generation != 1 and generation != -1: # -1 for final plot
                return

        # Remove old route line for this specific algorithm
        if algo_name in self.route_lines and self.route_lines[algo_name]:
            self.route_lines[algo_name].pop(0).remove()

        tour_coords = np.array([self.cities[i] for i in best_tour_indices + [best_tour_indices[0]]])
        line_color = 'blue' if "SGA" in algo_name else 'green'

        # Plot the best individual in each generation
        line = self.route_ax.plot(tour_coords[:, 0], tour_coords[:, 1], color=line_color, linestyle='-', marker='.',
                                  label=f"{algo_name} Best Tour")
        self.route_lines[algo_name] = line

        # Set title and legend
        self.route_ax.set_title(f"TSP Route - {algo_name} - Gen: {generation}, Cost: {best_cost:.2f}")
        handles, labels = self.route_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicate labels
        self.route_ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        plt.pause(0.01)

    def update_convergence_plot(self, history, algo_name, color_val):
        generations_axis = list(range(len(history)))

        # Plot cost of the best individual in each generation
        if algo_name in self.convergence_lines:
            self.convergence_lines[algo_name].set_data(generations_axis, history)
        else:
            line, = self.convergence_ax.plot(generations_axis, history, label=f"{algo_name} Best Cost", color=color_val)
            self.convergence_lines[algo_name] = line

        # Set title and legend
        self.convergence_ax.relim()
        self.convergence_ax.autoscale_view()
        self.convergence_ax.legend(loc='upper right')
        plt.pause(0.01)

    def show_final_routes(self, sga_best_ind, hga_best_ind):
        self.route_ax.cla() # Clear axis for final combined plot
        self.route_ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', marker='o', label='Cities', zorder=5)
        for i, city_coord in enumerate(self.cities):
            self.route_ax.text(city_coord[0] + 0.5, city_coord[1] + 0.5, str(i), fontsize=9)

        # SGA final path
        sga_tour_coords = np.array([self.cities[i] for i in sga_best_ind.tour + [sga_best_ind.tour[0]]])
        self.route_ax.plot(sga_tour_coords[:, 0], sga_tour_coords[:, 1], 'b--',
                           label=f"SGA Final: {sga_best_ind.cost:.2f}", linewidth=1.5)
        # HGA final path
        hga_tour_coords = np.array([self.cities[i] for i in hga_best_ind.tour + [hga_best_ind.tour[0]]])
        self.route_ax.plot(hga_tour_coords[:, 0], hga_tour_coords[:, 1], 'g-',
                           label=f"HGA-ACO Final: {hga_best_ind.cost:.2f}", linewidth=2)

        # Set title and legend
        self.route_ax.set_title("Final Best Tours Comparison")
        self.route_ax.legend(loc='upper right')
        plt.pause(0.01)

    def keep_plot_open(self):
        plt.ioff()
        plt.show()

# ------------------------------------
# 5) SGA implementation (Selection-Crossover-Mutation loop)
# ------------------------------------
def sga_initialize_population(num_cities, population_size):
    # Initialize population for the current generation
    # random.sample ensures all elements from base_tour are used once
    population = []
    base_tour = list(range(num_cities))
    for _ in range(population_size):
        tour = random.sample(base_tour, num_cities)
        population.append(Individual(tour))
    return population

def sga_selection_tournament(population, k=DEFAULT_SGA_TOURNAMENT_K):
    # Define simple selection operator (k is tournament size)
    selected_parents = []
    for _ in range(len(population)): # Create a mating pool of the same size as population
        aspirants = random.sample(population, k)
        selected_parents.append(min(aspirants, key=lambda ind: ind.cost))
    return selected_parents

def sga_crossover_ordered(parent1_ind, parent2_ind):
    # Define simple crossover operator
    parent1_tour = parent1_ind.tour
    parent2_tour = parent2_ind.tour
    size = len(parent1_tour)
    child_tour = [-1]*size
    start, end = sorted(random.sample(range(size), 2))
    # Copy segment from parent1
    child_tour[start:end+1] = parent1_tour[start:end+1]
    # Fill remaining from parent2, maintaining order and avoiding duplicates
    p2_idx = 0
    for i in range(size):
        if child_tour[i] == -1:
            while parent2_tour[p2_idx] in child_tour[start:end+1]: # Find next city in P2 not in P1's segment
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
        # No need to return, individual.tour is mutated in-place

def solve_tsp_sga(cities, distance_matrix, plotter,
                  population_size=DEFAULT_SGA_POP_SIZE,
                  generations=DEFAULT_SGA_GENERATIONS,
                  crossover_rate=DEFAULT_SGA_CROSSOVER_RATE,
                  mutation_rate=DEFAULT_SGA_MUTATION_RATE,
                  elitism_size=DEFAULT_SGA_ELITISM_SIZE,
                  tournament_k=DEFAULT_SGA_TOURNAMENT_K):
    # Implement and run SGA for solving the TSP
    num_cities = len(cities)
    population = sga_initialize_population(num_cities, population_size)
    for ind in population:
        ind.calculate_cost(distance_matrix)

    population.sort() # Sort by cost (ascending)
    best_overall_individual = copy.deepcopy(population[0])
    cost_history = [best_overall_individual.cost]

    algo_name = "SGA"
    print(f"\n--- Running {algo_name} for {num_cities} cities ---")
    print(f"Initial best cost: {best_overall_individual.cost:.2f}")
    if LIVE_PLOT_UPDATE_FREQ > 0:
        plotter.update_live_route_plot(best_overall_individual.tour, algo_name, 0, best_overall_individual.cost)

    # Generate new population for current generation
    for gen in range(1, generations+1):
        new_population = []
        # Apply elitism
        if elitism_size > 0:
            elites = copy.deepcopy(population[:elitism_size])
            new_population.extend(elites)

        # Create the rest of the new population
        num_offspring_needed = population_size - len(new_population)
        mating_pool = sga_selection_tournament(population, tournament_k)
        offspring_idx = 0
        while len(new_population) < population_size:
            parent1 = mating_pool[offspring_idx % len(mating_pool)]
            offspring_idx += 1
            parent2 = mating_pool[offspring_idx % len(mating_pool)] # Ensure a different parent if pool is diverse
            offspring_idx += 1
            if random.random() < crossover_rate:
                child = sga_crossover_ordered(parent1, parent2)
            else: # Clone one parent if no crossover
                child = copy.deepcopy(random.choice([parent1, parent2]))

            sga_mutate_swap(child, mutation_rate)
            child.calculate_cost(distance_matrix)
            new_population.append(child)

        population = new_population
        population.sort()

        if population[0].cost < best_overall_individual.cost:
            best_overall_individual = copy.deepcopy(population[0])

        cost_history.append(best_overall_individual.cost)

        if gen % 10 == 0 or gen == generations: # Print every 10 generations on console and the last one
            print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {best_overall_individual.cost:.2f}")

        if LIVE_PLOT_UPDATE_FREQ > 0 and (gen % LIVE_PLOT_UPDATE_FREQ == 0 or gen == generations):
            plotter.update_live_route_plot(best_overall_individual.tour, algo_name, gen, best_overall_individual.cost)
        plotter.update_convergence_plot(cost_history, algo_name, "blue")

    print(f"{algo_name} Final Best Tour: {best_overall_individual.tour} with Cost: {best_overall_individual.cost:.2f}")
    if LIVE_PLOT_UPDATE_FREQ > 0: # Show final SGA route before HGA starts
        plotter.update_live_route_plot(best_overall_individual.tour, algo_name, generations,
                                       best_overall_individual.cost)
    return best_overall_individual, cost_history

# ------------------------------------
# 6) Hybrid GA-ACO Implementation (Pheromones, etc.)
# ------------------------------------
def initialize_pheromones(num_cities, initial_value=DEFAULT_HGA_INITIAL_PHEROMONE):
    # Ensure initial_value is positive
    initial_value = max(initial_value, 1e-6)
    return np.full((num_cities, num_cities), initial_value)

def hga_construct_individual_aco(num_cities, distance_matrix, pheromone_matrix,
                                 alpha=DEFAULT_HGA_ALPHA, beta=DEFAULT_HGA_BETA):
    tour = []
    # Create a list of cities to visit
    remaining_cities = list(range(num_cities))
    # Start at a random city
    current_city = random.choice(remaining_cities)
    tour.append(current_city)
    remaining_cities.remove(current_city)

    while remaining_cities:
        probabilities = []
        prob_sum = 0.0
        for next_city_candidate in remaining_cities:
            pheromone_val = pheromone_matrix[current_city, next_city_candidate]
            # Add a small epsilon to distance to avoid division by zero
            dist_val = distance_matrix[current_city, next_city_candidate]
            heuristic_val = (1.0/(dist_val+1e-10))**beta # Adding epsilon for safety
            # Ensure pheromone and heuristic are positive before raising to power if alpha/beta can be < 0
            # For typical ACO, alpha, beta >= 0. Pheromone should be > 0.
            prob = (max(pheromone_val, 1e-6)**alpha)*heuristic_val
            probabilities.append(prob)
            prob_sum += prob

        if prob_sum == 0 or not remaining_cities: # Fallback if all probs are zero or no cities left
            if not remaining_cities: break
            chosen_next_city = random.choice(remaining_cities)
        else:
            probabilities_norm = np.array(probabilities)/prob_sum
            chosen_next_city = np.random.choice(remaining_cities, p=probabilities_norm)

        tour.append(chosen_next_city)
        remaining_cities.remove(chosen_next_city)
        current_city = chosen_next_city

    return Individual(tour)

def update_pheromones(pheromone_matrix, population,
                      evaporation_rate=DEFAULT_HGA_EVAPORATION_RATE,
                      Q=DEFAULT_HGA_Q_PHEROMONE,
                      best_n_to_deposit=DEFAULT_HGA_BEST_N_DEPOSIT,
                      min_pheromone=1e-6): # Minimum pheromone value
    # Evaporation: tau=(1-rho)*tau
    pheromone_matrix *= (1.0-evaporation_rate)

    # Pheromone deposit from the best N individuals in the current population
    # Population should be sorted by cost beforehand if best_n_to_deposit relies on order
    sorted_pop = sorted(population, key=lambda ind: ind.cost)
    num_depositing_ants = min(best_n_to_deposit, len(sorted_pop))
    for i in range(num_depositing_ants):
        individual = sorted_pop[i]
        if individual.cost == 0:
            continue # Avoid division by zero for invalid cost

        deposit_amount = Q/individual.cost # Shorter tours deposit more pheromone
        tour = individual.tour
        num_cities_in_tour = len(tour)
        for j in range(num_cities_in_tour):
            city1 = tour[j]
            city2 = tour[(j+1) % num_cities_in_tour] # Connect back to start
            pheromone_matrix[city1, city2] += deposit_amount
            pheromone_matrix[city2, city1] += deposit_amount # Symmetric TSP

    # Ensure pheromones don't fall below a minimum threshold (helps prevent stagnation)
    pheromone_matrix[pheromone_matrix < min_pheromone] = min_pheromone

def solve_tsp_hga_aco(cities, distance_matrix, plotter,
                      population_size=DEFAULT_HGA_POP_SIZE,
                      generations=DEFAULT_HGA_GENERATIONS,
                      ga_crossover_rate=DEFAULT_HGA_GA_CROSSOVER_RATE,
                      aco_contribution_rate=DEFAULT_HGA_ACO_CONTRIBUTION_RATE,
                      mutation_rate=DEFAULT_HGA_MUTATION_RATE,
                      elitism_size=DEFAULT_HGA_ELITISM_SIZE,
                      tournament_k=DEFAULT_HGA_TOURNAMENT_K,
                      alpha=DEFAULT_HGA_ALPHA, beta=DEFAULT_HGA_BETA,
                      evaporation_rate=DEFAULT_HGA_EVAPORATION_RATE,
                      Q_pheromone=DEFAULT_HGA_Q_PHEROMONE,
                      initial_pheromone_val=DEFAULT_HGA_INITIAL_PHEROMONE,
                      best_n_deposit=DEFAULT_HGA_BEST_N_DEPOSIT):
    # Implement and run HGA for solving the TSP
    num_cities = len(cities)
    population = sga_initialize_population(num_cities, population_size) # Random initial population
    pheromone_matrix = initialize_pheromones(num_cities, initial_pheromone_val)

    for ind in population:
        ind.calculate_cost(distance_matrix) # Calculate initial costs

    population.sort() # Sort by cost
    best_overall_individual = copy.deepcopy(population[0])
    cost_history = [best_overall_individual.cost]

    algo_name = "HGA-ACO"
    print(f"\n--- Running {algo_name} for {num_cities} cities ---")
    print(f"Initial best cost: {best_overall_individual.cost:.2f}")
    if LIVE_PLOT_UPDATE_FREQ > 0:
        plotter.update_live_route_plot(best_overall_individual.tour, algo_name, 0, best_overall_individual.cost)

    for gen in range(1, generations + 1):
        new_population = []
        # Elitism
        if elitism_size > 0:
            elites = copy.deepcopy(population[:elitism_size])
            new_population.extend(elites)

        # Determine number of offspring from GA operators vs ACO construction
        num_offspring_needed = population_size - len(new_population)
        num_from_aco = int(num_offspring_needed*aco_contribution_rate)
        num_from_ga_ops = num_offspring_needed - num_from_aco
        # ACO-constructed Individuals
        for _ in range(num_from_aco):
            child = hga_construct_individual_aco(num_cities, distance_matrix, pheromone_matrix, alpha, beta)
            sga_mutate_swap(child, mutation_rate) # Mutate ACO-generated individuals
            child.calculate_cost(distance_matrix)
            new_population.append(child)

        # GA-constructed Individuals (Crossover & Mutation)
        if num_from_ga_ops > 0:
            mating_pool = sga_selection_tournament(population, tournament_k)
            offspring_idx = 0
            while len(new_population) < population_size: # Fill remaining spots for GA portion
                if not mating_pool:
                    break # Should not happen

                parent1 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1
                parent2 = mating_pool[offspring_idx % len(mating_pool)]
                offspring_idx += 1
                if random.random() < ga_crossover_rate:
                    child = sga_crossover_ordered(parent1, parent2)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2])) # Clone if no GA crossover

                sga_mutate_swap(child, mutation_rate) # Mutate GA-generated individuals
                child.calculate_cost(distance_matrix)
                new_population.append(child)

        population = new_population
        population.sort() # Sort the new complete population by cost
        if population[0].cost < best_overall_individual.cost:
            best_overall_individual = copy.deepcopy(population[0])

        cost_history.append(best_overall_individual.cost)
        # Update Pheromones based on the current population's performance
        update_pheromones(pheromone_matrix, population, evaporation_rate, Q_pheromone, best_n_deposit)

        if gen % 10 == 0 or gen == generations:
            print(f"{algo_name} Gen {gen}/{generations} - Best Cost: {best_overall_individual.cost:.2f}")

        if LIVE_PLOT_UPDATE_FREQ > 0 and (gen % LIVE_PLOT_UPDATE_FREQ == 0 or gen == generations):
            plotter.update_live_route_plot(best_overall_individual.tour, algo_name, gen, best_overall_individual.cost)
        plotter.update_convergence_plot(cost_history, algo_name, "green")

    print(f"{algo_name} Final Best Tour: {best_overall_individual.tour} with Cost: {best_overall_individual.cost:.2f}")
    if LIVE_PLOT_UPDATE_FREQ > 0: # Show final HGA-ACO route
        plotter.update_live_route_plot(best_overall_individual.tour, algo_name, generations,
                                       best_overall_individual.cost)
    return best_overall_individual, cost_history

# ------------------------------------
# 7) Main Execution
# ------------------------------------
if __name__ == "__main__":
    NUM_CITIES = 50 # Try different numbers of cities
    CITY_SEED = 1 # Seed for city positions reproducibility

    # GA parameters tuning (can be adjusted based on NUM_CITIES)
    SGA_PARAMS = {
        "population_size": DEFAULT_SGA_POP_SIZE, "generations": DEFAULT_SGA_GENERATIONS,
        "crossover_rate": DEFAULT_SGA_CROSSOVER_RATE, "mutation_rate": DEFAULT_SGA_MUTATION_RATE,
        "elitism_size": DEFAULT_SGA_ELITISM_SIZE, "tournament_k": DEFAULT_SGA_TOURNAMENT_K
    }
    HGA_PARAMS = {
        "population_size": DEFAULT_HGA_POP_SIZE, "generations": DEFAULT_HGA_GENERATIONS,
        "ga_crossover_rate": DEFAULT_HGA_GA_CROSSOVER_RATE,
        "aco_contribution_rate": DEFAULT_HGA_ACO_CONTRIBUTION_RATE,
        "mutation_rate": DEFAULT_HGA_MUTATION_RATE, "elitism_size": DEFAULT_HGA_ELITISM_SIZE,
        "tournament_k": DEFAULT_HGA_TOURNAMENT_K, "alpha": DEFAULT_HGA_ALPHA, "beta": DEFAULT_HGA_BETA,
        "evaporation_rate": DEFAULT_HGA_EVAPORATION_RATE, "Q_pheromone": DEFAULT_HGA_Q_PHEROMONE,
        "initial_pheromone_val": DEFAULT_HGA_INITIAL_PHEROMONE, "best_n_deposit": DEFAULT_HGA_BEST_N_DEPOSIT
    }
    current_live_plot_freq = LIVE_PLOT_UPDATE_FREQ

    # Adaptive GA parameters selection based on problem dimensionality
    if NUM_CITIES <= 50:
        SGA_PARAMS.update({"generations": 750, "population_size": 100})
        HGA_PARAMS.update({"generations": 250, "population_size": 100})
        current_live_plot_freq = 1
    elif NUM_CITIES <= 100:
        SGA_PARAMS.update({"generations": 1500, "population_size": 200, "elitism_size": 10})
        HGA_PARAMS.update({"generations": 500, "population_size": 100, "best_n_deposit": 5})
        current_live_plot_freq = 5
    else:
        SGA_PARAMS.update({"generations": 5000, "population_size": 200, "elitism_size": 15})
        HGA_PARAMS.update({"generations": 750, "population_size": 200, "elitism_size": 10, "best_n_deposit": 10})
        current_live_plot_freq = 10

    # Override global LIVE_PLOT_UPDATE_FREQ with the tiered one
    LIVE_PLOT_UPDATE_FREQ = current_live_plot_freq

    # Setup problem
    cities_coords = generate_cities(NUM_CITIES, seed=CITY_SEED)
    distance_mat = calculate_distance_matrix(cities_coords)

    print("GA-TSP\n"
          "Solving the Traveling Salesman Problem (TSP) by comparing a Simple Genetic Algorithm (SGA)"
          "with a Hybrid Genetic Algorithm - Ant Colony Optimization (HGA-ACO)\n")
    print(f"Generated {NUM_CITIES} cities. Distance matrix calculated.")
    if NUM_CITIES > 100 and LIVE_PLOT_UPDATE_FREQ > 0 and LIVE_PLOT_UPDATE_FREQ < 10:
        print(
            f"INFO: Live plot update frequency is {LIVE_PLOT_UPDATE_FREQ} for {NUM_CITIES} cities. This might be slow.")

    # Initialize plotter
    tsp_plotter = TSPPlotter(cities_coords)

    # Solve with Standard GA
    print(f"\nSGA Parameters: {SGA_PARAMS}")
    start_time_sga = time.time()
    sga_best_individual, sga_cost_history = solve_tsp_sga(cities_coords, distance_mat, tsp_plotter, **SGA_PARAMS)
    end_time_sga = time.time()
    sga_exec_time = end_time_sga - start_time_sga
    print(f"SGA execution time: {sga_exec_time:.2f} seconds")

    # Solve with Hybrid GA-ACO
    print(f"\nHGA-ACO Parameters: {HGA_PARAMS}")
    start_time_hga = time.time()
    hga_best_individual, hga_cost_history = solve_tsp_hga_aco(cities_coords, distance_mat, tsp_plotter, **HGA_PARAMS)
    end_time_hga = time.time()
    hga_exec_time = end_time_hga - start_time_hga
    print(f"HGA-ACO execution time: {hga_exec_time:.2f} seconds")

    # Final comparison output
    print("\n" + "="*20 + " Final Comparison " + "="*20)
    print(f"Problem: {NUM_CITIES} cities (Seed: {CITY_SEED})")
    print(f"\nStandard GA (SGA):")
    print(f"  Best Cost: {sga_best_individual.cost:.2f}")
    print(f"  Execution Time: {sga_exec_time:.2f}s")
    # print(f"  Best Tour: {sga_best_individual.tour}")

    print(f"\nHybrid GA-ACO (HGA-ACO):")
    print(f"  Best Cost: {hga_best_individual.cost:.2f}")
    print(f"  Execution Time: {hga_exec_time:.2f}s")
    # print(f"  Best Tour: {hga_best_individual.tour}")

    # Calculate improvement (from SGA to HGA-ACO)
    improvement_abs = sga_best_individual.cost-hga_best_individual.cost
    improvement_rel = (improvement_abs/sga_best_individual.cost*100) if sga_best_individual.cost > 0 else 0
    if hga_best_individual.cost < sga_best_individual.cost:
        print(f"\nHGA-ACO found a better solution by {improvement_abs:.2f} ({improvement_rel:.2f}% improvement).")
    elif sga_best_individual.cost < hga_best_individual.cost:
        print(f"\nSGA found a better solution by {-improvement_abs:.2f}.")
    else:
        print("\nBoth algorithms found solutions with the same cost.")

    # Update final plots
    tsp_plotter.show_final_routes(sga_best_individual, hga_best_individual)
    tsp_plotter.convergence_ax.set_title("Final Fitness Convergence Comparison: SGA vs HGA-ACO")
    tsp_plotter.convergence_ax.legend(loc='upper right')

    print("\nCheck the plots for visual comparison of routes and convergence.")
    print("Close the plot window to end the script.")
    tsp_plotter.keep_plot_open() # This will block until the plot window is closed

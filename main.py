#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GA-TSP
# Genetic Algorithm (GA) for the Traveling Salesman Problem (TSP)
# Author: Guglielmo Cimolai
# Date: 23/05/2025

import random
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
    # Initialize distance matrix
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

# 4) Standard GA implementation (Selection-Crossover-Mutation loop)
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

def solve_tsp_sga(cities, distance_matrix,
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

    print(f"\n--- Running SGA for {num_cities} cities ---")
    print(f"Initial best cost: {best_overall_individual.cost:.2f}")

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

        # Console output for progress
        if gen % 10 == 0 or gen == generations:  # Print every 10 generations and the last one
            print(f"SGA Gen {gen}/{generations} - Best Cost: {best_overall_individual.cost:.2f}")

    print(f"SGA Final Best Tour: {best_overall_individual.tour} with Cost: {best_overall_individual.cost:.2f}")
    return best_overall_individual

# 5) Main execution
if __name__ == "__main__":
    NUM_CITIES = 50 # Try different numbers of cities
    CITY_SEED = 1 # Seed for city positions reproducibility

    # GA parameter tuning (can be adjusted based on NUM_CITIES)
    SGA_PARAMS = {
        "population_size": DEFAULT_SGA_POP_SIZE, "generations": DEFAULT_SGA_GENERATIONS,
        "crossover_rate": DEFAULT_SGA_CROSSOVER_RATE, "mutation_rate": DEFAULT_SGA_MUTATION_RATE,
        "elitism_size": DEFAULT_SGA_ELITISM_SIZE, "tournament_k": DEFAULT_SGA_TOURNAMENT_K
    }

    # Setup problem
    cities_coords = generate_cities(NUM_CITIES, seed=CITY_SEED)
    distance_mat = calculate_distance_matrix(cities_coords)

    print(f"Generated {NUM_CITIES} cities. Distance matrix calculated.")

    # Solve with Standard GA
    print(f"\nSGA Parameters: {SGA_PARAMS}")
    start_time_sga = time.time()
    sga_best_individual = solve_tsp_sga(
        cities_coords, distance_mat, **SGA_PARAMS
    )
    end_time_sga = time.time()
    sga_exec_time = end_time_sga - start_time_sga

    # Final results output
    print("\n" + "="*10 + " Final Results " + "="*10)
    print(f"Problem: {NUM_CITIES} cities (Seed: {CITY_SEED})")
    print(f"\nStandard GA (SGA):")
    print(f"  Best Cost: {sga_best_individual.cost:.2f}")
    print(f"  Best Tour: {sga_best_individual.tour}")  # Printing the full tour
    print(f"  Execution Time: {sga_exec_time:.2f}s")

    print("\nSGA execution finished.")

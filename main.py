#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GA-TSP
# Genetic Algorithm (GA) for the Traveling Salesman Problem (TSP)
# Author: Guglielmo Cimolai
# Date: 22/05/2025

import random
import copy
import math  # For math.sqrt

# 1) Helpers for city and distance calculation
def euclidean_distance(city1, city2):
    # Compute Euclidean distance between any two cities
    return math.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

def calculate_distance_matrix(cities):
    num_cities = len(cities)
    # Initialize distance matrix with zeros using nested lists
    dist_matrix = [[0.0 for _ in range(num_cities)] for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            dist = euclidean_distance(cities[i], cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return dist_matrix

# 2) GA core components
def calculate_tour_cost(tour, distance_matrix):
    # Calculate total cost (= total path distance) for a full tour
    cost = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        cost += distance_matrix[tour[i]][tour[(i+1) % num_cities]]  # Accessing nested list
    return cost

class Individual:
    def __init__(self, tour):
        self.tour = list(tour)  # Ensure it's a list and a copy
        self.cost = float('inf')

    def calculate_cost(self, distance_matrix):
        self.cost = calculate_tour_cost(self.tour, distance_matrix)
        return self.cost

    def __lt__(self, other):  # For sorting by cost
        return self.cost < other.cost

    def __repr__(self):
        # Simplified representation for console
        return f"Cost: {self.cost:.2f}, Tour: {self.tour}"

# 3) Standard GA implementation (Selection-Crossover-Mutation loop)
def sga_initialize_population(num_cities, population_size):
    # Initialize population for the current generation
    population = []
    base_tour = list(range(num_cities))
    for _ in range(population_size):
        tour = random.sample(base_tour, num_cities)
        population.append(Individual(tour))
    return population

def sga_selection_tournament(population, k):  # k is tournament size
    # Define simple selection operator
    selected_parents = []
    for _ in range(len(population)):  # Create a mating pool of the same size as population
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
                  population_size, generations,
                  crossover_rate, mutation_rate, tournament_k):
    num_cities = len(cities)
    population = sga_initialize_population(num_cities, population_size)
    for ind in population:
        ind.calculate_cost(distance_matrix)

    population.sort()
    best_overall_individual = copy.deepcopy(population[0])

    print(f"\nRunning SGA for {num_cities} cities")
    print(f"Parameters: Pop_Size={population_size}, Gens={generations}, "
          f"CR={crossover_rate}, MR={mutation_rate}, Tourn_K={tournament_k}")
    print(f"Initial best: {best_overall_individual}")

    for gen in range(1, generations + 1):
        new_population = []
        # No elitism: new population is entirely offspring
        mating_pool = sga_selection_tournament(population, tournament_k)
        offspring_idx = 0
        while len(new_population) < population_size:
            parent1 = mating_pool[offspring_idx % len(mating_pool)]
            offspring_idx += 1
            parent2 = mating_pool[offspring_idx % len(mating_pool)]
            offspring_idx += 1
            if random.random() < crossover_rate:
                child = sga_crossover_ordered(parent1, parent2)
            else:  # If no crossover, clone one parent
                child = copy.deepcopy(random.choice([parent1, parent2]))

            sga_mutate_swap(child, mutation_rate)
            child.calculate_cost(distance_matrix)
            new_population.append(child)

        population = new_population
        population.sort()

        if population[0].cost < best_overall_individual.cost:
            best_overall_individual = copy.deepcopy(population[0])

        # Optional: minimal progress print
        # if gen%(generations//5) == 0 and generations >= 5: # Print ~5 updates
        #     print(f"Gen {gen}, Best: {best_overall_individual.cost:.2f}")

    print(f"SGA Final best: {best_overall_individual}")
    return best_overall_individual

# 4) Main execution
if __name__ == "__main__":
    # Fixed small set of cities for simplicity and deterministic testing
    # Each city is defined by (x, y) coordinates in 2D plane
    fixed_cities = [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120) # 10 cities
    ]
    num_cities_run = len(fixed_cities)

    # Configure GA parameters
    POP_SIZE = 50
    GENERATIONS = 100 # To be adjusted based on num_cities_run if needed
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15
    TOURNAMENT_K = 3

    # Setup problem
    distance_mat = calculate_distance_matrix(fixed_cities)
    print(f"Problem: {num_cities_run} fixed cities. Distance matrix calculated.")

    # Solve with Standard GA
    sga_best_individual = solve_tsp_sga(
        fixed_cities, distance_mat,
        population_size=POP_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        tournament_k=TOURNAMENT_K
    )

    # Final results output
    print("\n" + "="*10 + " SGA Run Complete " + "="*10)
    print(f"Final Best Individual found:")
    print(f"  Cost: {sga_best_individual.cost:.2f}")
    print(f"  Tour: {sga_best_individual.tour}")
    print("Execution finished.")

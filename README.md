# Genetic Algorithm (GA) and Hybrid GA-ACO for TSP with Visualization

## Overview

This project implements and compares two metaheuristic approaches for solving the Traveling Salesman Problem (TSP):
1.  A **Standard Genetic Algorithm (SGA)**.
2.  A **Hybrid Genetic Algorithm with Ant Colony Optimization elements (HGA-ACO)**, which leverages pheromone-based learning to guide the search process.

The project includes basic real-time visualization of the best routes found and the convergence of the algorithms using Matplotlib.

## Features

* Solves TSP using both SGA and HGA-ACO, allowing for performance comparison.
* **Random City Generation:** Creates a set of 2D city coordinates based on a specified number of cities and a seed for reproducibility.
* **Standard Genetic Algorithm (SGA):**
    * Tournament Selection
    * Ordered Crossover (OX1)
    * Swap Mutation
    * Elitism
* **Hybrid GA-ACO (HGA-ACO):**
    * Combines GA principles with ACO's constructive heuristic (ants building tours based on pheromones and distance).
    * Pheromone matrix update mechanism (evaporation and deposit based on solution quality).
    * Elitism
* **Visualization (Matplotlib):**
    * Live update of the best tour currently found by each algorithm (typically displayed sequentially in a shared plot area or showing final routes side-by-side).
    * A fitness convergence plot displaying the best tour cost against generations for both SGA and HGA-ACO.
* **Console Output:**
    * Details of the problem setup (number of cities, seed).
    * Parameters used for both algorithms.
    * Progress updates (best cost per N generations).
    * Final results: best tour, total cost, and execution time for both SGA and HGA-ACO.
    * A summary comparison of which algorithm performed better.

## How to Run

1.  **Prerequisites:**
    * Python 3.x installed.
    * Required libraries: `numpy` and `matplotlib`. You can install them using pip:
        `pip install numpy matplotlib`
2.  **Setup:**
    * Save the code as `hybrid_tsp_solver.py` (or your preferred name).
3.  **Configuration (Optional):**
    * You can modify `NUM_CITIES` and `CITY_SEED` variables at the beginning of the `if __name__ == "__main__":` block in the script to change the problem instance.
4.  **Execution:**
    * Open your terminal or command prompt.
    * Navigate to the directory where you saved the file.
    * Run the script using the command:
        `python hybrid_tsp_solver.py`

## Expected Output

* **Console:** Detailed logs including problem setup, GA/HGA-ACO parameters, periodic progress updates for both algorithms, their final best solutions (tour and cost), execution times, and a comparative summary.
* **Matplotlib Plot Window:**
    * A plot area showing the city locations and the best tour(s) found (represented in the image below).
    * A second plot showing the fitness convergence curves for both algorithms, allowing visual comparison of their performance over generations.
 
<img width="1624" alt="TSP_plotter_output" src="https://github.com/user-attachments/assets/45052cfc-50d6-4fa8-846e-917a4f9645e6" />

## Code Structure

The script is organized as follows:
* **City and Distance Management:** Functions for generating cities and calculating the distance matrix.
* **`Individual` Class:** Represents a tour and its cost.
* **`TSPPlotter` Class:** Manages the Matplotlib visualizations (route and convergence).
* **SGA Components:** Functions implementing the operators for the Standard GA (selection, crossover, mutation).
* **HGA-ACO Components:** Functions specific to the Hybrid GA-ACO, including pheromone initialization, ACO-guided tour construction, and pheromone updates.
* **Solver Functions:** `solve_tsp_sga` and `solve_tsp_hga_aco` which orchestrate each algorithm.
* **Main Execution Block (`if __name__ == "__main__":`)**: Handles problem setup, parameter configuration, calls both solvers, prints summary results, and displays the plots.

---

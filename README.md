# GA-TSP
A simple Genetic Algorithm (GA) implementation for solving the Traveling Salesman Problem (TSP).

## Overview
This project contains a simple Python implementation of a Genetic Algorithm (GA) designed to find solutions for the famous Traveling Salesman Problem (TSP). This version prioritizes simplicity in its codebase and outputs all results directly to the console. It's an excellent starting point for understanding the fundamental mechanics of a genetic algorithm applied to a classic combinatorial optimization problem.

## Features
* Solves the Traveling Salesman Problem using a simple Genetic Algorithm
* Uses a fixed, predefined set of city coordinates for straightforward demonstration and testing
* Implements core GA operators:
    * *Tournament Selection*
    * *Ordered Crossover (OX1)*
    * *Swap Mutation*
* Console-only output:
    * Details of the problem setup (number of cities)
    * Genetic Algorithm parameters used
    * The initial best solution found (cost and tour)
    * The final best solution (cost and tour) discovered by the SGA after all generations
* **No external library dependencies** beyond standard Python (`random`, `copy`, `math`)

## How to Run
1.  **Setup:**
    * Save the code as `ga_tsp.py` (or your preferred name)
2.  **Execution:**
    * Open your terminal or command prompt
    * Navigate to the directory where you saved the file
    * Run the script using the command: `python ga_tsp.py`

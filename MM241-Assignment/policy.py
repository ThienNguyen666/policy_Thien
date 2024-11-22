import random
from abc import abstractmethod

import numpy as np


class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)


class RandomPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Random choice a stock idx
                pos_x, pos_y = None, None
                for _ in range(100):
                    # random choice a stock
                    stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                    stock = observation["stocks"][stock_idx]

                    # Random choice a position
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)

                    if not self._can_place_(stock, (pos_x, pos_y), prod_size):
                        continue

                    break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class GreedyPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class GeneticPolicy(Policy):
    def __init__(self, population_size, generations, mutation_rate, stock_width, stock_height):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.stock_width = stock_width
        self.stock_height = stock_height

    def initialize_population(self, observation):
        # Initialize a population of random solutions
        population = []
        for _ in range(self.population_size):
            individual = []
            for _ in range(len(observation["products"])):
                stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_size = observation["products"][0]["size"]
                pos_x = random.randint(0, stock_w - prod_size[0])
                pos_y = random.randint(0, stock_h - prod_size[1])
                individual.append((stock_idx, pos_x, pos_y))
            population.append(individual)
        return population

    def calculate_fitness(self, observation, individual):
        # Calculate the fitness of an individual solution
        fitness = 0
        for i, (stock_idx, pos_x, pos_y) in enumerate(individual):
            stock = observation["stocks"][stock_idx]
            prod_size = observation["products"][i]["size"]
            if self._can_place_(stock, (pos_x, pos_y), prod_size):
                fitness += 1
        return fitness

    def select_parents(self, population, fitnesses):
        # Select two parents using tournament selection
        parents = []
        for _ in range(2):
            tournament = random.sample(list(zip(population, fitnesses)), 3)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def cross_over(self, parent1, parent2):
        child1 = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
        child2 = parent2[:len(parent2)//2] + parent1[len(parent1)//2:]
        return child1, child2

    def mutate(self, individual, observation):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_size = observation["products"][i]["size"]
                pos_x = random.randint(0, stock_w - prod_size[0])
                pos_y = random.randint(0, stock_h - prod_size[1])
                individual[i] = (stock_idx, pos_x, pos_y)
        return individual

    def get_action(self, observation, info):

        population = self.initialize_population(observation)
        for _ in range(self.generations):
            fitnesses = [self.calculate_fitness(observation, individual) for individual in population]
            parents = self.select_parents(population, fitnesses)
            children = []
            for _ in range(self.population_size // 2):
                child1, child2 = self.cross_over(parents[0], parents[1])
                children.extend([self.mutate(child1, observation), self.mutate(child2, observation)])
            population = children
        best_individual = max(population, key=lambda x: self.calculate_fitness(observation, x))
        return {"stock_idx": best_individual[0][0], "size": observation["products"][0]["size"], "position": (best_individual[0][1], best_individual[0][2])}
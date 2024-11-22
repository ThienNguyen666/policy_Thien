import random
from policy import Policy

class Policy2313237(Policy):
      def __init__(self,population_size,generations,mutation_rate,stock_width,stock_height):
            super().__init__()
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.stock_width = stock_width
            self.stock_height = stock_height

      
      def get_action(self, observation, info):
            """
            Implement the genetic algorithm here
            """
            return self.genetic_algorithm(observation,info)
      
      
      def initialize_population(self,observation):
            population = []
            products = observation["products"]

            for _ in range(self.population_size):
                  individual = [(random.randint(0, self.stock_width - product["size"][0]), 
                                 random.randint(0, self.stock_height - product["size"][1])) 
                                 for product in products]
                  population.append(individual)

            return population
      

      def calculate_fitness(self,individual,observation):            
            fitness = 0
            products = observation["products"]

            for i, product in enumerate(products):
                  pos_x, pos_y = individual[i]
                  if not self._can_place_(observation["stocks"], (pos_x, pos_y), product["size"]):
                        fitness += 1

            return fitness

      def select_parents(self,population,fitnesses):
            parents = []
            for i in range(2):
                  parent = random.choices(population, weights=fitnesses)[0]
                  parents.append(parent)
            return parents


      def cross_over(self,parent1,parent2):
            cutoff = random.randint(1,len(parent1) - 1)
            children1 = parent1[:cutoff] + parent2[cutoff:]
            children2 = parent2[:cutoff] + parent1[cutoff:]
            return children1,children2


      def mutate(self,individual,pieces):
            for i in range(len(individual)):
                  if random.random() < self.mutation_rate:
                        individual[i] = (random.randint(0,self.stock_width - pieces[0][0]),
                                         random.randint(0,self.stock_height - pieces[0][1]))


      def genetic_algorithm(self,observation,info):
            products = observation["products"]
            stock = observation["stocks"]

            population = self.initialize_population(products)

            for gen in range(self.generations):
                  fitnesses = [self.calculate_fitness(stock, individual, products) for individual in population]

                  new_population = []

                  while len(new_population) < self.population_size:
                        parent1, parent2 = self.select_parents(population, fitnesses)
                        children1, children2 = self.cross_over(parent1, parent2)

                        self.mutate(children1, products)
                        self.mutate(children2, products)

                        new_population.extend([children1, children2])

                  population = new_population[: self.population_size]

                  fitnesses = [self.calculate_fitness(stock, individual, products) for individual in population]

                  best_index = fitnesses.index(min(fitnesses))
                  return population[best_index]
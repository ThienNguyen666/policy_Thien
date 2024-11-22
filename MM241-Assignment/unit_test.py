import unittest
from student_submissions.s2313237.policy2313237 import Policy2313237
import random

class TestPolicy2313237(unittest.TestCase):
    def test_init(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        self.assertEqual(policy.population_size, 10)
        self.assertEqual(policy.generations, 5)
        self.assertEqual(policy.mutation_rate, 0.1)
        self.assertEqual(policy.stock_width, 10)
        self.assertEqual(policy.stock_height, 10)

    def test_initialize_population(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        pieces = [(1, 1), (2, 2), (3, 3)]
        population = policy.initialize_population(pieces)
        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertEqual(len(individual), 3)
            for pos_x, pos_y in individual:
                self.assertLessEqual(pos_x, 10 - pieces[0][0])
                self.assertLessEqual(pos_y, 10 - pieces[0][1])

    def test_calculate_fitness(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        stock = [[0]*10 for _ in range(10)]
        pieces = [(1, 1), (2, 2), (3, 3)]
        individual = [(0, 0), (1, 1), (2, 2)]
        fitness = policy.calculate_fitness(stock, individual, pieces)
        self.assertEqual(fitness, 14)

    def test_select_parents(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        population = [[(0, 0), (1, 1), (2, 2)] for _ in range(10)]
        fitnesses = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        parents = policy.select_parents(population, fitnesses)
        self.assertEqual(len(parents), 2)
        self.assertIn(parents[0], population)
        self.assertIn(parents[1], population)

    def test_cross_over(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        parent1 = [(0, 0), (1, 1), (2, 2)]
        parent2 = [(3, 3), (4, 4), (5, 5)]
        children = policy.cross_over(parent1, parent2)
        self.assertEqual(len(children), 2)
        self.assertEqual(len(children[0]), 3)
        self.assertEqual(len(children[1]), 3)

    def test_mutate(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        individual = [(0, 0), (1, 1), (2, 2)]
        pieces = [(1, 1), (2, 2), (3, 3)]
        policy.mutate(individual, pieces)
        self.assertEqual(len(individual), 3)
        for pos_x, pos_y in individual:
            self.assertLessEqual(pos_x, 10 - pieces[0][0])
            self.assertLessEqual(pos_y, 10 - pieces[0][1])

    def test_genetic_algorithm(self):
        policy = Policy2313237(10, 5, 0.1, 10, 10)
        observation = {"pieces": [(1, 1), (2, 2), (3, 3)], "stock": [[0]*10 for _ in range(10)]}
        info = {}
        action = policy.get_action(observation, info)
        self.assertEqual(len(action), 3)
        for pos_x, pos_y in action:
            self.assertLessEqual(pos_x, 10 - observation["pieces"][0][0])
            self.assertLessEqual(pos_y, 10 - observation["pieces"][0][1])

if __name__ == '__main__':
    unittest.main()
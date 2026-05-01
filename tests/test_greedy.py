import unittest
import pandas as pd
from src.algorithms.greedy import GreedyOptimizer

class TestGreedyOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = GreedyOptimizer()

    def test_optimal_locations_basic(self):
        data = pd.DataFrame({
            'location': ['A', 'B', 'C', 'D'],
            'demand': [10, 20, 30, 40]
        })
        expected_locations = ['D', 'C']  # Example expected output
        optimal_locations = self.optimizer.find_optimal_locations(data)
        self.assertEqual(optimal_locations, expected_locations)

    def test_optimal_locations_edge_case(self):
        data = pd.DataFrame({
            'location': ['A'],
            'demand': [50]
        })
        expected_locations = ['A']
        optimal_locations = self.optimizer.find_optimal_locations(data)
        self.assertEqual(optimal_locations, expected_locations)

    def test_optimal_locations_no_demand(self):
        data = pd.DataFrame({
            'location': ['A', 'B', 'C'],
            'demand': [0, 0, 0]
        })
        expected_locations = []  # No locations should be chosen
        optimal_locations = self.optimizer.find_optimal_locations(data)
        self.assertEqual(optimal_locations, expected_locations)

if __name__ == '__main__':
    unittest.main()
import unittest

import pandas as pd

from src.algorithms.greedy import GreedyOptimizer


class TestGreedyOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = GreedyOptimizer(max_locations=2, max_response_time=500)

    def test_optimal_locations_basic(self):
        traveltimes = pd.DataFrame(
            [[100, 1000, 1000], [1000, 100, 1000], [1000, 1000, 100]],
            index=['A', 'B', 'C'],
            columns=['A', 'B', 'C'],
        )
        demand = pd.DataFrame({'Postal codes': ['A', 'B', 'C'], 'Demand': [10, 20, 30]})

        result = self.optimizer.find_optimal_locations(traveltimes, demand)

        self.assertEqual(result['chosen_bases'], ['C', 'B'])
        self.assertEqual(result['covered_people'], 50.0)

    def test_optimal_locations_edge_case(self):
        traveltimes = pd.DataFrame(
            [[1000, 1000], [1000, 1000]],
            index=['A', 'B'],
            columns=['A', 'B'],
        )
        demand = pd.DataFrame({'Postal codes': ['A', 'B'], 'Demand': [50, 25]})

        result = self.optimizer.find_optimal_locations(traveltimes, demand)

        self.assertEqual(result['chosen_bases'], [])
        self.assertEqual(result['covered_people'], 0.0)

    def test_optimal_locations_no_demand(self):
        traveltimes = pd.DataFrame(
            [[100, 100], [100, 100]],
            index=['A', 'B'],
            columns=['A', 'B'],
        )
        demand = pd.DataFrame({'Postal codes': ['A', 'B'], 'Demand': [0, 0]})

        result = self.optimizer.find_optimal_locations(traveltimes, demand)

        self.assertEqual(result['covered_people'], 0.0)
        self.assertEqual(result['covered_people_pct'], 0)


if __name__ == '__main__':
    unittest.main()

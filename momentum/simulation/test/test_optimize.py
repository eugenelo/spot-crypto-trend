import unittest
import pandas as pd
import numpy as np

from simulation.optimize import generate_parameter_sets


class TestGenerateParameterSets(unittest.TestCase):
    def test_missing_required(self):
        # Missing required param
        optimize_params = {}
        with self.assertRaises(RuntimeError):
            parameter_sets = generate_parameter_sets(optimize_params)

    def test_invalid_spec(self):
        # Invalid use of dictionary
        optimize_params = {
            "signal": "rohrbach_exponential",
            "direction": "LongOnly",
            "volatility_target": {"invalid_key": "invalid_value"},
        }
        with self.assertRaises(AssertionError):
            parameter_sets = generate_parameter_sets(optimize_params)

    def test_nominal(self):
        optimize_params = {
            "signal": "rohrbach_exponential",
            "direction": "LongOnly",
            "volatility_target": {
                "step_size": 0.05,
                "num_steps": 10,
            },
            "rebalancing_buffer": [0.01, 0.02, 0.05, 0.1, 0.5],
            "min_daily_volume": None,
            "with_fees": [False, True],
        }
        parameter_sets = generate_parameter_sets(optimize_params)
        self.assertEqual(100, len(parameter_sets))
        self.assertEqual(
            "volatility_target: 0.05, rebalancing_buffer: 0.01, with_fees: False",
            parameter_sets[0].name,
        )


if __name__ == "__main__":
    unittest.main()

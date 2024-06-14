import unittest

import numpy as np
import pandas as pd

from data.utils import interpolate_missing_ids, missing_elements


class TestDataUtils(unittest.TestCase):
    def test_missing_elements(self):
        full_seq = list(range(10))
        self.assertEqual([], missing_elements(full_seq))

        missing_seq = [1, 3, 5, 6, 7, 10]
        missing_items = [2, 4, 8, 9]
        self.assertEqual(missing_items, missing_elements(missing_seq))

    def test_interpolate_missing_ids(self):
        s_ffill = pd.Series([1, np.nan, np.nan, np.nan])
        s_expected = [1, 2, 3, 4]
        self.assertEqual(s_expected, interpolate_missing_ids(s_ffill).to_list())

        s_bfill = pd.Series([np.nan, np.nan, np.nan, 4])
        s_expected = [1, 2, 3, 4]
        self.assertEqual(s_expected, interpolate_missing_ids(s_bfill).to_list())

        s_both = pd.Series([np.nan, 2, np.nan, np.nan, np.nan, 6, 7, np.nan, np.nan])
        s_expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(s_expected, interpolate_missing_ids(s_both).to_list())

        s_ffill_over_bfill = pd.Series([1, 2, np.nan, np.nan, 9])
        s_expected = [1, 2, 3, 4, 9]
        self.assertEqual(
            s_expected, interpolate_missing_ids(s_ffill_over_bfill).to_list()
        )


if __name__ == "__main__":
    unittest.main()

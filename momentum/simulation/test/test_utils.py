import unittest

from simulation.utils import get_segment_mask, rebal_freq_supported


class TestSimulationUtils(unittest.TestCase):
    def test_rebal_freq_supported(self):
        # Supported
        self.assertTrue(rebal_freq_supported("1d"))
        self.assertTrue(rebal_freq_supported("2d"))
        self.assertTrue(rebal_freq_supported("D"))
        self.assertTrue(rebal_freq_supported("s"))
        self.assertTrue(rebal_freq_supported("min"))
        self.assertTrue(rebal_freq_supported("h"))
        self.assertTrue(rebal_freq_supported("ms"))
        self.assertTrue(rebal_freq_supported("us"))
        self.assertTrue(rebal_freq_supported("ns"))
        # Unsupported
        self.assertFalse(rebal_freq_supported("W"))
        self.assertFalse(rebal_freq_supported("ME"))
        self.assertFalse(rebal_freq_supported("YE"))
        self.assertFalse(rebal_freq_supported("QE"))

    def test_get_segment_mask(self):
        periods_per_day = 1
        rebalancing_freq = "1d"
        self.assertEqual(
            1,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 1
        rebalancing_freq = "2d"
        self.assertEqual(
            2,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 1
        rebalancing_freq = "5d"
        self.assertEqual(
            5,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 2
        rebalancing_freq = "4d"
        self.assertEqual(
            8,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 24
        rebalancing_freq = "2d"
        self.assertEqual(
            48,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 4
        rebalancing_freq = "6h"
        self.assertEqual(
            1,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 24
        rebalancing_freq = "7d"
        self.assertEqual(
            168,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 86400
        rebalancing_freq = "1s"
        self.assertEqual(
            1,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )

        periods_per_day = 1440 * 2
        rebalancing_freq = "1min"
        self.assertEqual(
            2,
            get_segment_mask(
                periods_per_day=periods_per_day, rebalancing_freq=rebalancing_freq
            ),
        )


if __name__ == "__main__":
    unittest.main()

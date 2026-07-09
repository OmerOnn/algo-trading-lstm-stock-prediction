import unittest

import numpy as np

from src.decision import apply_class_thresholds, tune_class_thresholds


class DecisionThresholdTest(unittest.TestCase):
    def test_apply_thresholds_defaults_to_hold_when_action_probs_are_low(self):
        probabilities = np.array(
            [
                [0.20, 0.70, 0.10],
                [0.35, 0.30, 0.35],
                [0.10, 0.15, 0.75],
            ]
        )

        predicted = apply_class_thresholds(probabilities, {"sell": 0.4, "buy": 0.6})

        np.testing.assert_array_equal(predicted, np.array([1, 1, 2]))

    def test_tuning_can_raise_action_thresholds_to_reduce_false_positives(self):
        probabilities = np.array(
            [
                [0.42, 0.40, 0.18],
                [0.10, 0.20, 0.70],
                [0.18, 0.60, 0.22],
                [0.15, 0.18, 0.67],
                [0.41, 0.44, 0.15],
                [0.12, 0.72, 0.16],
            ]
        )
        true_labels = np.array([1, 2, 1, 2, 1, 1])

        thresholds = tune_class_thresholds(
            probabilities,
            true_labels,
            threshold_grid=[0.4, 0.5, 0.6, 0.7],
            minimum_action_rate=0.05,
        )

        baseline_predictions = apply_class_thresholds(probabilities, {"sell": 0.4, "buy": 0.4})
        tuned_predictions = apply_class_thresholds(probabilities, thresholds)

        baseline_action_rate = float(np.mean(baseline_predictions != 1))
        tuned_action_rate = float(np.mean(tuned_predictions != 1))

        self.assertLessEqual(tuned_action_rate, baseline_action_rate)


if __name__ == "__main__":
    unittest.main()

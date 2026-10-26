import unittest
from experiments.evaluate_prediction_handoff import regression_metrics, classification_metrics

class PredictionDiagnosticTests(unittest.TestCase):
    def test_regression_metrics_exact_simple_case(self):
        result = regression_metrics([1.0, 2.0], [1.0, 4.0])
        self.assertAlmostEqual(result['mae'], 1.0)
        self.assertAlmostEqual(result['rmse'], (2.0 ** 0.5))
        self.assertAlmostEqual(result['r2'], -7.0)

    def test_classification_confusion_matrix(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result['confusion_matrix_tn_fp_fn_tp'], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result['precision'], 0.5)
        self.assertAlmostEqual(result['recall'], 0.5)
        self.assertAlmostEqual(result['f1'], 0.5)

if __name__ == '__main__':
    unittest.main()

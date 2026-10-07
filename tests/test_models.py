import unittest
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import fit_score, experiment

class ModelTests(unittest.TestCase):
    def test_known_linear_fit(self):
        x = np.arange(20).reshape(-1,1)
        beta, aic, aicc, bic = fit_score(x, 3+2*x[:,0])
        np.testing.assert_allclose(beta, [3,2], atol=1e-10)
        self.assertGreater(aicc, aic)

    def test_rank_deficiency(self):
        with self.assertRaises(ValueError):
            fit_score(np.ones((20,2)), np.arange(20))

    def test_selection_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = experiment(42, 25, directory)
        self.assertEqual(len(scores), 15)
        self.assertAlmostEqual(scores.akaike_weight.sum(), 1)
        self.assertAlmostEqual(scores.bootstrap_aicc_frequency.sum(), 1)
        self.assertAlmostEqual(scores.bootstrap_bic_frequency.sum(), 1)

if __name__ == '__main__':
    unittest.main()

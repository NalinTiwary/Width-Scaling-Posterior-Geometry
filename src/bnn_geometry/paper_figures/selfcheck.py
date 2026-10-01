"""Run with: python selfcheck.py. Synthetic fixtures only, not paper results."""
import unittest
import numpy as np
from figure_metrics import (acf_1d, acf_replicate, aggregate_replicates,
    spectral_target, spectral_replicates, reconstruct_rejected_moves,
    validate_iterations)

class CalculationChecks(unittest.TestCase):
    def test_fft_matches_direct(self):
        x = np.random.default_rng(17).normal(size=1024)
        z = x - x.mean()
        expected = np.array([np.dot(z[:len(z)-k], z[k:]) / np.dot(z,z)
                             for k in range(101)])
        np.testing.assert_allclose(acf_1d(x, 100), expected, atol=1e-10, rtol=0)
        self.assertEqual(acf_1d(x,100)[0],1)

    def test_affine_invariance(self):
        x = np.random.default_rng(18).normal(size=2048)
        np.testing.assert_allclose(acf_1d(x,100),acf_1d(-7*x+200,100),atol=1e-10,rtol=0)

    def test_invalid_input_is_rejected(self):
        for x in [np.ones(1000), np.array([1.,np.nan,2.]), np.array([1.,np.inf,2.])]:
            with self.assertRaises(ValueError): acf_1d(x,1)
        with self.assertRaises(ValueError): acf_1d(np.arange(100.),100)
        with self.assertRaises(ValueError): spectral_target([.5,-.1])

    def test_chain_and_replicate_aggregation(self):
        x = np.random.default_rng(19).normal(size=(4,2048))
        x += np.array([0.,20.,-40.,80.])[:,None]
        result = acf_replicate(x,100,1024)
        expected = np.mean([acf_1d(row[-1024:],100) for row in x],axis=0)
        np.testing.assert_allclose(result['replicate_acf'],expected,atol=1e-12)
        aggregate = aggregate_replicates(np.array([expected,expected+.02,expected-.03]))
        np.testing.assert_allclose(aggregate['median'],expected,atol=1e-12)
        np.testing.assert_allclose(aggregate['replicate_min'],expected-.03,atol=1e-12)
        with self.assertRaises(ValueError): acf_replicate(x,100,4096)

    def test_spectral_quantiles_and_counts(self):
        x = np.array([.7,.8,1.,1.05])
        s = spectral_target(x)
        self.assertEqual((s['n_inside'],s['n_outside'],s['n_ambiguous']),(2,1,1))
        self.assertEqual(s['inside_fraction_lower'],.5)
        self.assertEqual(s['inside_fraction_upper'],.75)
        np.testing.assert_allclose([s['q50'],s['q95'],s['q99']],np.quantile(x,[.5,.95,.99]))
        merged = spectral_replicates([s,s,s])
        self.assertEqual(merged['n_inspected'],12)
        self.assertEqual(merged['q50']['median'],s['q50'])

    def test_rejections_and_indices(self):
        out = reconstruct_rejected_moves(10.,np.array([False,False,True,False,True]),[9.,8.])
        np.testing.assert_equal(out,[10.,10.,9.,9.,8.])
        np.testing.assert_equal(reconstruct_rejected_moves(2.,np.array([False,False]),[]),[2.,2.])
        validate_iterations(np.arange(200,205),5)
        for idx in [np.array([0,1,3]),np.array([0,1,1]),np.array([0.,1.,2.])]:
            with self.assertRaises(ValueError): validate_iterations(idx,3)

    def test_ar1_sanity(self):
        rng = np.random.default_rng(20261001)
        a = .97
        n = 200000
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1,n): x[i] = a*x[i-1]+np.sqrt(1-a*a)*e[i]
        acf = acf_1d(x,100)
        lags = np.array([1,10,50,100])
        self.assertLess(float(np.max(np.abs(acf[lags]-a**lags))),.025)

if __name__ == '__main__':
    unittest.main(verbosity=2)

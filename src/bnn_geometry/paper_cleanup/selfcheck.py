"""Meaningful calculation and rendering checks; no campaign data are used."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt
from clean_figures import spectral_figure, acf_figure, check_legends, empirical_cdf, threshold_counts, read_theory_context
from supplement_metrics import (acf_1d, probability_acfs, loss_acfs,
                                exact_lag_indices, duration_window, aggregate_replicates)


def fixture_csvs(directory):
    """Synthetic layout data only, intentionally distinct from campaign tables."""
    d = Path(directory)
    spectral = d/'synthetic_spectral.csv'
    with spectral.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['width','replicate','chain','draw','T'])
        rng=np.random.default_rng(2701)
        for i,m in enumerate([32,64,128,256]):
            for r in [0,1,2]:
                for c in range(4):
                    for j in range(64):
                        value=1.94+0.01*i+(r-1)*.002+rng.normal(scale=.055/(i+1)**.4)
                        w.writerow([m,r,c,16*(j+1),value])
    context=d/'synthetic_context.json'
    context.write_text(json.dumps({'weight_layers':3,'n':128,'sigma':1.,'r0':0.,
                                   'hidden_center_max_normalized_opnorm':0.,
                                   'statistic':'opnorm_W2_over_sqrt_m',
                                   'target_manifest_sha256':hashlib.sha256(b'SYNTHETIC FIXTURE').hexdigest()}))
    acf = d/'synthetic_acf.csv'
    with acf.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['architecture','width','replicate','lag','acf'])
        for a,ww in [('shallow',[64,256,1024,4096]),('deep',[32,64,128,256])]:
            for i,m in enumerate(ww):
                for r in [0,1,2]:
                    for k in range(401):
                        w.writerow([a,m,r,k,np.exp(-k/(45-6*i+r))])
    return spectral, acf, context


class Checks(unittest.TestCase):
    def test_ecdf_threshold_equality_and_conversion(self):
        values=np.array([1.5,2.,2.1,2.25])
        cuts=np.array([1.,1.5,2.,2.1,2.5])
        np.testing.assert_allclose(empirical_cdf(values,cuts),[np.mean(values<=a) for a in cuts])
        c=threshold_counts(values,2.)
        self.assertEqual(c['n_above'],2)
        self.assertEqual(c['n_at_or_below'],2)
        self.assertEqual(c['n_near_threshold'],1)
        self.assertEqual(c['n_definitely_above'],2)
        self.assertEqual(c['n_possibly_above'],3)
        self.assertEqual(np.sum(values/2.5>.8),c['n_above'])

    def test_cdf_replicate_aggregation_and_context(self):
        # Unequal pools detect accidental pooling rather than equal-replicate summary.
        samples=[np.array([1.]),np.array([3.,3.,3.]),np.array([1.,1.])]
        curves=np.stack([empirical_cdf(x,[2.]) for x in samples])
        self.assertEqual(float(np.median(curves)),1.)
        self.assertEqual(float(empirical_cdf(np.concatenate(samples),[2.])[0]),.5)
        with tempfile.TemporaryDirectory() as td:
            sp,ac,context=fixture_csvs(td)
            self.assertEqual(read_theory_context(context)[1],2.)
            raw=json.loads(context.read_text());raw['sigma']=2
            context.write_text(json.dumps(raw))
            with self.assertRaises(ValueError): read_theory_context(context)

    def test_fft_against_direct_and_affine(self):
        rng=np.random.default_rng(20261001)
        x=rng.normal(size=61)
        z=x-x.mean(); direct=np.array([np.dot(z[:len(x)-k],z[k:])/np.dot(z,z) for k in range(31)])
        np.testing.assert_allclose(acf_1d(x,30),direct,atol=1e-12)
        np.testing.assert_allclose(acf_1d(-7*x+91,30),direct,atol=1e-12)

    def test_prediction_normalizes_each_point(self):
        rng=np.random.default_rng(7)
        p=rng.uniform(0.2,0.8,size=(4,127,8))
        result=probability_acfs(p,20)
        expected=np.stack([[acf_1d(p[c,:,j],20) for j in range(8)] for c in range(4)])
        np.testing.assert_allclose(result['mean8'],expected.mean(axis=(0,1)))
        # Point-specific affine rescaling preserves the prescribed summary.
        scale=np.linspace(.01,.3,8)
        np.testing.assert_allclose(probability_acfs(.3+p*scale,20)['mean8'],result['mean8'],atol=1e-12)

    def test_bad_probability_inputs_and_degeneracy(self):
        with self.assertRaises(ValueError): probability_acfs(np.ones((4,60,8))*.5,10)
        with self.assertRaises(ValueError): probability_acfs(np.zeros((4,60,7)),10)
        with self.assertRaises(ValueError): probability_acfs(np.ones((4,60,8))*2,10)

    def test_step_index_and_duration(self):
        np.testing.assert_array_equal(exact_lag_indices([.25,.5,1,2],.01,400),[25,50,100,200])
        np.testing.assert_array_equal(exact_lag_indices([.25,.5,1,2],.005,400),[50,100,200,400])
        with self.assertRaises(ValueError): exact_lag_indices([.253],.01,400)
        x=np.tile(np.arange(1100),(4,1))
        self.assertEqual(duration_window(x,.01,10).shape,(4,1000))
        self.assertEqual(duration_window(x,.01,10)[0,0],100)

    def test_replicate_and_chain_separation(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(4,80))+np.arange(4)[:,None]*20
        expected=np.mean([acf_1d(row,20) for row in x],axis=0)
        np.testing.assert_allclose(loss_acfs(x,20)['mean4'],expected)
        a=aggregate_replicates(np.stack([expected,expected*.8,expected*.9]))
        self.assertTrue(np.all(a['low']<=a['median']))
        self.assertTrue(np.all(a['median']<=a['high']))

    def test_render_integrity_and_input_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            sp,ac,context=fixture_csvs(td)
            for width in [5.5,6.75]:
                for make,path,nleg in [(spectral_figure,sp,2),(acf_figure,ac,2)]:
                    kwargs={'context_path':context,'expected_per_chain':64} if make is spectral_figure else {}
                    fig,meta=make(path,width_in=width,**kwargs)
                    fig.canvas.draw()
                    check_legends(fig,nleg)
                    self.assertEqual(len(fig.axes),2)
                    if make is acf_figure:
                        np.testing.assert_allclose(fig.axes[0].get_ylim(),fig.axes[1].get_ylim())
                    else:
                        self.assertEqual(sum(r['n'] for r in meta['threshold_counts']),3072)
                        for m in [32,64,128,256]:
                            cc=meta['_coverage_arrays'][f'width_{m}_cdf_replicates']
                            self.assertTrue(np.all(np.diff(cc,axis=1)>=0))
                            self.assertTrue(np.all((cc>=0)&(cc<=1)))
                    renderer=fig.canvas.get_renderer()
                    fb=fig.bbox
                    for ax in fig.axes:
                        bb=ax.get_tightbbox(renderer)
                        self.assertGreaterEqual(bb.x0,-1)
                        self.assertLessEqual(bb.x1,fb.x1+1)
                        self.assertGreaterEqual(bb.y0,-1)
                        self.assertLessEqual(bb.y1,fb.y1+1)
                    plt.close(fig)
            # Duplicated keys must not be silently pooled or overwritten.
            lines=sp.read_text().splitlines()
            sp.write_text('\n'.join(lines+[lines[1]])+'\n')
            with self.assertRaises(ValueError): spectral_figure(sp,context_path=context,expected_per_chain=64)
            # Missing a lag must not be interpolated or silently cropped away.
            lines=ac.read_text().splitlines()
            ac.write_text('\n'.join(lines[:-1])+'\n')
            with self.assertRaises(ValueError): acf_figure(ac)


if __name__=='__main__':
    unittest.main(verbosity=2)

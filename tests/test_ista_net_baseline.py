import copy
import math
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from three_dried.imaging.rmist_kernel import RMISTKernel
from three_dried.imaging.sar_operator import MeasuredSAR, classical_ista
from three_dried.imaging.basic_bp import direct_slice_bp, direct_range_bp
from three_dried.models.ista_net import ISTANet, ista_net_loss, to_channels, to_complex
from three_dried.simulation.ista_net_data import PointScenes
from three_dried.baseline.config import load_config
from three_dried.baseline.validation_metrics import complex_nmse, point_localization, sidelobes_single_point
from test_baseline_suite import small_config

ROOT = Path(__file__).resolve().parents[1]


class ISTANetBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261027)
        torch.set_num_threads(2)
        self.kernel = RMISTKernel(78.8e9,.001,.002,16)
        self.mask = torch.zeros(16,16,dtype=torch.bool)
        self.mask[2:14:2,2:15] = True

    def test_measured_adjoint_and_gradient_complex128(self):
        op=MeasuredSAR(self.kernel,.614,self.mask,dtype=torch.complex128)
        x=torch.randn(2,16,16,dtype=torch.complex128,requires_grad=True)
        y=torch.randn_like(x)
        lhs=torch.vdot(op(x).flatten(),y.flatten())
        rhs=torch.vdot(x.flatten(),op.adjoint(y).flatten())
        self.assertLess(float((lhs-rhs).abs()/max(lhs.abs(),rhs.abs())),1e-12)
        loss=.5*(op(x)-y*self.mask).abs().square().sum();loss.backward()
        torch.testing.assert_close(x.grad,op.adjoint(op(x.detach())-y*self.mask),rtol=1e-12,atol=1e-12)

    def test_physical_scaling_does_not_change_forward(self):
        op=MeasuredSAR(self.kernel,.614,self.mask)
        x=torch.randn(2,16,16,dtype=torch.complex64)
        expected=op.encode_physical_measurement(self.kernel.rm(x,.614))
        torch.testing.assert_close(op(x),expected,rtol=1e-5,atol=1e-6)

    def test_network_shape_complex_phase_and_all_phase_backpropagation(self):
        op=MeasuredSAR(self.kernel,.614,self.mask)
        truth=torch.zeros(2,16,16,dtype=torch.complex64)
        truth[:,8,8]=1+.5j
        net=ISTANet(phases=3,features=8)
        pred,sym=net(op(truth),op,return_symmetry=True)
        self.assertEqual(tuple(pred.shape),tuple(truth.shape));self.assertTrue(torch.is_complex(pred))
        self.assertEqual(len(sym),3)
        self.assertTrue(torch.equal(to_complex(to_channels(truth)),truth))
        loss=ista_net_loss(pred,truth,sym);loss.backward()
        for phase in net.phases:
            for p in phase.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()),0)

    def test_short_single_scene_training_reduces_reconstruction_loss(self):
        op=MeasuredSAR(self.kernel,.614,self.mask)
        gt=torch.zeros(1,16,16,dtype=torch.complex64);gt[0,8,8]=1+.5j
        net=ISTANet(phases=3,features=8)
        y=op(gt);optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        initial=float((net(y,op)-gt).abs().square().sum())
        for _ in range(30):
            optimizer.zero_grad();prediction,sym=net(y,op,return_symmetry=True)
            ista_net_loss(prediction,gt,sym).backward();optimizer.step()
        final=float((net(y,op)-gt).abs().square().sum())
        self.assertLess(final,initial*.9)

    def test_ista_uses_monotonic_measured_objective(self):
        op=MeasuredSAR(self.kernel,.614,self.mask)
        gt=torch.randn(1,16,16,dtype=torch.complex64)
        _,history=classical_ista(op(gt),op,30,.01)
        self.assertTrue(np.all(np.diff(history)<=1e-5))

    def test_procedural_split_hashes_are_disjoint_and_repeatable(self):
        cfg=small_config()
        settings=yaml.safe_load((ROOT/'configs/ista_net_baseline.yaml').read_text())['dataset']
        settings=copy.deepcopy(settings);settings['split_counts']={s:10 for s in settings['split_counts']}
        sets={s:{PointScenes(cfg,settings,s).scene(i)[2] for i in range(10)} for s in settings['split_counts']}
        self.assertFalse(sets['train']&sets['validation']);self.assertFalse(sets['train']&sets['test'])
        d=PointScenes(cfg,settings,'train')
        self.assertEqual(d.scene(0)[2],d.scene(0)[2])

    def test_literal_bp_phase_compensation_coherently_sums(self):
        x=np.array([-.002,0.,.002]);z=np.array([-.004,0.,.004]);depth=.6
        k=4*math.pi*78.8e9/299792458.
        echo=np.empty((3,3),complex)
        for iz,az in enumerate(z):
            for ix,ax in enumerate(x):
                echo[iz,ix]=np.exp(1j*k*math.sqrt(depth**2+az**2+ax**2))
        full=direct_slice_bp(echo,x,z,[0.],[0.],depth,78.8e9,np.ones((3,3),bool))
        self.assertLess(abs(full[0,0]-9),1e-10)
        mask=np.zeros((3,3),bool);mask[0]=True
        sparse=direct_slice_bp(echo,x,z,[0.],[0.],depth,78.8e9,mask)
        self.assertLess(abs(sparse[0,0]-3),1e-10)

    def test_metrics_do_not_align_phase_or_hide_missing_targets(self):
        gt=np.zeros((1,16,16),complex);gt[0,8,8]=1j
        self.assertEqual(complex_nmse(gt,gt),0.)
        self.assertAlmostEqual(complex_nmse(-gt,gt),4.)
        metrics=point_localization(gt,gt,.001,.002)
        self.assertEqual(metrics['localization_max_m'],0.)
        multi=gt.copy();multi[0,4,4]=1
        self.assertEqual(sidelobes_single_point(multi,multi)['pslr_islr_status'],'SKIPPED_MULTIPLE_TARGETS')

    def test_basic_range_bp_preserves_complex_interpolation(self):
        profiles=np.array([[[1j]],[[3j]],[[5j]]])
        output=direct_range_bp(profiles,np.array([.5,.6,.7]),[0.],[0.],[0.],[.65],[0.],0.,np.ones((1,1),bool))
        self.assertAlmostEqual(output[0,0,0].real,0.)
        self.assertAlmostEqual(output[0,0,0].imag,4.)


if __name__=='__main__':
    unittest.main()

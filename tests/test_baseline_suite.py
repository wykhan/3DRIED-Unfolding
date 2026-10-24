"""Numerical contracts for the independent baseline suite."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import yaml

from three_dried.baseline.config import load_config,ranges_m,support_slices,support
from three_dried.baseline.sampling import nested_masks
from three_dried.baseline.operators import SliceRM
from three_dried.baseline.methods import backproject,green_kernel,cv_ista,rm_iteration,load_rmist_parameters
from three_dried.baseline.datasets import point_scene

ROOT=Path(__file__).resolve().parents[1]

def small_config(n=16):
    cfg=load_config(ROOT/'configs/baseline_3dried.yaml')
    cfg['data'].update(nx=n-3,nz=n-4)
    cfg['aperture'].update(horizontal_span_m=(n-4)*.001,vertical_span_m=(n-5)*.002)
    cfg['imaging'].update(n_fft_space=n,matlab_range_bins=[17],representative_matlab_bin=17,
                           x_offset_mm=0,z_offset_mm=0)
    cfg['simulation']['points']=[dict(matlab_bin=17,x_m=0,z_m=0,amplitude=1,phase_rad=.3)]
    return cfg

class AcquisitionAndSamplingTests(unittest.TestCase):
    def test_shared_configuration_matches_accepted_hammer(self):
        cfg=load_config(ROOT/'configs/baseline_3dried.yaml')
        accepted=yaml.safe_load((ROOT/'configs/hammer.yaml').read_text())
        for group in ['data','radar','aperture']:
            for key,value in accepted[group].items():self.assertEqual(cfg[group][key],value,(group,key))
        self.assertEqual(cfg['data']['adc_samples'],256)
        self.assertAlmostEqual(cfg['radar']['sampled_bandwidth_hz'],cfg['radar']['chirp_slope_hz_s']*256/5e6)
        self.assertAlmostEqual(ranges_m(cfg)[5],.6143103953,places=9)

    def test_exact_nested_counts_and_height_structure(self):
        for pattern in ['random_2d','height_lines']:
            masks=nested_masks((200,407),[.3,.5,.7,1],7,pattern)
            for sr,m in masks.items():
                self.assertEqual(int(m.sum()),int(round(81400*sr)))
                self.assertTrue(np.array_equal(m,nested_masks((200,407),[sr],7,pattern)[sr]))
                if pattern=='height_lines': self.assertTrue(np.all(m==m[:,:1]))
            for a,b in zip([.3,.5,.7],[.5,.7,1]):self.assertTrue(np.all(masks[a]<=masks[b]))

    def test_padding_is_not_acquired_data(self):
        cfg=small_config();op=SliceRM(cfg,ranges_m(cfg)[0])
        actual=op.pad(np.ones((12,13),dtype=bool))
        self.assertEqual(int(actual.sum()),156)
        self.assertTrue(np.array_equal(actual,support(cfg)))

class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.cfg=small_config();self.op=SliceRM(self.cfg,ranges_m(self.cfg)[0]);self.rng=np.random.default_rng(4)
    def complex(self):return self.rng.normal(size=(16,16))+1j*self.rng.normal(size=(16,16))

    def test_masked_forward_adjoint_inner_product(self):
        mask=self.op.pad(nested_masks((12,13),[.5],6)[.5]);x,y=self.complex(),self.complex()
        lhs=np.vdot(self.op.measured_forward(x,mask),y)
        rhs=np.vdot(x,self.op.measured_adjoint(y,mask))
        self.assertLess(abs(lhs-rhs)/max(abs(lhs),abs(rhs)),1e-12)
        # The reciprocal RM inverse is deliberately not the adjoint.
        self.assertGreater(np.linalg.norm(self.op.inverse(y)-self.op.adjoint(y)),1)

    def test_rma_matches_accepted_implementation(self):
        from three_dried.imaging.rma import reconstruct_slice
        s=self.complex()[:12,:13]
        got=self.op.inverse(self.op.pad(s))
        ref=reconstruct_slice(s,self.op.range_m,78.8e9,.001,.002,16)
        self.assertLess(np.linalg.norm(got-ref)/np.linalg.norm(ref),1e-7)

    def test_numpy_rmist_matches_accepted_torch_model(self):
        import torch
        from three_dried.imaging.rmist_kernel import RMISTKernel
        from three_dried.models.rmist_net import RMISTNet
        tau,theta=load_rmist_parameters(ROOT/'results/review/M6/summary_metrics.json')
        op=RMISTKernel(78.8e9,.001,.002,16);net=RMISTNet(op)
        with torch.no_grad():net.tau.copy_(torch.tensor(tau));net.theta.copy_(torch.tensor(theta))
        y=self.complex()*1e-4
        expected=net(torch.tensor(y,dtype=torch.complex128),self.op.range_m).detach().numpy()
        actual=rm_iteration(y,self.op,tau,theta)
        self.assertLess(np.linalg.norm(expected-actual)/np.linalg.norm(expected),1e-10)

    def test_cv_ista_objective_nonincreasing(self):
        mask=self.op.pad(nested_masks((12,13),[.5],6)[.5]);truth=self.complex()
        y=self.op.measured_forward(truth,mask)
        result,diag=cv_ista(y,self.op,mask,40,.01)
        objectives=np.array(diag['objective'])
        self.assertTrue(np.all(np.diff(objectives)<=max(objectives[0]*1e-12,1e-20)))
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(objectives[-1],objectives[0])

    def test_cv_ista_complex_phase_equivariance(self):
        mask=self.op.pad(nested_masks((12,13),[.7],8)[.7]);y=self.op.measured_forward(self.complex(),mask)
        x,_=cv_ista(y,self.op,mask,12,.01);rot=np.exp(.73j)
        xr,_=cv_ista(y*rot,self.op,mask,12,.01)
        self.assertLess(np.linalg.norm(xr-rot*x)/np.linalg.norm(x),1e-12)

    def test_fft_bp_matches_direct_geometric_sum(self):
        y=self.complex()*support(self.cfg);g=green_kernel(self.cfg,self.op.range_m)
        expected=np.zeros((16,16),complex);s=support(self.cfg)
        az,ax=np.nonzero(s)
        for iz in range(16):
            for ix in range(16):
                h=g[15+iz-az,15+ix-ax]
                expected[iz,ix]=np.sum(y[az,ax]*h.conj())/np.sum(np.abs(h)**2)
        got=backproject(y,self.cfg,self.op.range_m)
        self.assertLess(np.linalg.norm(got-expected)/np.linalg.norm(expected),1e-12)

    def test_bp_focal_response_for_independent_spherical_point(self):
        g=green_kernel(self.cfg,self.op.range_m);iz,ix=8,8
        az,ax=np.indices((16,16));y=g[15+az-iz,15+ax-ix]*support(self.cfg)
        actual=backproject(y,self.cfg,self.op.range_m)
        self.assertLess(abs(actual[iz,ix]-1),1e-12)

    def test_point_scene_uses_physical_aperture_and_exact_gt(self):
        echo,gt,projected,meta=point_scene(self.cfg)
        self.assertEqual(echo.shape,(1,12,13));self.assertEqual(gt.shape,(1,16,16))
        self.assertTrue(meta['inverse_crime']);self.assertFalse(meta['synthetic_raw_fmcw'])
        self.assertEqual(np.count_nonzero(gt),1)
        full=self.op.forward(gt[0]);zs,xs=support_slices(self.cfg)
        self.assertTrue(np.array_equal(echo[0],full[zs,xs]))
        self.assertLess(np.linalg.norm(projected[0]-self.op.inverse(full)),1e-12)

class RunnerTests(unittest.TestCase):
    def test_matrix_and_missing_real_data_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run'
            proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_baseline_suite.py'),
                '--smoke','--no-figures','--output-dir',str(out),'--dataset-dir',str(Path(temp)/'absent')],
                capture_output=True,text=True)
            self.assertEqual(proc.returncode,2,proc.stderr)
            summary=json.loads((out/'run_summary.json').read_text())
            self.assertEqual(summary['rows'],40)
            self.assertEqual(summary['datasets']['point_sim']['status'],'COMPLETED')
            self.assertEqual(summary['datasets']['real']['status'],'BLOCKED_MISSING_DATA')
            self.assertEqual(summary['status'],'INCOMPLETE')
            self.assertTrue((out/'summary_metrics.csv').is_file())
            proc2=subprocess.run([sys.executable,str(ROOT/'scripts/run_baseline_suite.py'),
                '--smoke','--no-figures','--output-dir',str(out)],capture_output=True,text=True)
            self.assertNotEqual(proc2.returncode,0)
            self.assertIn('not empty',proc2.stderr)


class RealLoaderIntegrationTests(unittest.TestCase):
    def test_all_methods_use_fixture_echo_and_never_report_real_gt_metrics(self):
        import h5py
        from three_dried.baseline.runner import run_suite
        cfg=small_config()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'capture';folder.mkdir();path=folder/'output_rawdata.mat'
            d=cfg['data'];frames=d['nx']*d['nz']+d['motion_error_samples']
            with h5py.File(path,'w') as handle:
                raw=handle.create_group('adcRawData');dim=raw.create_group('dim')
                for key,val in dict(numFrames=frames,numRxChan=4,numSamples=256).items():dim.create_dataset(key,data=[[val]])
                raw.create_group('rfParams').create_dataset('startFreq',data=[[77]])
                refs=raw.create_dataset('data',(frames,1),dtype=h5py.ref_dtype)
                group=handle.create_group('#refs#')
                t=np.arange(256);tone=np.exp(2j*np.pi*16*t/256).astype('complex64')
                sample=np.zeros((256,4,1),dtype='complex64');sample[:,0,0]=tone
                sample[:,1:,0]=100  # Verify that unused receivers cannot leak in.
                for i in range(frames):refs[i,0]=group.create_dataset(str(i),data=sample).ref
            before=path.read_bytes()
            summary=run_suite(cfg,ROOT,['point_sim','real'],Path(temp)/'out',folder,False)
            self.assertEqual(summary['status'],'COMPLETED');self.assertEqual(summary['rows'],80)
            self.assertEqual(before,path.read_bytes())
            import csv
            rows=list(csv.DictReader((Path(temp)/'out/summary_metrics.csv').open()))
            real=[r for r in rows if r['dataset']=='real']
            self.assertEqual(len(real),40)
            self.assertTrue(all(r['complex_nrmse_gt']=='' for r in real))
            self.assertTrue(all(r['finite']=='True' for r in real))


class MetricPrecisionTests(unittest.TestCase):
    def test_reference_cosine_stays_in_range_with_complex64_volume(self):
        from three_dried.baseline.metrics import image_metrics
        rng=np.random.default_rng(9)
        x=(rng.normal(size=(3,64,64))+1j*rng.normal(size=(3,64,64))).astype('complex64')
        got=image_metrics(x,reference=x.astype('complex128'))
        self.assertLessEqual(got['rma_reference_magnitude_cosine_proxy'],1.0)
        self.assertAlmostEqual(got['rma_reference_magnitude_cosine_proxy'],1.0,places=12)

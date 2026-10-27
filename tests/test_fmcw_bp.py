"""Declared FMCW IF/FFT phase, migrated sampling and CUDA sum contracts."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
import yaml

from three_dried.imaging.fmcw_bp import synthetic_if, range_profiles, reference_bp, FusedFMCWBP, C0
from three_dried.simulation.fmcw_adc_reference import direct_adc_match
from three_dried.baseline.runtime_environment import capture_runtime_environment
from test_baseline_suite import small_config

ROOT=Path(__file__).resolve().parents[1]
F0=77e9; S=70.295e12; FS=5e6; N=256; NF=2048; DELAY=6.2516e-10
FREF=F0+S*(N-1)/(2*FS)


class FMCWPhysicsTests(unittest.TestCase):
    def test_cpu_closed_endpoints_and_zero_outside(self):
        profile=np.array([[1+2j],[3-4j],[5+6j],[7-8j]])
        for query in (0.,3.,-0.25,3.25):
            with self.subTest(query=query):
                delay=query/4
                value=reference_bp(profile,[[0.,0.]],[[0.,0.]],0.,1.,1.,0.,4,delay)[0]
                expected=(profile[int(query),0]*np.exp(1j*np.pi*delay**2)
                          if 0<=query<=3 else 0j)
                np.testing.assert_allclose(value,expected,rtol=1e-13,atol=1e-13)

    def test_reference_rejects_mismatched_fft_length(self):
        with self.assertRaisesRegex(ValueError,'range length'):
            reference_bp(np.ones((8,1),complex),[[0.,0.]],[[0.,0.]],.6,FS,S,FREF,4)

    def test_environment_is_collected_fresh(self):
        a=capture_runtime_environment('cpu');b=capture_runtime_environment('cpu')
        self.assertNotEqual(a['capture_id'],b['capture_id'])
        self.assertEqual(b['environment_source'],'live_process_and_hardware')
        self.assertEqual(b['python_executable'],sys.executable)

    def test_offgrid_multidepth_adc_reference_and_interpolation_bound(self):
        aperture=np.array([(x*.001,z*.002) for x in (-80,0,80) for z in (-50,0,50)])
        rho=.7+.3j
        for depth in (.43,.61431,.83):
            point=(.00137,depth,-.00243)
            raw=synthetic_if(aperture,[(*point,rho)],F0,S,FS,N,DELAY)
            exact=direct_adc_match(raw,aperture,[point],FS,S,F0,DELAY)[0]
            np.testing.assert_allclose(exact,len(aperture)*rho,rtol=2e-12,atol=2e-12)
            for oversampling in (1,2,4,8):
                with self.subTest(depth=depth,oversampling=oversampling):
                    nfft=N*oversampling
                    profile=range_profiles(torch.tensor(raw),FS,S,nfft).numpy()
                    interpolated=reference_bp(profile,aperture,[[point[0],point[2]]],depth,FS,S,FREF,nfft,DELAY)[0]
                    # Linear interpolation curvature bound on the centered
                    # finite ADC sum, normalized by known signal amplitude.
                    bound=np.pi**2*(1-1/N**2)/(24*oversampling**2)
                    self.assertLess(abs(interpolated-exact)/(len(aperture)*abs(rho)),bound+1e-11)
                    self.assertLess(abs(np.angle(interpolated*np.conj(exact))),2e-11)

    def test_unified_entry_cannot_silently_substitute_narrowband_for_fmcw(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            acquisition=folder/'acquisition.yaml'
            acquisition.write_text(yaml.safe_dump(small_config()))
            settings=yaml.safe_load((ROOT/'configs/ista_net_baseline.yaml').read_text())
            settings['acquisition_config']=str(acquisition)
            settings['device']='cpu';settings['evaluation']['warmup']=0;settings['evaluation']['repetitions']=1
            config=folder/'settings.yaml';config.write_text(yaml.safe_dump(settings))
            for explicit in (False,True):
                output=folder/('diagnostic' if explicit else 'full')
                command=[sys.executable,str(ROOT/'scripts/run_imaging_baseline.py'),'--algorithm','BP',
                         '--dataset','point_sim','--rates','0.2','--config',str(config),
                         '--no-figures','--output',str(output)]
                if explicit:command+=['--bp-model','narrowband_diagnostic']
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                row=json.loads((output/'summary.json').read_text())['rows'][0]
                self.assertEqual(row['status'],'DIAGNOSTIC' if explicit else 'SKIPPED')
                if not explicit:self.assertIn('NOT_RAW_FMCW',row['reason'])

    def test_centered_fft_peak_retains_expected_carrier_and_rvp(self):
        tau=17*FS/(S*N)
        depth=(tau-DELAY)*C0/2
        rho=.7+.3j
        raw=synthetic_if([[0.,0.]],[(0.,depth,0.,rho)],F0,S,FS,N,DELAY)
        profile=range_profiles(torch.tensor(raw),FS,S,NF).numpy()
        peak=17*(NF//N)
        phi=2*np.pi*FREF*tau-np.pi*S*tau*tau
        self.assertLess(abs(profile[peak,0]*np.exp(-1j*phi)-rho),1e-11)

    def test_range_migration_follows_geometry(self):
        aperture=np.array([[0.,0.],[.2,.2]])
        raw=synthetic_if(aperture,[(0.,.6,0.,1.)],F0,S,FS,N,DELAY)
        profile=range_profiles(torch.tensor(raw),FS,S,NF).numpy()
        axis=C0/2*(np.arange(NF)*FS/(S*NF)-DELAY)
        peaks=axis[np.argmax(np.abs(profile),axis=0)]
        expected=np.sqrt(.6**2+(aperture**2).sum(axis=1))
        self.assertLess(np.max(np.abs(peaks-expected)),(axis[1]-axis[0])/2+1e-10)
        focused=reference_bp(profile,aperture,[[0.,0.]],.6,FS,S,FREF,NF,DELAY)[0]
        self.assertLess(abs(focused-2),.02)

    def test_reference_complex_linearity_without_gain_or_phase_alignment(self):
        aperture=np.array([[0.,0.],[.003,.004]])
        raw=synthetic_if(aperture,[(0.,.6,0.,1.)],F0,S,FS,N,DELAY)
        profile=range_profiles(torch.tensor(raw),FS,S,NF).numpy()
        args=(aperture,[[0.,0.],[.01,.02]],.6,FS,S,FREF,NF,DELAY)
        x=reference_bp(profile,*args)
        rotated=reference_bp(profile*(.4+.7j),*args)
        np.testing.assert_allclose(rotated,x*(.4+.7j),rtol=1e-12,atol=1e-12)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
class FMCWCUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aperture=np.array([(x*.001,z*.002) for z in range(-3,4) for x in range(-3,4)])
        cls.pixels=np.array([(x*.001,z*.002) for z in range(-4,5) for x in range(-4,5)])
        raw=synthetic_if(cls.aperture,[(0.,.6,0.,.7+.3j),(.003,.6,.004,.5-.2j)],F0,S,FS,N,DELAY)
        cls.profiles=range_profiles(torch.tensor(raw,device='cuda:0'),FS,S,NF)
        cls.operator=FusedFMCWBP(cls.aperture,cls.pixels,.6,FS,S,FREF,NF,DELAY,
                                ROOT/'results/fmcw_bp_build')

    def test_double_cuda_matches_numpy_full_complex_sum(self):
        expected=reference_bp(self.profiles.cpu().numpy(),self.aperture,self.pixels,.6,FS,S,FREF,NF,DELAY)
        actual=self.operator(self.profiles).cpu().numpy()
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-11)

    def test_all_modes_reject_different_fft_lengths(self):
        self.operator.prepare_lattice_cache(.001,.002)
        for length in (NF//2,NF+1):
            profile=torch.ones((length,len(self.aperture)),device='cuda:0',dtype=torch.complex128)
            for precision in ('double','mixed','cached_double'):
                with self.subTest(length=length,precision=precision):
                    with self.assertRaisesRegex(ValueError,'does not match operator n_fft'):
                        self.operator(profile,precision)

    def test_all_cuda_modes_include_both_endpoints_and_zero_outside(self):
        profile=torch.tensor([[1+2j],[3-4j],[5+6j],[7-8j]],device='cuda:0',dtype=torch.complex128)
        for query in (0.,3.,-0.25,3.25):
            operator=FusedFMCWBP([[0.,0.]],[[0.,0.]],0.,1.,1.,0.,4,query/4,
                                ROOT/'results/fmcw_bp_build')
            operator.prepare_lattice_cache(.001,.002)
            expected=(profile[int(query),0].item()*np.exp(1j*np.pi*(query/4)**2)
                      if 0<=query<=3 else 0j)
            for precision in ('double','mixed','cached_double'):
                with self.subTest(query=query,precision=precision):
                    value=operator(profile,precision).cpu().item()
                    np.testing.assert_allclose(value,expected,rtol=1e-7 if precision=='mixed' else 1e-13,
                                               atol=1e-7 if precision=='mixed' else 1e-13)

    @unittest.skipUnless(torch.cuda.device_count()>=2,'SKIPPED: two CUDA devices required')
    def test_launch_uses_operator_device_and_restores_other_current_device(self):
        for owner,other in ((0,1),(1,0)):
            device='cuda:%s'%owner
            profile=self.profiles.to(device)
            operator=FusedFMCWBP(self.aperture,self.pixels,.6,FS,S,FREF,NF,DELAY,
                                ROOT/'results/fmcw_bp_build',device)
            operator.prepare_lattice_cache(.001,.002)
            stream=torch.cuda.Stream(device=owner)
            with torch.cuda.device(owner):
                previous=torch.cuda.current_stream()
                stream.wait_stream(previous)
                torch.cuda.set_stream(stream)
            try:
                expected=reference_bp(profile.cpu().numpy(),self.aperture,self.pixels,.6,FS,S,FREF,NF,DELAY)
                with torch.cuda.device(other):
                    for precision in ('double','mixed','cached_double'):
                        with self.subTest(owner=owner,other=other,precision=precision):
                            actual=operator(profile,precision).cpu().numpy()
                            self.assertEqual(torch.cuda.current_device(),other)
                            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),
                                            1e-3 if precision=='mixed' else 1e-11)
            finally:
                with torch.cuda.device(owner):torch.cuda.set_stream(previous)

    def test_offgrid_multidepth_oversampling_cpu_cuda_and_adc_errors(self):
        aperture=np.array([(x*.001,z*.002) for x in (-80,0,80) for z in (-50,0,50)])
        pixels=np.array([(x*.001,z*.002) for x in (-20,0,20) for z in (-10,0,10)])
        points=[(.00137,.43,-.00243,.7+.3j),(-.02317,.61431,.01141,-.4+.8j),(.03723,.83,-.03183,.6-.5j)]
        raw=synthetic_if(aperture,points,F0,S,FS,N,DELAY)
        scale=len(aperture)*sum(abs(p[3]) for p in points)
        for depth in (.43,.61431,.83):
            queries=np.column_stack([pixels[:,0],np.full(len(pixels),depth),pixels[:,1]])
            direct=direct_adc_match(raw,aperture,queries,FS,S,F0,DELAY)
            for oversampling in (1,4,8):
                with self.subTest(depth=depth,oversampling=oversampling):
                    nfft=N*oversampling
                    profile=range_profiles(torch.tensor(raw,device='cuda:0'),FS,S,nfft)
                    cpu=reference_bp(profile.cpu().numpy(),aperture,pixels,depth,FS,S,FREF,nfft,DELAY)
                    bound=np.pi**2*(1-1/N**2)/(24*oversampling**2)
                    self.assertLess(np.max(np.abs(cpu-direct))/scale,bound+1e-11)
                    operator=FusedFMCWBP(aperture,pixels,depth,FS,S,FREF,nfft,DELAY,ROOT/'results/fmcw_bp_build')
                    operator.prepare_lattice_cache(.001,.002)
                    for precision in ('double','cached_double'):
                        cuda=operator(profile,precision).cpu().numpy()
                        self.assertLess(np.linalg.norm(cuda-cpu)/np.linalg.norm(cpu),1e-11)

    def test_mixed_precision_error_is_bounded(self):
        expected=self.operator(self.profiles).cpu().numpy()
        actual=self.operator(self.profiles,'mixed').cpu().numpy()
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-3)

    def test_regular_lattice_cache_preserves_migrated_double_sum(self):
        expected=self.operator(self.profiles).cpu().numpy()
        self.operator.prepare_lattice_cache(.001,.002)
        actual=self.operator(self.profiles,'cached_double').cpu().numpy()
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-11)

    def test_cache_rejects_incorrect_lattice(self):
        with self.assertRaises(ValueError):self.operator.prepare_lattice_cache(.0013,.002)

    def test_sparse_profile_and_geometry_selection_match_reference(self):
        chosen=np.arange(0,len(self.aperture),3)
        aperture=self.aperture[chosen]
        profiles=self.profiles[:,chosen].contiguous()
        operator=FusedFMCWBP(aperture,self.pixels,.6,FS,S,FREF,NF,DELAY,
                            ROOT/'results/fmcw_bp_build')
        actual=operator(profiles).cpu().numpy()
        expected=reference_bp(profiles.cpu().numpy(),aperture,self.pixels,.6,FS,S,FREF,NF,DELAY)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-11)


if __name__=='__main__':unittest.main()

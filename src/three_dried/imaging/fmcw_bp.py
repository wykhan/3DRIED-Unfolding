"""FMCW range-migration BP with a declared, tested positive-beat convention.

The canonical synthetic IF is rho*exp(j*(2*pi*(f0+S*t)*tau-pi*S*tau**2)),
tau=2*R/c+delay. f0 is explicitly the frequency at the first ADC sample.
The FFT profile is centered in fast time before interpolation; the remaining
phase uses f_ref=f0+S*(N-1)/(2*fs) and retains quadratic residual video phase.
This synthetic convention is not a claim to calibrate Hammer's mixer/ADC.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch

C0 = 299792458.0


def synthetic_if(aperture_xz, points_xyz_rho, f0_hz, slope_hz_s, fs_hz,
                 n_adc=256, delay_s=0.0):
    """Return [aperture, ADC] complex128, chunked to bound CPU memory."""
    aperture = np.asarray(aperture_xz, dtype=np.float64)
    time = np.arange(n_adc)/fs_hz
    echo = np.zeros((len(aperture), n_adc), np.complex128)
    for first in range(0,len(aperture),1024):
        a = aperture[first:first+1024]
        for x,y,z,rho in points_xyz_rho:
            distance = np.sqrt((a[:,0]-x)**2+y*y+(a[:,1]-z)**2)
            tau = 2*distance/C0+delay_s
            phase = 2*np.pi*(f0_hz+slope_hz_s*time[None,:])*tau[:,None]-np.pi*slope_hz_s*tau[:,None]**2
            echo[first:first+len(a)] += rho*np.exp(1j*phase)
    return echo


def range_profiles(raw, fs_hz, slope_hz_s, n_fft):
    """[aperture, ADC] -> centered [range, aperture] complex128.

    Backward FFT / N_ADC preserves a unit-scatterer amplitude. Zero padding
    refines interpolation, not physical range resolution. No legacy one-bin
    coordinate convention is changed: this is a separate physical IF path.
    """
    if raw.ndim != 2 or not torch.is_complex(raw) or n_fft < raw.shape[-1]:
        raise ValueError('complex [aperture,ADC] and n_fft >= ADC count required')
    n_adc = raw.shape[-1]
    fbeat = torch.arange(n_fft,device=raw.device,dtype=torch.float64)*fs_hz/n_fft
    t_center = (n_adc-1)/(2*fs_hz)
    phase = torch.exp(2j*torch.pi*fbeat*t_center)
    # A native zero-based FFT beat-frequency coordinate is intentional.
    return (torch.fft.fft(raw.to(torch.complex128),n=n_fft,dim=-1,norm='backward')
            *phase[None,:]/n_adc).T.contiguous()


def reference_bp(profiles, aperture_xz, pixels_xz, depth_m, fs_hz, slope_hz_s,
                 f_ref_hz, n_fft, delay_s=0.0):
    """Float64 interpolation reference, closed [0,n_fft-1], zero outside.

    The exact final sample is included using the final interval at weight
    one. No extrapolation, wraparound or tolerance-based clamping is used.
    """
    profiles = np.asarray(profiles)
    if profiles.ndim!=2 or profiles.shape[0]!=n_fft or n_fft<2:
        raise ValueError('profile range length must equal n_fft >= 2')
    aperture = np.asarray(aperture_xz,dtype=np.float64)
    result = np.zeros(len(pixels_xz),np.complex128)
    indices = np.arange(len(aperture))
    for i,(x,z) in enumerate(pixels_xz):
        distance = np.sqrt((x-aperture[:,0])**2+depth_m**2+(z-aperture[:,1])**2)
        tau = 2*distance/C0+delay_s
        query = slope_hz_s*tau*n_fft/fs_hz
        valid = np.isfinite(query)&(query>=0)&(query<=n_fft-1)
        lo = np.floor(np.where(valid,query,0)).astype(np.int64)
        lo = np.minimum(lo,n_fft-2)
        a = indices[valid];k = lo[valid];fraction = query[valid]-k
        sample = profiles[k,a]+fraction*(profiles[k+1,a]-profiles[k,a])
        phase = 2*np.pi*f_ref_hz*tau[valid]-np.pi*slope_hz_s*tau[valid]**2
        result[i] = np.sum(sample*np.exp(-1j*phase))
    return result


def compile_kernel(build_dir, device=0, nvcc='/usr/local/cuda/bin/nvcc'):
    """Compile a standalone CUDA library, independent of PyTorch CUDA ABI.

    The CUDA runtime uses the current PyTorch device stream/pointers; no
    additional Python dependency or system package installation is needed.
    """
    source = Path(__file__).with_name('fmcw_bp_kernel.cu')
    major,minor = torch.cuda.get_device_capability(device)
    digest = hashlib.sha256(source.read_bytes()+('%s%s'%(major,minor)).encode()).hexdigest()[:16]
    build_dir = Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
    library = build_dir/('fmcw_bp_'+digest+'.so')
    command = [nvcc,'-O3','-shared','-Xcompiler','-fPIC','-arch=sm_%s%s'%(major,minor),
               '-Xlinker','-rpath','-Xlinker',str(Path(nvcc).parents[1]/'lib64'),
               str(source),'-o',str(library)]
    if not library.exists():
        process = subprocess.run(command,capture_output=True,text=True)
        (build_dir/'compile.log').write_text(process.stdout+process.stderr)
        if process.returncode:
            raise RuntimeError('CUDA compilation failed; see '+str(build_dir/'compile.log'))
    return library,command


class FusedFMCWBP:
    """One fixed-depth output slice, full range interpolation at every term."""
    def __init__(self, aperture_xz, pixels_xz, depth_m, fs_hz, slope_hz_s,
                 f_ref_hz, n_fft, delay_s, build_dir, device='cuda:0'):
        self.device = torch.device(device)
        if self.device.type != 'cuda':raise ValueError('CUDA device required')
        if isinstance(n_fft,bool) or int(n_fft)!=n_fft or n_fft<2:
            raise ValueError('n_fft must be an integer >= 2')
        self.n_fft=int(n_fft)
        with torch.cuda.device(self.device):
            self.device=torch.device('cuda',torch.cuda.current_device())
            path,self.compile_command = compile_kernel(build_dir,self.device)
        self.library = ctypes.CDLL(str(path))
        self.library.launch_fmcw_bp.argtypes = ([ctypes.c_void_p]*6+[ctypes.c_int]*3+
            [ctypes.c_double]*5+[ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_void_p])
        self.library.launch_fmcw_bp.restype = ctypes.c_int
        self.library.fmcw_bp_cuda_error.argtypes = [ctypes.c_int]
        self.library.fmcw_bp_cuda_error.restype = ctypes.c_char_p
        aperture = torch.as_tensor(aperture_xz,dtype=torch.float64,device=self.device)
        pixels = torch.as_tensor(pixels_xz,dtype=torch.float64,device=self.device)
        self.ax = aperture[:,0].contiguous();self.az = aperture[:,1].contiguous()
        self.px = pixels[:,0].contiguous();self.pz = pixels[:,1].contiguous()
        self.parameters = (float(depth_m),float(delay_s),float(fs_hz),float(slope_hz_s),
                           float(f_ref_hz),int(n_fft),C0)

    def __call__(self, profiles, precision='double'):
        if precision not in {'double','mixed','cached_double'}:raise ValueError('unknown precision')
        if profiles.device != self.device or profiles.dtype != torch.complex128 or not profiles.is_contiguous():
            raise ValueError('contiguous complex128 range profiles on the selected device required')
        if profiles.ndim!=2 or profiles.shape[1]!=len(self.ax):raise ValueError('aperture shape mismatch')
        if profiles.shape[0]!=self.n_fft:
            raise ValueError('profile range length %s does not match operator n_fft=%s' %
                             (profiles.shape[0],self.n_fft))
        # The CUDA runtime launch and stream belong to the operator's device,
        # even when another CUDA device is current on entry. Restore it on exit.
        with torch.cuda.device(self.device):
            output = torch.empty(len(self.px),device=self.device,dtype=torch.complex128)
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if precision=='cached_double':
                if not hasattr(self,'lookup'):raise ValueError('prepare_lattice_cache must be called first')
                code=self.library.launch_lattice_bp(profiles.data_ptr(),self.aperture_indices.data_ptr(),
                    self.pixel_indices.data_ptr(),self.lookup.data_ptr(),output.data_ptr(),
                    len(self.ax),profiles.shape[0],len(self.px),self.lookup.shape[1],*self.lookup_shift,stream)
            else:
                code = self.library.launch_fmcw_bp(profiles.data_ptr(),self.ax.data_ptr(),self.az.data_ptr(),
                    self.px.data_ptr(),self.pz.data_ptr(),output.data_ptr(),len(self.ax),profiles.shape[0],len(self.px),
                    *self.parameters,int(precision=='mixed'),stream)
            if code:raise RuntimeError(self.library.fmcw_bp_cuda_error(code).decode())
        return output

    def prepare_lattice_cache(self,dx_m,dz_m):
        """Cache exact offset-dependent phase/indices on a verified lattice.

        This is an FP64 lookup of the same range-interpolating formula, not
        a single-frequency FFT convolution or a nearest-range approximation.
        Rebuild for each depth, radar definition or aperture/image lattice.
        """
        aperture=np.column_stack([self.ax.cpu().numpy(),self.az.cpu().numpy()])
        pixels=np.column_stack([self.px.cpu().numpy(),self.pz.cpu().numpy()])
        origin=pixels.min(axis=0);spacing=np.array([dx_m,dz_m],float)
        if np.any(~np.isfinite(spacing)) or np.any(spacing<=0):raise ValueError('positive spacings required')
        indices=[]
        for coords in (aperture,pixels):
            value=(coords-origin)/spacing
            if np.max(np.abs(value-np.rint(value)))>1e-7:
                raise ValueError('coordinates do not share the requested regular lattice')
            indices.append(np.rint(value).astype(np.int32))
        a,p=indices
        half=np.maximum(np.abs(p.max(axis=0)-a.min(axis=0)),np.abs(a.max(axis=0)-p.min(axis=0)))
        dx=np.arange(-half[0],half[0]+1)*dx_m
        dz=np.arange(-half[1],half[1]+1)*dz_m
        depth,delay,fs,slope,fref,nfft,c0=self.parameters
        distance=np.sqrt(dx[None,:]**2+dz[:,None]**2+depth**2)
        tau=2*distance/c0+delay;query=slope*tau*nfft/fs
        valid=np.isfinite(query)&(query>=0)&(query<=nfft-1)
        lo=np.minimum(np.floor(np.where(valid,query,0)),nfft-2)
        fraction=np.where(valid,query-lo,0)
        lo=np.where(valid,lo,-1)
        phase=2*np.pi*fref*tau-np.pi*slope*tau*tau
        table=np.stack([lo,fraction,np.cos(phase),np.sin(phase)],axis=-1)
        self.lookup=torch.as_tensor(table,device=self.device)
        self.aperture_indices=torch.as_tensor(a,device=self.device)
        self.pixel_indices=torch.as_tensor(p,device=self.device)
        self.lookup_shift=tuple(map(int,half))
        self.library.launch_lattice_bp.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*6+[ctypes.c_void_p]
        self.library.launch_lattice_bp.restype=ctypes.c_int
        return self.lookup.numel()*self.lookup.element_size()

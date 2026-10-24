"""NumPy RM pair plus a separate masked Hilbert adjoint.

The accepted M2 linspace / fftshift convention is intentional. Dagger and
adjoint are distinct operations; the existing PyTorch kernels are unchanged.
"""
import numpy as np
from .config import C0, support_slices

class SliceRM:
    def __init__(self,cfg,range_m):
        self.n=cfg['imaging']['n_fft_space']; self.range_m=float(range_m)
        k=2*np.pi*cfg['radar']['center_frequency_hz']/C0
        a=cfg['aperture']
        kx=np.linspace(-np.pi/a['dx_m'],np.pi/a['dx_m'],self.n)
        kz=np.linspace(-np.pi/a['dz_m'],np.pi/a['dz_m'],self.n)[:,None]
        rad=(2*k)**2-kx[None,:]**2-kz**2
        ky=np.sqrt(np.maximum(rad,0)); valid=(rad>=0)&(ky>1e-12)
        self.valid=np.fft.fftshift(valid)
        safe=np.where(valid,ky,1)
        self.phi=np.fft.fftshift(np.where(valid,np.exp(1j*ky*self.range_m)/safe,0))
        self.phi_dagger=np.fft.fftshift(np.where(valid,ky*np.exp(-1j*ky*self.range_m),0))
        self.physical_slices=support_slices(cfg)

    def _check(self,x):
        x=np.asarray(x)
        if x.shape[-2:]!=(self.n,self.n) or not np.iscomplexobj(x):
            raise ValueError('complex [...,n,n] input required')
        return x

    def _apply(self,x,multiplier):
        return np.fft.ifft2(np.fft.fft2(self._check(x),axes=(-2,-1))*multiplier,axes=(-2,-1))

    def forward(self,x):
        return self._apply(x,self.phi)

    def inverse(self,y):
        return self._apply(y,self.phi_dagger)

    def adjoint(self,y):
        return self._apply(y,self.phi.conj())

    def measured_forward(self,x,mask):
        self._mask_check(mask)
        return mask*self.forward(x)

    def measured_adjoint(self,y,mask):
        self._mask_check(mask)
        return self.adjoint(mask*y)

    def _mask_check(self,mask):
        if mask.shape!=(self.n,self.n) or not np.all((mask==0)|(mask==1)):
            raise ValueError('binary mask must match the padded domain')

    def pad(self,x):
        zs,xs=self.physical_slices
        if x.shape[-2:]!=(zs.stop-zs.start,xs.stop-xs.start):
            raise ValueError('unexpected physical aperture shape')
        out=np.zeros(x.shape[:-2]+(self.n,self.n),dtype=x.dtype)
        out[...,zs,xs]=x
        return out

    def lipschitz_bound(self):
        # ||M|| <= 1; the FFT convolution is unitarily diagonalizable.
        return float(np.max(np.abs(self.phi)**2))

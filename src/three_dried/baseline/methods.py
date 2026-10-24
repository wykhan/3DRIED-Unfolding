"""Five named baselines; no training or parameter searches on real data."""
import json
import numpy as np
from scipy.signal import fftconvolve
from .config import C0, support

METHODS=('BP','RMA','ISTA','CV-ISTA','RMIST')

def shrink(x,threshold):
    if threshold<0 or not np.isfinite(threshold):
        raise ValueError('finite nonnegative threshold required')
    mag=np.abs(x)
    return x*(np.maximum(mag-threshold,0)/(mag+1e-12))

def load_rmist_parameters(path):
    doc=json.loads(path.read_text())
    if doc.get('best_epoch')!=99:
        raise ValueError('expected accepted epoch-99 RMIST parameters')
    params=doc['best_parameters']
    tau=np.array(params['tau'],dtype=np.float32).astype(np.float64)
    theta=np.array(params['theta'],dtype=np.float32).astype(np.float64)
    if tau.shape!=(9,) or theta.shape!=(9,) or not np.all(np.isfinite(tau)) or not np.all(np.isfinite(theta)) or np.any(tau<0) or np.any(theta<0):
        raise ValueError('invalid accepted 18 RMIST scalars')
    return tau,theta

def rm_iteration(y,op,tau,theta):
    """Literal accepted RMIST update. Missing echoes stay zero-filled.

    This deliberately preserves RM(estimate), without reapplying the mask
    in the residual. Do not describe this as a masked-data least-squares
    solver. It is the historical RM-kernel reproduction control.
    """
    x=op.inverse(y)
    for t,h in zip(tau,theta):
        x=shrink(x+t*op.inverse(y-op.forward(x)),h)
    return x

def cv_ista(y,op,mask,iterations=100,lambda_relative=0.01):
    """Minimize .5||M RM(x)-y||^2 + lambda sum(abs(x)).

    CV-ISTA here is classical complex-valued ISTA, not a learned network or
    a claimed reproduction of an unspecified CV-ISTA paper. Lambda is a
    fixed fraction of ||A^H y||_infinity; tau uses a safe spectral bound.
    """
    if iterations<0 or int(iterations)!=iterations or lambda_relative<0 or not np.isfinite(lambda_relative):
        raise ValueError('invalid CV-ISTA parameters')
    op._mask_check(mask)
    if np.any(y[~mask.astype(bool)]):
        raise ValueError('measurement must be zero outside the sampling mask')
    step=1/max(op.lipschitz_bound(),np.finfo(float).tiny)
    lam=lambda_relative*float(np.max(np.abs(op.measured_adjoint(y,mask))))
    x=np.zeros_like(y,dtype=np.complex128)
    history=[]
    for _ in range(iterations):
        residual=op.measured_forward(x,mask)-y
        x=shrink(x-step*op.measured_adjoint(residual,mask),step*lam)
        residual=op.measured_forward(x,mask)-y
        history.append(float(.5*np.linalg.norm(residual)**2+lam*np.abs(x).sum()))
    return x,dict(step=step,lambda_value=lam,objective=history,iterations=iterations,
        termination='fixed_iteration_budget',optimality_certificate=False)

def green_kernel(cfg,range_m):
    """Discrete scalar Weyl Green kernel, at centered relative offsets.

    g=dx*dz/(2*pi*j)*exp(j*2*k*R)/R. Uses a slice-wise narrowband
    scalar model compatible with RM's 1/ky approximation, not full FMCW BP.
    """
    n=cfg['imaging']['n_fft_space']; a=cfg['aperture']
    dz=np.arange(-(n-1),n)*a['dz_m']; dx=np.arange(-(n-1),n)*a['dx_m']
    distance=np.sqrt(float(range_m)**2+dz[:,None]**2+dx[None,:]**2)
    k=2*np.pi*cfg['radar']['center_frequency_hz']/C0
    return a['dx_m']*a['dz_m']/(2j*np.pi)*np.exp(2j*k*distance)/distance

def backproject(y,cfg,range_m):
    """Spherical-wave matched-filter BP on a regular planar aperture.

    FFT linear convolution accelerates the exact discrete spatial sum; no
    circular wrap or RMA phase kernel is used. Full-aperture sensitivity
    normalization is held fixed across sampling rates, preserving SR loss.
    """
    g=green_kernel(cfg,range_m)
    image=fftconvolve(y,g.conj(),mode='same')
    sensitivity=fftconvolve(support(cfg).astype(float),np.abs(g)**2,mode='same').real
    return image/np.maximum(sensitivity,np.finfo(float).tiny)

def reconstruct(method,y,op,mask,cfg,rmist_params):
    if method=='RMA': return op.inverse(y),{}
    if method=='BP': return backproject(y,cfg,op.range_m),{}
    if method=='ISTA':
        c=cfg['algorithms']['ista']; it=int(c['iterations'])
        return rm_iteration(y,op,[c['tau']]*it,[c['theta']]*it),{}
    if method=='RMIST': return rm_iteration(y,op,*rmist_params),{}
    if method=='CV-ISTA':
        c=cfg['algorithms']['cv_ista']
        return cv_ista(y,op,mask,c['iterations'],c['lambda_relative'])
    raise ValueError('unknown baseline: '+method)

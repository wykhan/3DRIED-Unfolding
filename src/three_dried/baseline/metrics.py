"""Explicit domains and ground-truth policies for each reported metric."""
import numpy as np

def relative_error(x,ref):
    denominator=np.linalg.norm(ref)
    if denominator==0: return None
    return float(np.linalg.norm(x-ref)/denominator)

def image_metrics(volume,gt=None,projected_gt=None,reference=None):
    mag=np.abs(np.asarray(volume,dtype=np.complex128)); total=np.sum(mag**2)
    p=mag.ravel()**2/max(float(total),np.finfo(float).tiny)
    nonzero=p>0
    out=dict(finite=bool(np.isfinite(volume).all()),max_magnitude=float(mag.max()),
        energy=float(total),entropy=float(-np.sum(p[nonzero]*np.log(p[nonzero]))))
    if gt is not None:
        out['complex_nrmse_gt']=relative_error(volume,gt)
        out['magnitude_nrmse_gt']=relative_error(mag,np.abs(gt))
        out['complex_nrmse_bandlimited_gt']=relative_error(volume,projected_gt)
    # A full-data RMA is a proxy reference, never ground truth.
    if reference is not None:
        rmag=np.abs(np.asarray(reference,dtype=np.complex128)); norm=np.linalg.norm(mag)*np.linalg.norm(rmag)
        out['rma_reference_magnitude_cosine_proxy']=float(np.clip(np.sum(mag*rmag)/norm,0.0,1.0)) if norm else None
        out['rma_reference_complex_relative_distance_proxy']=relative_error(volume,reference)
    return out

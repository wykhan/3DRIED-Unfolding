"""Nested, exact-count masks on the acquired aperture, never on padding."""
import hashlib
import numpy as np

def nested_masks(shape, rates, seed=20261024, pattern='random_2d'):
    nz,nx=shape
    if nz<1 or nx<1 or pattern not in {'random_2d','height_lines'}:
        raise ValueError('invalid physical shape or pattern')
    rng=np.random.default_rng(seed)
    size=nz*nx if pattern=='random_2d' else nz
    order=rng.permutation(size)
    masks={}
    for rate in rates:
        if not np.isfinite(rate) or not 0<rate<=1:
            raise ValueError('rate must be in (0,1]')
        # Round half up, including configurations whose rate*size is not integral.
        count=int(np.floor(size*rate+0.5))
        flat=np.zeros(size,dtype=bool); flat[order[:count]]=True
        masks[float(rate)]=(flat.reshape(nz,nx) if pattern=='random_2d'
                            else np.broadcast_to(flat[:,None],(nz,nx)).copy())
    return masks

def mask_record(mask,rate,pattern,seed):
    return dict(pattern=pattern,seed=seed,requested_rate=float(rate),
                observed_positions=int(mask.sum()),physical_positions=int(mask.size),
                actual_rate=float(mask.mean()),
                observed_height_rows=int(np.any(mask,axis=1).sum()),
                sha256_uint8=hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest(),
                shared_across_fast_time_and_ranges=True)

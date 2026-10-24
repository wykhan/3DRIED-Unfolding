"""GT-bearing RM point scenes and read-only single-Rx real measurements."""
from pathlib import Path
import hashlib
import numpy as np
from .config import coordinates, ranges_m, support_slices, configuration_fingerprint
from .operators import SliceRM

def point_scene(cfg):
    """Oracle point volume -> RM -> acquired physical aperture.

    This is an explicitly model-matched (inverse-crime) solver regression.
    It does not claim sim-to-real performance or full FMCW waveform realism.
    """
    n=cfg['imaging']['n_fft_space']; bins=cfg['imaging']['matlab_range_bins']
    x,z=coordinates(cfg); ys=ranges_m(cfg)
    gt=np.zeros((len(bins),n,n),dtype=np.complex128); placed=[]
    for p in cfg['simulation']['points']:
        if p['matlab_bin'] not in bins:
            raise ValueError('point bin is outside the reconstruction bins')
        if not x[0]<=p['x_m']<=x[-1] or not z[0]<=p['z_m']<=z[-1]:
            raise ValueError('point is outside the image field of view')
        ix=int(np.argmin(np.abs(x-p['x_m']))); iz=int(np.argmin(np.abs(z-p['z_m'])))
        ib=bins.index(p['matlab_bin'])
        gt[ib,iz,ix]+=p['amplitude']*np.exp(1j*p['phase_rad'])
        placed.append(dict(p,grid_x_m=float(x[ix]),grid_z_m=float(z[iz]),
                           range_m=float(ys[ib]),index_yzx=[ib,iz,ix]))
    if not np.any(gt): raise ValueError('point scene must be nonempty')
    shape=(len(bins),cfg['data']['nz'],cfg['data']['nx'])
    echo=np.empty(shape,dtype=np.complex128); projected=np.empty_like(gt)
    zs,xs=support_slices(cfg)
    for i,y in enumerate(ys):
        op=SliceRM(cfg,y); full=op.forward(gt[i]); echo[i]=full[zs,xs]
        projected[i]=op.inverse(full)
    snr=cfg['simulation']['snr_db']
    if snr is not None:
        if not np.isfinite(snr): raise ValueError('SNR must be finite or null')
        rng=np.random.default_rng(cfg['seed'])
        # Noise referenced to RMS across the entire recorded range/aperture cube.
        power=np.mean(np.abs(echo)**2)*10**(-snr/10)
        echo+=(rng.normal(size=shape)+1j*rng.normal(size=shape))*np.sqrt(power/2)
    return echo,gt,projected,dict(kind='point_sim',model='rm_range_slices',
        inverse_crime=True,synthetic_raw_fmcw=False,points=placed,snr_db=snr,
        shape=list(echo.shape),configuration_fingerprint=configuration_fingerprint(cfg))

def real_echo(cfg,dataset_dir=None):
    from three_dried.data.three_dried import load_single_rx_aperture,inspect_raw_mat
    d=cfg['data']; path=Path(dataset_dir or d['dataset_dir'])/d['raw_filename']
    if not path.is_file(): raise FileNotFoundError('3DRIED raw MAT unavailable: '+str(path))
    audit=inspect_raw_mat(path)
    if int(audit['dimensions']['numSamples'])!=d['adc_samples'] or int(audit['dimensions']['numRxChan'])!=d['available_rx']:
        raise ValueError('raw metadata conflicts with the shared acquisition configuration')
    stat_before=path.stat()
    raw,meta=load_single_rx_aperture(path,rx_index=d['rx_index'],nx=d['nx'],nz=d['nz'],
        motion_error_samples=d['motion_error_samples'],serpentine_correct=d['serpentine_correct'])
    if raw.shape!=(d['nz'],d['nx'],d['adc_samples']): raise ValueError('unexpected raw shape')
    spectrum=np.fft.fft(raw,n=cfg['imaging']['n_range_fft'],axis=-1)
    bins=np.array(cfg['imaging']['matlab_range_bins'])-1
    echo=np.moveaxis(spectrum[:,:,bins],-1,0)
    stat_after=path.stat()
    if (stat_before.st_size,stat_before.st_mtime_ns)!=(stat_after.st_size,stat_after.st_mtime_ns):
        raise RuntimeError('source MAT changed during the read-only load')
    meta.update(kind='real',source=str(path),voxel_ground_truth=False,raw_audit=audit,
        source_size=stat_after.st_size,source_mtime_ns=stat_after.st_mtime_ns,
        shape=list(echo.shape),configuration_fingerprint=configuration_fingerprint(cfg))
    return echo,None,None,meta

def echo_sha256(echo):
    return hashlib.sha256(np.ascontiguousarray(echo).view(np.uint8)).hexdigest()

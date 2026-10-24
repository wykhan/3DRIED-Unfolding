"""Single source of truth for acquisition and reconstruction parameters."""
from pathlib import Path
import hashlib
import json
import numpy as np
import yaml

C0 = 299792458.0

def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get('schema_version') != 1:
        raise ValueError('unsupported baseline config schema')
    cfg['_config_path'] = str(path)
    validate_config(cfg)
    return cfg

def validate_config(cfg):
    d, r, a, im = (cfg[k] for k in ['data','radar','aperture','imaging'])
    if d['rx_index'] != 0:
        raise ValueError('accepted baseline uses Rx1 / Python index 0 only')
    if any(int(d[k]) != d[k] or d[k] < 1 for k in ['nx','nz','adc_samples']):
        raise ValueError('positive integral acquisition sizes required')
    n = im['n_fft_space']
    if n < max(d['nx'], d['nz']) or int(n) != n:
        raise ValueError('spatial FFT must cover the physical aperture')
    if im['n_range_fft'] != d['adc_samples']:
        raise ValueError('reference baseline requires ADC length = range FFT length')
    bins = im['matlab_range_bins']
    if not bins or len(set(bins)) != len(bins) or any(int(b)!=b or b<1 or b>im['n_range_fft'] for b in bins):
        raise ValueError('unique one-based MATLAB bins inside the FFT required')
    if any(not np.isfinite(r[k]) or r[k] <= 0 for k in ['center_frequency_hz','sample_rate_hz','chirp_slope_hz_s']):
        raise ValueError('positive finite radar parameters required')
    if any(not np.isfinite(a[k]) or a[k] <= 0 for k in ['dx_m','dz_m']):
        raise ValueError('positive finite aperture spacings required')
    if any(not 0 < s <= 1 for s in cfg['sampling']['rates']):
        raise ValueError('sampling rates must be in (0,1]')
    if not set(cfg['sampling']['patterns']) <= {'random_2d','height_lines'}:
        raise ValueError('unknown sampling pattern')
    if r['speed_of_light_m_s'] != C0:
        raise ValueError('reference speed of light must match the accepted operator')
    derived = [(r['adc_window_s'], d['adc_samples']/r['sample_rate_hz']),
               (r['sampled_bandwidth_hz'], r['chirp_slope_hz_s']*d['adc_samples']/r['sample_rate_hz']),
               (a['horizontal_span_m'], (d['nx']-1)*a['dx_m']),
               (a['vertical_span_m'], (d['nz']-1)*a['dz_m'])]
    if any(not np.isclose(value, expected, rtol=1e-10, atol=1e-14) for value,expected in derived):
        raise ValueError('derived acquisition parameters conflict with the explicit configuration')
    if im['range_convention'] != 'accepted_matlab_one_based':
        raise ValueError('unsupported range convention; do not silently change the accepted geometry')
    if cfg['simulation']['model'] != 'rm_range_slices':
        raise ValueError('unsupported simulation model')
    if im['representative_matlab_bin'] not in bins:
        raise ValueError('representative slice must be in the reconstructed bins')
    return cfg

def ranges_m(cfg):
    r, im = cfg['radar'], cfg['imaging']
    # Intentional, accepted legacy convention: use b, rather than b-1.
    return C0/2*(np.array(im['matlab_range_bins'])*r['sample_rate_hz'] /
                 (r['chirp_slope_hz_s']*im['n_range_fft'])-r['instrument_delay_s'])

def coordinates(cfg):
    n=cfg['imaging']['n_fft_space']; a=cfg['aperture']; im=cfg['imaging']
    i=np.arange(n)-(n-1)/2
    return (i*a['dx_m']+im['x_offset_mm']/1000,
            i*a['dz_m']+im['z_offset_mm']/1000)

def support_slices(cfg):
    n=cfg['imaging']['n_fft_space']; d=cfg['data']
    z=(n-d['nz'])//2; x=(n-d['nx'])//2
    return slice(z,z+d['nz']),slice(x,x+d['nx'])

def support(cfg):
    n=cfg['imaging']['n_fft_space']; out=np.zeros((n,n),dtype=bool)
    out[support_slices(cfg)]=True
    return out

def configuration_fingerprint(cfg):
    value={k:cfg[k] for k in ['data','radar','aperture','imaging']}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

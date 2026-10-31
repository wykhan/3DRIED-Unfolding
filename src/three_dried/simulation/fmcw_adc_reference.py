"""Independent raw-ADC matched sum for the declared positive-beat model.

No FFT, range-profile interpolation, centered-frequency phase helper, image
gain fit or phase alignment is used. 1/N_ADC is the fixed normalization in
the BP contract. This reference describes simulation, not Hammer calibration.
"""
import numpy as np

C0=299792458.0


def direct_adc_match(raw, aperture_xz, points_xyz, fs_hz, slope_hz_s,
                     first_adc_frequency_hz, delay_s=0.0):
    raw=np.asarray(raw,dtype=np.complex128)
    aperture=np.asarray(aperture_xz,dtype=np.float64)
    antenna=np.column_stack((aperture[:,0],np.zeros(len(aperture)),aperture[:,1]))
    sample_times=np.arange(raw.shape[1],dtype=np.float64)/fs_hz
    result=[]
    for point in np.asarray(points_xyz,dtype=np.float64):
        delay=2*np.linalg.norm(antenna-point,axis=1)/C0+delay_s
        # Direct conjugate of the expected delayed/dechirped ADC waveform.
        carrier=np.exp(-2j*np.pi*first_adc_frequency_hz*delay)
        chirp=np.exp(-2j*np.pi*slope_hz_s*delay[:,None]*sample_times[None,:])
        residual=np.exp(1j*np.pi*slope_hz_s*delay**2)
        result.append(np.sum(raw*chirp*(carrier*residual)[:,None])/raw.shape[1])
    return np.asarray(result,dtype=np.complex128)

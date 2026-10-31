"""Complex error, point localization, conditional sidelobes and timed calls."""
import time
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import torch


def complex_nmse(image, target):
    energy=float(np.abs(target).astype('float64').ravel().dot(np.abs(target).astype('float64').ravel()))
    return float(np.sum(np.abs(image-target).astype('float64')**2)/energy) if energy else None


def point_localization(image, target, dx, dz):
    truth=np.argwhere(np.abs(target)>0)
    if not len(truth): return {}
    magnitude=np.abs(image)
    maxima=(magnitude==maximum_filter(magnitude,size=(1,3,3))) & (magnitude>0)
    candidates=np.argwhere(maxima)
    if len(candidates)<len(truth):
        return {'localization_status':'FAIL_INSUFFICIENT_PEAKS'}
    order=np.argsort(magnitude[tuple(candidates.T)])[::-1][:len(truth)]
    peaks=candidates[order]
    # Cross-range assignment is evaluated within the known synthetic range bin.
    costs=np.sqrt(((truth[:,None,1]-peaks[None,:,1])*dz)**2+((truth[:,None,2]-peaks[None,:,2])*dx)**2)
    costs+=np.where(truth[:,None,0]==peaks[None,:,0],0,1e3)
    a,b=linear_sum_assignment(costs)
    return {'localization_mean_m':float(costs[a,b].mean()),'localization_max_m':float(costs[a,b].max()),
            'localization_status':'MEASURED_TOP_K_PEAKS','point_count':len(truth),
            'resolution_note':'top-K peak assignment; no claim of a universal resolution limit'}


def sidelobes_single_point(image, target):
    indices=np.argwhere(np.abs(target)>0)
    if len(indices)!=1:
        return {'pslr_islr_status':'SKIPPED_MULTIPLE_TARGETS'}
    ib,iz,ix=indices[0]
    peak=np.unravel_index(np.argmax(np.abs(image[ib])),image[ib].shape)
    result={}
    for name,profile in [('x',np.abs(image[ib,peak[0],:])),('z',np.abs(image[ib,:,peak[1]]))]:
        at=int(np.argmax(profile));mins=find_peaks(-profile)[0]
        left=mins[mins<at];right=mins[mins>at]
        if not len(left) or not len(right):
            result[name+'_sidelobe_status']='SKIPPED_MAINLOBE_NOT_BOUNDED';continue
        lo,hi=int(left[-1]),int(right[0])
        power=profile.astype('float64')**2
        side=np.r_[power[:lo],power[hi+1:]];main=power[lo:hi+1]
        if not len(side) or not power[at] or not main.sum():
            result[name+'_sidelobe_status']='SKIPPED_EMPTY_REGION';continue
        result[name+'_pslr_db']=float(10*np.log10(max(float(side.max()),1e-300)/power[at]))
        result[name+'_islr_db']=float(10*np.log10(max(float(side.sum()),1e-300)/main.sum()))
        result[name+'_mainlobe_bounds']=[lo,hi]
        result[name+'_sidelobe_status']='MEASURED_NEAREST_LOCAL_MINIMA'
    return result


def timed_reconstruction(call, device, warmup=2, repetitions=5):
    cuda=str(device).startswith('cuda')
    for _ in range(warmup): call()
    seconds=[];result=None
    for _ in range(repetitions):
        if cuda: torch.cuda.synchronize(torch.device(device))
        started=time.perf_counter();result=call()
        if cuda: torch.cuda.synchronize(torch.device(device))
        seconds.append(time.perf_counter()-started)
    return result, {'runtime_median_seconds':float(np.median(seconds)), 'runtime_repetitions':repetitions,
                    'runtime_warmup':warmup,'runtime_device':str(device),'runtime_preprocessing_included':False,
                    'runtime_samples_seconds':seconds}

"""Literal phase-only backprojection; no FFT convolution or fast BP.

Every candidate voxel and every retained aperture location is evaluated in
Python. A voxel budget can interrupt a diagnostic; an interrupted volume is
never returned or reported as a reconstruction. No adaptive weights or
sensitivity normalization are applied.
"""
import cmath
import math
import time
import numpy as np

C0 = 299792458.


def direct_slice_bp(echo, aperture_x, aperture_z, image_x, image_z, depth,
                    center_frequency_hz, mask, max_seconds=None):
    """Narrowband control on supplied fixed-depth echo, explicit direct sum."""
    observations = [(float(aperture_x[ix]), float(aperture_z[iz]), complex(echo[iz, ix]))
                    for iz in range(len(aperture_z)) for ix in range(len(aperture_x)) if mask[iz, ix]]
    image = np.zeros((len(image_z), len(image_x)), dtype=np.complex128)
    phase_rate = 4 * math.pi * center_frequency_hz / C0
    started = time.monotonic()
    for iz, z in enumerate(image_z):
        for ix, x in enumerate(image_x):
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                raise TimeoutError('literal BP exceeded diagnostic budget; no valid full volume')
            value = 0j
            for ax, az, sample in observations:
                distance = math.sqrt((x - ax)**2 + depth**2 + (z - az)**2)
                value += sample * cmath.exp(-1j * phase_rate * distance)
            image[iz, ix] = value
    return image


def direct_range_bp(range_echo, range_axis_m, aperture_x, aperture_z,
                    image_x, image_y, image_z, phase_per_m, mask, max_seconds=None):
    """Geometric range-profile BP with basic complex linear interpolation.

Range profile phase convention and phase_per_m MUST be verified by a
physical point control before this path is used on Hammer. No reference
one-bin range adjustment or chirp residual phase is guessed here.
"""
    observations = [(iz, ix) for iz in range(len(aperture_z)) for ix in range(len(aperture_x)) if mask[iz, ix]]
    out = np.zeros((len(image_y), len(image_z), len(image_x)), dtype=np.complex128)
    started = time.monotonic()
    for iy, depth in enumerate(image_y):
        for iz, z in enumerate(image_z):
            for ix, x in enumerate(image_x):
                if max_seconds is not None and time.monotonic() - started > max_seconds:
                    raise TimeoutError('literal BP exceeded budget; partial volume is invalid')
                value = 0j
                for az, ax in observations:
                    distance = math.sqrt((x - aperture_x[ax])**2 + depth**2 + (z - aperture_z[az])**2)
                    if distance < range_axis_m[0] or distance > range_axis_m[-1]:
                        continue
                    profile = range_echo[:, az, ax]
                    sample = complex(np.interp(distance, range_axis_m, profile.real), np.interp(distance, range_axis_m, profile.imag))
                    value += sample * cmath.exp(-1j * phase_per_m * distance)
                out[iy, iz, ix] = value
    return out

# M2 Conventional RMA Hammer Report

Date: 2026-09-26 (Asia/Shanghai)  
Status: **PASS — STOP FOR HUMAN REVIEW**

## Result

The configured single-Rx conventional RMA completed without NaN or Inf and
produced a complex 3-D volume of shape `[range=13, z=512, x=512]`. The
500 mm visualization ROI contains a clearly recognizable claw hammer: a
broad, asymmetric metal head connected to a long straight diagonal handle.
The peak lies at 614.310 mm. All voxels above -12 dB are in that range plane;
the -18 dB point cloud spans 572.662--655.959 mm. This is physically
consistent with the supplied 3-D reference, whose plotted target occupies
approximately 500--700 mm in range.

## Receiver and raw layout

MATLAB Rx1 (Python index 0) is the only receiver used. The other three
channels are neither loaded into the image cube nor averaged, fused, or
calibrated. The choice reproduces the bundled `Demo_RMA2D_Imaging.m`, which
selects `Raw_echo(1:4:end,:)`.

The HDF5 cell array has 82,278 frames. Each logical MATLAB frame is complex
single with shape `[chirp=1, Rx=4, ADC=256]` (stored by HDF5 as
`[256,4,1]`). Single-Rx selection followed by the reference motion correction
produces `[z=200, x=407, fast-time=256]` complex64 samples.

## Preprocessing and imaging

The exact pipeline is:

1. read MATLAB v7.3/HDF5 in read-only mode;
2. extract Rx1 only;
3. reproduce the reference 43-frame cumulative scan-motion correction;
4. reverse every second horizontal row to undo serpentine scanning;
5. apply a 256-point complex FFT over ADC fast time;
6. select MATLAB FFT bins 12--24;
7. for each bin, pad the 200×407 aperture to 512×512;
8. apply the reference plane-wave phase factor
   `ky * exp(-j*y*ky)`, where `ky=sqrt((2k)^2-kx^2-kz^2)` and evanescent
   components are zeroed;
9. inverse 2-D FFT and stack the complex slices;
10. derive normalized-dB slices, an x-z MIP, and an -18 dB 3-D point cloud.

To reproduce MATLAB exactly, FFTs use NumPy's default convention rather than
orthonormal scaling: forward transforms are unnormalized and inverse
transforms include `1/N`. Phase is preserved until visualization.

## Parameters and coordinates

- start frequency in MAT metadata: 77 GHz;
- reference center frequency: 78.8 GHz;
- chirp slope: 70.295 THz/s;
- ADC sampling rate: 5 MHz;
- instrument delay: 0.62516 ns;
- aperture spacing: 1 mm in horizontal x, 2 mm in vertical z;
- spatial FFT: 512×512;
- coordinate convention: x horizontal, y range, z vertical;
- reconstructed y range: 0.406069--0.905848 m (MATLAB bins 12--24);
- full x grid: -305.5--205.5 mm;
- full z grid: -391--631 mm;
- plotted ROI: x=-249.5--205.5 mm, z=-249--249 mm;
- peak: x=42.5 mm, y=614.310 mm, z=141 mm.

The x and z offsets (-50 mm and +120 mm) are taken directly from the bundled
RMA reference. The full 512-point grid is retained in `volume.npy`; figures
apply the configured 500 mm ROI to suppress periodic wraparound replicas.

## Qualitative comparison and orientation

The reconstruction and supplied 2-D/3-D JPEGs agree on the diagnostic
features required for identification: a compact high-reflectivity claw/head,
a narrow elongated handle, a diagonal pose, and strongest depth near
0.61 m. The physical scene photograph also shows the same head/handle
proportions and diagonal placement. The recovered diagonal is therefore
physically plausible.

There is one coordinate-handedness uncertainty: in the configured reference
coordinate labels, the recovered handle slopes down-left from the head,
whereas the supplied 2-D JPEG slopes down-right. No horizontal flip was
silently introduced because the bundled MATLAB code fixes the current scan
ordering and coordinate offsets. This is a display/absolute-aperture-sign
ambiguity, not a recognition or focusing failure, and should be checked at
the human checkpoint before the coordinate convention is used by M3.

## Reproducibility and validation

Final command:

```bash
/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rma.py --config configs/hammer.yaml
```

Two consecutive full runs produced the identical `volume.npy` SHA-256:
`257bc8c26a38af67ba4b2e5c10242930335031201cb16e0d10cecab4c6c4a40a`.
Full runs took approximately 14--15 seconds on CPU. Unit tests pass (3/3), NaN count is
zero, and Inf count is zero. Recomputed source hashes match the pre-run audit;
the raw MAT hash remains
`167c5977d75deddac782710c287d5bd49cffefe04a67e0c88fa45ab8e30a7475`.

Required outputs are in `results/rma_hammer/`, including `volume.npy`,
`mip_xz.png`, `representative_slices/`, `volume_3d.png`, `run.log`,
`config_used.yaml`, `environment.txt`, and `metrics_or_stats.json`.

## Remaining uncertainties

- Absolute horizontal handedness is not independently calibrated, as noted
  above; changing it requires human review because the coordinate convention
  is now established.
- Range-bin coordinates intentionally reproduce the bundled MATLAB formula,
  which uses the one-based MATLAB bin number directly. A conventional
  zero-based FFT-frequency formula would be offset by one bin and is not used.
- The RMA is a per-range-bin narrowband phase-compensation reconstruction;
  it does not perform wideband Stolt interpolation. This is the supplied
  reference baseline, not a new physical-model choice.

M2 meets its documented acceptance criteria. M3 and later milestones remain
locked pending human review.

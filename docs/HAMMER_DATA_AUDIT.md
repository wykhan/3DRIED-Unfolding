# M1 Hammer Dataset Audit

Date: 2026-09-26 (Asia/Shanghai)  
Status: **PASS**

The audit is reproducible with:

```bash
python3 scripts/inspect_hammer.py --hash-files --output results/hammer_audit.json
```

## Files and formats

The target directory contains one MATLAB v7.3/HDF5 file and three JPEGs:

- `output_rawdata.mat` (676,300,265 bytes), raw complex radar capture;
- `Near-Field 2-D MMW  Image.jpg` (875×656), reference x-z image;
- `Near-Field 3-D MMW  Image.jpg` (875×656), reference 3-D rendering;
- `target (5).jpg` (4000×3000), scene photograph.

The complete hashes and image metadata are in `results/hammer_audit.json`.

## MAT structure and axis interpretation

The root contains `adcRawData` and MATLAB's internal `#refs#` group.
`adcRawData.data` is a cell array containing 82,278 unique frames. MATLAB
sees each frame as `[1, 4, 256]` = `[chirp, Rx, ADC fast-time]`; HDF5 stores
the transposed representation `[256, 4, 1]`. Samples are compound
float32 real/imaginary values and are loaded as `complex64`; phase is present
and retained.

The bundled `Demo_RMA2D_Imaging.m` establishes the remaining acquisition
layout. After its motion correction, the selected data are
`[vertical z=200, horizontal x=407, fast-time=256]`. Horizontal spacing is
1 mm and vertical spacing is 2 mm. The scan is serpentine, so every second
vertical row is reversed before imaging. The reference removes a cumulative
43-frame horizontal motion error using
`floor(matlab_vertical_index / 200 * 43)`.

## Metadata and processing parameters

The file records four Rx channels, one chirp/frame, 256 samples, start
frequency 77 GHz, sample rate 5 MHz, chirp slope 70.294998 MHz/µs,
bandwidth 3.599104 GHz, and nominal range resolution 0.0416770 m. The bundled
RMA code uses center frequency 78.8 GHz and an instrument-delay correction of
`6.2516e-10 s`. These values, rather than inferred parameters, define M2.

## Single-Rx decision

Only MATLAB Rx 1 / Python index 0 is selected. This is not an arbitrary
preference: the official nearby preprocessing and RMA scripts use
`Raw_echo(1:4:end,:)`, explicitly selecting the first of four receivers. No
other Rx is loaded into the reconstruction array, and no fusion, averaging,
or joint calibration is performed.

## References and conclusion

Reference processing exists at
`/home/superws/dataset/3DRIED/Demo_datapreprocessing.m` and
`/home/superws/dataset/3DRIED/Demo_RMA2D_Imaging.m`. Reference images are
included beside the raw capture. The raw-to-range-processing layout and all
parameters required for the reference RMA are therefore unambiguous. M1
passes and no stop condition is triggered.


# M9 Scale-Normalized RMIST Report

## Result

M9 completed the four predetermined inference-only normalization comparisons.
The milestone execution passed, but the scientific outcome is:

```text
NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION
```

All four rules restored strong learned phase dynamics. None satisfied the
predeclared real-data proxy-benefit criteria. This is evidence against using
these fixed normalizations as a Hammer reconstruction improvement, not a
ground-truth image-quality claim.

## Fixed protocol and provenance

The run used Rx1, MATLAB bins 12--24, the accepted M6 mask (seed `20260928`,
SHA-256 `43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2`),
and the accepted epoch-99 M6 parameters. The local checkpoint SHA-256 was
`4b1925f2a6d361a848a81b7e70a2dedc2dc3e520726a29b2ac92a6de999c26ce`.
The loaded `tau` and `theta` tensors were frozen and byte-for-byte unchanged
after inference. No optimizer, backward pass, training, fine-tuning, scale
search, or parameter update occurred.

The four rules and their materiality thresholds were fixed before the final
comparison. Scale factors were recorded before normalized inference and were
never clipped. Every scale was finite, positive, and inside the broad
`[1e-14, 1e-4]` numerical sanity interval.

## Synthetic references and regressions

The accepted 32-sample M6 validation control regenerated:

| Statistic | Median reference |
|---|---:|
| alpha0 RMS | 0.01376872018135817 |
| alpha0 p95 | 0.02026866101659833 |
| observed-measurement RMS | 8.869680634409644e-06 |

The alpha0 RMS matched the accepted M8 value exactly at recorded precision.
N0 matched the locally retained accepted M7 complex volume with relative
difference `0.0`; the `c=1` execution also matched ordinary RMIST.

## Scale factors and phase dynamics

| Method | Scale range (min--max) | Median final E_RMA | Dynamics restored |
|---|---:|---:|---|
| N0 raw | 1 | 1.1320e-06 | No |
| N1 global alpha0 RMS | 2.8565e-09 (one scale) | 0.8243 | Yes |
| N2 per-slice alpha0 RMS | 4.3283e-10--4.4081e-09 | 0.8261 | Yes |
| N3 per-slice alpha0 p95 | 4.2207e-10--3.6096e-09 | 0.9117 | Yes |
| N4 per-slice measurement RMS | 3.1775e-10--2.9091e-09 | 0.9161 | Yes |

All recorded phase metrics were finite. The result confirms the M8 mechanism:
matching amplitude scale makes the absolute thresholds active. Stronger
activity alone did not imply a better reconstruction.

## Real-data proxy evidence

M2 full-data RMA is used only as a reference proxy, never as ground truth.
The following reports the mean of the 13 per-slice magnitude comparisons:

| Method | Magnitude cosine | Normalized magnitude distance | Mean observed consistency |
|---|---:|---:|---:|
| N0 raw | 0.8913 | 0.4656 | 0.0850 |
| N1 global | 0.6881 | 0.7537 | 0.7348 |
| N2 per-slice RMS | 0.6778 | 0.7830 | 0.8460 |
| N3 per-slice p95 | 0.5888 | 0.8828 | 0.8993 |
| N4 measurement RMS | 0.5357 | 0.9382 | 0.9191 |

Every normalized variant worsened the mean per-slice cosine and distance.
The same direction holds for full-volume comparisons. N1 alone improved the
x-z MIP cosine (`0.9496` versus `0.9349` raw), but failed the per-slice and
full-volume criteria and was about 8.6 times worse in mean observed-sample
consistency. N2--N4 were worse still. No method passed the combined proxy gate.

## Range and visual assessment

All methods retained dominant bin 17. Range-profile cosine to M2 was
`0.99991` for raw RMIST, then `0.96585`, `0.95041`, `0.92465`, and `0.91592`
for N1--N4. Thus every normalized rule exceeded the predeclared material
degradation allowance of `0.02`. Per-slice variants increasingly suppressed
off-peak bins; they did not merely align phase activity.

On the shared ROI and one common dB peak, the head and handle remain
recognizable, so there is no complete target destruction. The normalized
methods strongly suppress background and substantial slice content. That
appearance cannot override their worse per-slice/full-volume proxy agreement,
range-profile preservation, and observed-sample consistency.

N1 is the least degraded normalization among the four and better preserves
inter-slice structure than the per-slice variants. It is not proxy-beneficial
and is not recommended as a production normalization.

## Limitations and conclusion

Hammer has no voxel ground truth, so no true NRMSE, PSNR, TBR, or universal
image-quality conclusion is possible. M2 agreement and observed consistency
are limited proxies and may favor conventional reconstruction structure.
Only one target, one accepted mask realization, and four fixed statistics
were evaluated. These limitations do not reverse the stated M9 gate result:
the evaluated rules restore dynamics but degrade all required combined
real-data proxy evidence. M9 stops here for independent review; no adaptive
normalizer or retraining is authorized.

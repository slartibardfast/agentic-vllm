# Oracle run record (2026-08-27)

Dated measurement artifact for the reference Turing W4A16 backend (both
kernels, float64 reference, max absolute error, tiled and naive). This file
is the frozen record of the run; the live summary lives in
reference-backend.md. Excluded from the naming audit as a dated artifact.

```
M16  N64  K64  G64  sym : 0.0038  / 0.0048
M64  N128 K256 G64  sym : 0.0078  / 0.0109
M1   N64  K128 G32  zp  : 0.0074  / 0.0074
M7   N192 K384 G128 zp  : 0.0156  / 0.0176
M100 N256 K256 G64  sym : 0.0078  / 0.0127
M64  N512 K512 G128 zp  : 0.0156  / 0.0257
M33  N128 K768 G32  sym : 0.0156  / 0.0203
```

Case parameters, in row order: N, K, and group size as printed; M 16, 64,
1, 7, 100, 64, 33; alternating symmetric and zero-point forms; activations
standard normal, weights uniform 4-bit, scales uniform in 0.05 to 0.15.
Tolerance: 5e-3 times the reference's largest magnitude. Raw harness:
`turing_lab/reference/oracle.py` at fork branch sm75-marlin pin 164539ad.

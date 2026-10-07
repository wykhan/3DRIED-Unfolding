# Research Portfolio

## Stream A — Learned sensing + learned reconstruction

Goal: develop a publishable SCI study that jointly optimizes sparse aperture observation selection and learned reconstruction on 3DRIED.

First-paper scope:
- learn 2-D aperture sampling positions under a fixed measurement budget;
- use the accepted 3DRIED physics/operator stack;
- compare against random/uniform/fixed sparse masks;
- evaluate simulation and real 3DRIED data;
- avoid simultaneously learning frequency, Rx selection, and range-bin selection in the first paper.

Core mathematical form:

[
\min_{\phi,\theta}\;\mathbb E\,\mathcal L(R_\theta(M_\phi\odot y),x)
\quad\text{s.t.}\quad \|M_\phi\|_0=K.
]

## Stream B — Learning-imaging baseline/library

Goal: build a stable research SDK with reproducible implementations and unified evaluation for:
- 2-D BP;
- 2-D RMA / holography;
- ISTA;
- FISTA / KFISTA;
- RMIST-Net;
- LFIST-Net;
- additional Wang-series methods only after literature/code inventory confirms relevance.

Required outputs:
- common data/operator/mask API;
- simulation examples;
- 3DRIED experiments;
- common metrics, runtime, memory, visualization;
- reproducible configs and tests;
- clear method provenance and paper alignment.

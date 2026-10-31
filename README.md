# The self-aligning bundle: simulations and figures

Code for the figures of *The self-aligning bundle: how matter with slow, coarse-grained dynamics comes to infer
and learn* (`minimal-fep.md`). The single unit is simulated as a finite-state, continuous-time jump process whose
rates obey local detailed balance; **no inference or learning rule is written into the rates**. The networks use the
same units with synchronous renewal and the local rule derived in the paper. Each notebook describes its
implementation in words in markdown cells and serves as the appendix for its figure.

## Layout

| Folder | Content |
|---|---|
| `network/` | library code used by the notebooks |
| `notebooks/` | one notebook per figure; each computes (or loads cached) results, saves every plot as a separate PDF and SVG, and composes the figure |
| `fig/` | `fig1_theory`, `fig2_single_unit`, `fig3_networks` as editable SVG composites with PDF/PNG renderings; `fig/panels/` holds one PDF and one SVG per plot |
| `data/` | cached simulation results (`.npz`) and summary numbers (`.json`); delete a file to recompute it |

### `network/`
- `family.py`: the bounded exponential family on a grid (log-partition, mean map, variance), the Langevin function and the continuous Bernoulli density.
- `unit.py`: exact generators of a single unit. `GatedUnit` includes microstates, encounter windows, a boundary, presentations and a slow structure; `EncounterUnit` is its limit of complete equilibration within windows. Also stationary laws, the average log-evidence ℓ, the averaged structural chain (the learning law) and spectra.
- `sim.py`: Monte Carlo sample paths of a single unit.
- `renewal.py`: networks of conditional units with synchronous renewal and the lagged rule $s_j^+(\mu_i^+-L(h_i))$, with the two controls (preceding-boundary pairing, reset state); undriven runs; attractor readout with sign pairs counted once; orthogonality and gain statistics.
- `digits.py`: the ten training digits of the FEP-ANN digits task.
- `artwork_fig1.py`: the Figure 1 artwork (matplotlib, mm coordinates).
- `plotting.py`: style, `save_panel` (one PDF and one SVG with editable text per plot) and `compose` (SVG composites, rendered to PDF/PNG with cairosvg).

### Notebooks
| Notebook | Figure | Run time |
|---|---|---|
| `01_theory.ipynb` | Fig. 1: jump process and observables, bundle, centred tangent, Legendre pair | seconds |
| `02_single_unit.ipynb` | Fig. 2: timescale separation, continuous Bernoulli, perception, learning | ~15 min from scratch; seconds from cache |
| `03_networks.ipynb` | Fig. 3: sequence learning and replay, self-orthogonalization with controls | ~10 min from scratch on CPU; seconds from cache |

Run the notebooks from inside `notebooks/`, because paths are relative to it. Panels keep their native size in mm,
so the composites can be rearranged in Inkscape or Illustrator. Text stays text: the fonts are Arial or Helvetica,
falling back to DejaVu Sans.

## Installation
```
pip install -r requirements.txt
```

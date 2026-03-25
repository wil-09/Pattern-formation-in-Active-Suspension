# Pattern-formation-in-Active-Suspension

Numerical simulations of pattern formation in active suspensions.

## Scripts

- **GR2D.py** – Computes and plots the 2D growth rate dispersion surface and contour for a linearised active-suspension model.
- **defects.py** – Time-steps the full nonlinear PDE system on a periodic 2D domain and tracks topological defects in the director field.

## Requirements

Python 3.10+ with [NumPy](https://numpy.org/) ≥ 1.26 and [Matplotlib](https://matplotlib.org/) ≥ 3.8.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the code

```bash
# Growth-rate analysis (fast, produces two plots)
python GR2D.py

# Full nonlinear simulation with live plotting (long-running)
python defects.py
```

A [GitHub Actions workflow](.github/workflows/run.yml) runs `GR2D.py` and validates `defects.py` automatically on every push and pull request.
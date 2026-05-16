# Liquid State Machine for Polar-Curve Learning

This project now includes a complete **equation-generated training pipeline** for testing whether an LSM can learn and emulate special polar curves and polar conics.

Implemented families:
- limacon
- cardioid
- rose curves
- circles (multiple polar forms)
- lemniscates
- conics in polar form (`r = l / (1 ± e cos θ)` or `r = l / (1 ± e sin θ)`)

The pipeline supports three tasks at once:
1. family classification
2. parameter regression
3. curve reconstruction (`r(θ)`)

## Requirements

- Python 3.x
- NumPy

Install:

```bash
pip install numpy
```

## Run

```bash
python3 simple_lsm.py
```

By default, the script will:
1. Generate synthetic datasets for `train`, `val`, `iid_test`, `interp_test`, and `ood_test`
2. Train a non-spiking baseline (raw sequence + ridge readouts)
3. Train an LSM feature extractor with ridge readout heads
4. Print comparable metrics tables for both approaches

## Useful options

```bash
# Faster smoke run
python3 simple_lsm.py --train-samples 200 --val-samples 60 --test-samples 80

# Skip baseline or skip LSM
python3 simple_lsm.py --skip-baseline
python3 simple_lsm.py --skip-lsm

# Export generated dataset
python3 simple_lsm.py --save-dataset polar_lsm_dataset.npz

# Tune reservoir size/connectivity
python3 simple_lsm.py --lsm-exc 80 --lsm-inh 40 --lsm-connectivity 0.12
```

## Metrics reported

- `class_acc`: family classification accuracy
- `param_mae`: MAE over the full shared parameter vector
- `active_mae`: MAE only on active (family-relevant) parameters
- `radial_rmse`: reconstruction RMSE on normalized `r(θ)`
- `geo_rmse`: Cartesian reconstruction RMSE

## Notes on split design

- `iid_test`: same distribution as training
- `interp_test`: held-out parameter bands inside training ranges
- `ood_test`: outside-range parameters (e.g., larger scales, higher rose `n`, conic `e > 1`)

## License

This project is proprietary software. All rights reserved for Brian Ting.
Please see the [LICENSE](LICENSE) file for more details.

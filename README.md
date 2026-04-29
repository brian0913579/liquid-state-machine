# Simple Liquid State Machine (LSM)

This project contains a simple implementation of a Liquid State Machine (LSM) using Python and NumPy. It simulates a neural network with a mixture of excitatory and inhibitory neurons using the Leaky Integrate-and-Fire (LIF) model.

## Features
- **Heterogeneous Neurons**: Includes 10 excitatory and 20 inhibitory neurons.
- **Dale's Principle**: Excitatory neurons only have positive synaptic weights, and inhibitory neurons only have negative synaptic weights.
- **Leaky Integrate-and-Fire**: Simulates basic neural spiking dynamics with configurable leak rates and thresholds.
- **Dynamic Synapses**: Demonstrates how to dynamically update synaptic connections, such as pruning weak connections.

## Prerequisites
- Python 3.x
- NumPy (`pip install numpy`)

## Usage

1. Open your terminal.
2. Navigate to the project directory.
3. Run the Python script:

```bash
python3 simple_lsm.py
```

### What to Expect
When you run the script, it will:
1. Initialize the LSM with 30 neurons.
2. Output the initial number of active synaptic connections.
3. Simulate the network for 5 time steps, feeding random input currents and showing how many neurons spike at each step.
4. Call `update_connections()` to prune weak synapses, and print the updated number of active connections.

## Customization
You can customize the LSM behavior directly in `simple_lsm.py`:
- Change the number of neurons by modifying `num_excitatory` and `num_inhibitory` when initializing `LiquidStateMachine`.
- Modify the `update_connections` function to implement different learning rules or pruning thresholds (currently it removes connections with weights below a certain threshold).

## License

This project is proprietary software. All rights reserved for Brian Ting.
Please see the [LICENSE](LICENSE) file for more details.

# Experiments

Expriments with different activations on the simple neural network (2 hidden nodes, trained on 3 data points), written before the cleaned-up version in the main folder. Gradients have not been double-checked, so treat this folder as evidence of exploration into neural networks.

## Activation functions
- `ReLU.py`: Uses the ReLU activation function, which fits a piecewise-linear curve to the three data points.
- `LeakyReLU.py`: Uses the Leaky ReLU activation function (adds a negative slope of 0.1). Also a piecewise-linear curve.
- `ELU.py`: Uses the ELU activation function (a smooth curve that levels off at -alpha).
- `AlphaELU.py`: Uses the ELU activation function with a trainable alpha. The alpha gradient is incomplete.

## Weight initialization
- `xavier_uniform_init.py`: Creates intial weights using Xavier (Glorot) uniform with a ReLU gain (√2).
- `xavier_normal_init.py`:  Creates intial weights using Xavier (Glorot) normal with a ReLU gain.

The ReLU variants used Xavier-generated starting weights.
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configuration
OBSERVED_DOSAGE = [0, 0.5, 1]
OBSERVED_EFFICACY = [0, 1, 0]
OBSERVED_DOSAGE_INDICES = list(range(len(OBSERVED_DOSAGE)))
CONVERGENCE_THRESHOLD = 0.0001
LEARNING_RATE = 0.1
MAX_TRIALS = 12000

# Initialize neural network parameters
class NeuralNetwork:
    def __init__(self):
        self.weight_1 = 2.74
        self.weight_2 = -1.13
        self.weight_3 = 0.36
        self.weight_4 = 0.63
        self.bias_1 = 0
        self.bias_2 = 0
        self.bias_3 = 0

        # Step sizes for tracking
        self.step_sizes = {
            'weight_1': 0, 'weight_2': 0, 'weight_3': 0, 'weight_4': 0,
            'bias_1': 0, 'bias_2': 0, 'bias_3': 0
        }

    def softplus_activation(self, x):
        """Softplus activation with numerical stability."""
        return math.log(1 + math.exp(min(x, 500)))  # Prevent overflow

    def softplus_derivative(self, x):
        """Derivative of softplus activation."""
        exp_x = math.exp(min(x, 500))  # Prevent overflow
        return exp_x / (1 + exp_x)

    def forward_pass(self, dosage_input):
        """Complete forward pass through the network."""
        # First hidden node
        x1 = dosage_input * self.weight_1 + self.bias_1
        y1 = self.softplus_activation(x1)

        # Second hidden node
        x2 = dosage_input * self.weight_2 + self.bias_2
        y2 = self.softplus_activation(x2)

        # Output
        predicted_efficacy = y1 * self.weight_3 + y2 * self.weight_4 + self.bias_3

        return predicted_efficacy, (x1, y1, x2, y2)

    def compute_derivatives(self):
        """Compute all gradients in a single pass."""
        derivatives = {
            'weight_1': 0, 'weight_2': 0, 'weight_3': 0, 'weight_4': 0,
            'bias_1': 0, 'bias_2': 0, 'bias_3': 0
        }

        for i in OBSERVED_DOSAGE_INDICES:
            dosage = OBSERVED_DOSAGE[i]
            predicted, (x1, y1, x2, y2) = self.forward_pass(dosage)
            residual = OBSERVED_EFFICACY[i] - predicted
            base_factor = -2 * residual

            # Compute derivatives
            softplus_deriv_1 = self.softplus_derivative(x1)
            softplus_deriv_2 = self.softplus_derivative(x2)

            derivatives['weight_1'] += base_factor * self.weight_3 * softplus_deriv_1 * dosage
            derivatives['weight_2'] += base_factor * self.weight_4 * softplus_deriv_2 * dosage
            derivatives['weight_3'] += base_factor * y1
            derivatives['weight_4'] += base_factor * y2
            derivatives['bias_1'] += base_factor * self.weight_3 * softplus_deriv_1
            derivatives['bias_2'] += base_factor * self.weight_4 * softplus_deriv_2
            derivatives['bias_3'] += base_factor

        return derivatives

    def update_parameters(self, derivatives, learning_rate):
        """Update all parameters and compute step sizes."""
        parameters = ['weight_1', 'weight_2', 'weight_3', 'weight_4',
                     'bias_1', 'bias_2', 'bias_3']

        for param in parameters:
            step_size = derivatives[param] * learning_rate
            self.step_sizes[param] = abs(step_size)
            current_value = getattr(self, param)
            setattr(self, param, current_value - step_size)

    def get_max_step_size(self):
        """Get maximum step size for convergence check."""
        return max(self.step_sizes.values())

    def generate_prediction_curve(self, num_points=100):
        """Generate x,y points for plotting the current model."""
        x_vals = [i / num_points for i in range(1, num_points + 1)]
        y_vals = [self.forward_pass(x)[0] for x in x_vals]
        return x_vals, y_vals

    def get_status_string(self, trial):
        """Generate status string for display."""
        weight_status = f"weight_1: {self.weight_1:.4f}\nweight_2: {self.weight_2:.4f}\nweight_3: {self.weight_3:.4f}\nweight_4: {self.weight_4:.4f}"
        bias_status = f"bias_1: {self.bias_1:.4f}\nbias_2: {self.bias_2:.4f}\nbias_3: {self.bias_3:.4f}"
        step_status = "\n".join([f"step_{k}: {v:.6f}" for k, v in self.step_sizes.items()])
        return f"\nWeights:\n{weight_status}\n\nBias:\n{bias_status}\n\nSteps:\n{step_status}"

def train_neural_network():
    """Main training loop with optimizations."""
    # Initialize
    fig, ax = plt.subplots()
    frames = []
    nn = NeuralNetwork()

    print("Starting training...")

    for trial in range(1, MAX_TRIALS + 1):
        # Compute gradients
        derivatives = nn.compute_derivatives()

        # Update parameters
        nn.update_parameters(derivatives, LEARNING_RATE)

        # Generate animation frame (every 10th iteration to reduce memory)
        if trial % 10 == 0 or trial == 1:
            x_values, y_values = nn.generate_prediction_curve()

            container, = ax.plot(x_values, y_values, linewidth=1, color='green')
            text_1 = ax.text(0.7, 0.1, nn.get_status_string(trial), fontsize=8, color='red')
            text_2 = ax.text(0, 1, f"Steps: {trial}", fontsize=12, color='purple')
            frames.append([container, text_1, text_2])

        # Print progress every 1000 iterations
        if trial % 1000 == 0:
            print(f"Trial {trial}: Max step size = {nn.get_max_step_size():.6f}")

        # Check convergence
        if nn.get_max_step_size() < CONVERGENCE_THRESHOLD:
            print(f"Converged at trial {trial}")
            break

    # Final results
    print(f"\nFinal parameters:")
    print(f"Weights: {nn.weight_1:.4f}, {nn.weight_2:.4f}, {nn.weight_3:.4f}, {nn.weight_4:.4f}")
    print(f"Biases: {nn.bias_1:.4f}, {nn.bias_2:.4f}, {nn.bias_3:.4f}")

    # Create animation
    ani = animation.ArtistAnimation(fig, frames, interval=50, repeat=False)
    ax.scatter(OBSERVED_DOSAGE, OBSERVED_EFFICACY, s=100, color='red', zorder=5)
    ax.text(0, 1.5, f"Learning Rate: {LEARNING_RATE}", fontsize=12, color="black")
    ax.set_xlabel("Dosage", fontsize=14)
    ax.set_ylabel("Effectiveness", fontsize=14)
    ax.set_title("Neural Network Training Progress")

    plt.tight_layout()
    plt.show()

    return nn

# Run the training
if __name__ == "__main__":
    trained_network = train_neural_network()
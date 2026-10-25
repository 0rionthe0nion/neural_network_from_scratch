# Neural Network from Scratch

## Overview
This project implements a basic neural network based on StatQuest with Josh Starmer's "Backpropagation Details Pt. 2: Going Bonkers with the Chain Rule" on Youtube. 

No machine learning libraries were used in the building of this model. The final implementation was refined for clarity and structure with assistance from AI tools.

## Running the Project
Download `neural_network_training_and_visualization.py`, change hyperparameters under `# Configuration` as needed, change parameters under `# Initialize neural network parameters` as needed, and run the file.

## Tech Used
- Python
- Matplotlib

## Features
- 2 hidden nodes (4 weights, 3 biases) trained on 3 data points
- Customizable initial weights and biases
- Forward pass and backpropagation
- Softplus activation function
- Gradient descent optimization using loss minimization
- Visualization of training process (parameter updates and step tracking)

## Visualization of Training
<img width="1440" height="754" alt="image" src="https://github.com/user-attachments/assets/963f711d-fca6-4548-bfca-b194cf6cb94c" />
<img width="1439" height="786" alt="image" src="https://github.com/user-attachments/assets/0c53aded-8b5e-4ad4-b086-9cac7391e9d5" />
<img width="1440" height="752" alt="image" src="https://github.com/user-attachments/assets/16edc3b5-8b3e-412e-a9a4-7f975c53a403" />

## What I learned
- How a simple neural network trains on a mathematical level without relying on deep learning frameworks
- Backpropagation and gradient-based optimization
- Debugging training issues

## Development Notes
- Explored ReLU, Leaky ReLU, and ELU (trainable alpha) activations, plus Xavier weight initialization, in early exploratory versions. These are in [`experiments/`](experiments/) and were not cleaned up or verified.

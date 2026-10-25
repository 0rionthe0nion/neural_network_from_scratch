import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
frames, x_values, y_values = [], [], []

observed_x = [0, 0.5, 1]
observed_y = [0, 1, 0]
indexs = [x for x in range(0, len(observed_x))]

step_limit = 1500
limit = 0.000001
learning_rate = 0.05

# Starting values
w_1 = -0.57
w_2 = 0.92
w_3 = -0.3
w_4 = 0.37
b_1 = 0
b_2 = 0
b_3 = 0

a = 0.25

def ELU(x):
    # ELU equation that takes in x and returns y.
    if x >= 0:
        return x
    else:
        return a * (math.e**x - 1)
    
def d_ELU(x):
    # Returns the derivative of a ELU equation.
    if x >= 0:
        return 1
    else:
        return a * math.e**x


class ELUNode:
    """Simulates a node that uses softplus as its activation function."""

    def __init__(self, weight_1, bias_1, weight_2):
        """Initializes the variables used by the node."""
        self.w_1 = weight_1
        self.b_1 = bias_1
        self.w_2 = weight_2

        self.x = 0
        self.y = 0
    
    def run(self, input):
        """Runs one pass through the activation function and returns result."""
        self.x = input * self.w_1 + self.b_1
        self.y = ELU(self.x)
        result = self.y * self.w_2
        return result

def predicted(input):
    # Uses input and starting bias 3 for neural network, returns predicted value.
    # First node.
    blue = first_node.run(input)

    # Second node.
    orange = second_node.run(input)

    # Output
    output = blue + orange + b_3
    return output

def ReLUPredicted(input):
    # Runs the predicted value through the ReLU activation function.

    output = ELU(predicted(input))
    return output

def dSSR_dw_1(index):
    # Calculates the gradient descent of SSR with respect to w_1.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * w_3 * d_ELU(first_node.x) * observed_x[i] * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_dw_2(index):
    # Calculates the gradient descent of SSR with respect to w_2.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * w_4 * d_ELU(second_node.x) * observed_x[i] * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_dw_3(index):
    # Calculates the gradient descent of SSR with respect to w_3.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * first_node.y * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_dw_4(index):
    # Calculates the gradient descent of SSR with respect to w_4.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * second_node.y * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_db_1(index):
    # Calculates the gradient descent of SSR with respect to b_1.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * w_3 * d_ELU(first_node.x) * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_db_2(index):
    # Calculates the gradient descent of SSR with respect to b_2.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * w_4 * d_ELU(second_node.x) * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dSSR_db_3(index):
    # Calculates the gradient descent of SSR with respect to b_3.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum

def dPredicted_da(input):
    # Calculates the derivative of Predicted with respect to a.
    if input >= 0:
        return 0
    else:
        return math.e**input - 1

def dSSR_da(index):
    # Calcaultes the gradient descent of SSR with respect to a.
    sum = 0
    for i in index:
        derivative = -2 * (observed_y[i] - ReLUPredicted(observed_x[i])) * d_ELU(predicted(observed_x[i]))
        sum += derivative

    return sum
        
def gradient_descent(sum, oldpara, learning_rate=learning_rate):
    '''Calculates step size and returns new paramter.'''
    step_size = sum * learning_rate
    newpara = oldpara - step_size
    return newpara

def step_size(sum, learning_rate=0.1):
    '''Calculates step size and returns step size.'''
    step_size = sum * learning_rate
    return step_size

# Neural Network BackPropogation
for trial in range(1, step_limit + 1):
    # Write new values to nodes.
    first_node = ELUNode(weight_1=w_1, bias_1=b_1, weight_2=w_3)
    second_node = ELUNode(weight_1=w_2, bias_1=b_2, weight_2=w_4)

    # Store current parameters.
    weight_status = f"w_1: {w_1}\nw_2: {w_2}\nw_3: {w_3}\nw_4: {w_4}\n"
    bias_status = f"b_1: {b_1}\nb_2: {b_2}\nb_3: {b_3}\n"
    alpha_status = f"alpha: {a}\n"
    data = f"\nWeights:\n{weight_status}\nBias:\n{bias_status}\nAlpha:\n{alpha_status}\n"

    # Grab values for the current graph.
    for input in range(1, 1001):
        x = input/1000
        y = predicted(x)
        x_values.append(x)
        y_values.append(y)

    # Add the values for animation and clear the lists.
    container, = ax.plot(x_values, y_values, linewidth=3, color='green')
    text_1 = ax.text(0.7, 1, data, fontsize=10, color='red')
    text_2 = ax.text(0, 1, f"Steps: {trial}", fontsize=12, color='purple')
    frames.append([container, text_1, text_2])
    x_values.clear()
    y_values.clear()

    # Calculate the derivative of the SSR, step size, and create new w_1.
    sum_w_1 = dSSR_dw_1(indexs)
    step1 = float(step_size(sum_w_1))

    # Calculate the derivative of the SSR, step size, and create new w_2.
    sum_w_2 = dSSR_dw_2(indexs)
    step2 = float(step_size(sum_w_2))

    # Calculate the derivative of the SSR, step size, and create new w_3.
    sum_w_3 = dSSR_dw_3(indexs)
    step3 = float(step_size(sum_w_3))

    # Calculate the derivative of the SSR, step size, and create new w_4.
    sum_w_4 = dSSR_dw_4(indexs)
    step4 = float(step_size(sum_w_4))

    # Calculate the derivative of the SSR, step size, and create new b_1.
    sum_b_1 = dSSR_db_1(indexs)
    step5 = float(step_size(sum_b_1))

    # Calculate the derivative of the SSR, step size, and create new b_2.
    sum_b_2 = dSSR_db_2(indexs)
    step6 = float(step_size(sum_b_2))

    # Calculate the derivative of the SSR, step size, and create new b_3.
    sum_b_3 = dSSR_db_3(indexs)
    step7 = float(step_size(sum_b_3))

    # Calculate the derivative of the SSR, step size and create new a.
    sum_a = dSSR_da(indexs)
    step8 = float(step_size(sum_a))
    
    # Set all the new values.
    b_1 = gradient_descent(sum_b_1, b_1)
    b_2 = gradient_descent(sum_b_2, b_2)
    b_3 = gradient_descent(sum_b_3, b_3)
    w_1 = gradient_descent(sum_w_1, w_1)
    w_2 = gradient_descent(sum_w_2, w_2)    
    w_3 = gradient_descent(sum_w_3, w_3)
    w_4 = gradient_descent(sum_w_4, w_4)

    # Set alpha.
    if a <= 0:
        a += 0.1
    else:
        a = gradient_descent(sum_a, a)

    if step1 < limit and step2 < limit and step3 < limit and step4 < limit and step5 < limit and step6 < limit and step7 < limit and step8 < limit:
        break

ani = animation.ArtistAnimation(fig, frames, interval=25, repeat=False)
ax.scatter(observed_x, observed_y, s=100)
ax.text(0, 1.5, f"Learning Rate: {learning_rate}", fontsize=12, color="black")
ax.set_xlabel("Doseage", fontsize=14)
ax.set_ylabel("Effectiveness", fontsize=14)
plt.xlim(0, 1)
plt.ylim(-2, 2)
plt.show()
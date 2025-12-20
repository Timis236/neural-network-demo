import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
INPUT_SIZE = 21

# Helper Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Class for a simple perceptron-type neural network
class Neural_Network:
    def __init__(self, num_inputs, learning_rate):
        # Initialize weights and bias randomly
        i = 1
        self.weight_vector = np.array([np.random.uniform(-1,1)])
        while (i < num_inputs):
            self.weight_vector = np.append(self.weight_vector, np.random.uniform(-1,1))
            i += 1
        self.bias = np.random.uniform(-1,1)
        
        self.learning_rate = learning_rate

    # Give a prediction for a particular input vector
    def predict(self, input_vector):
        # Weighted sum given as dot product of input vector and weights, plus bias
        wsum = np.dot(input_vector, self.weight_vector) + self.bias
        
        # Sigmoid activation function
        output = sigmoid(wsum)
        
        # If output < 0.5, we consider it a prediction of 0, otherwise it is a prediction of 1
        return round(output)
    
    # Computes the gradient of the error with respect to the bias/weights and adjusts the bias/weights accordingly
    def adjust_weights_bias(self, input_vector, target):
        # Weighted sum given as dot product of input vector and weights, plus bias
        wsum = np.dot(input_vector, self.weight_vector) + self.bias
        
        # Sigmoid activation function
        output = sigmoid(wsum)

        # Use backpropagation to adjust the weights and bias
        # Derivative of error function with respect to output -> d((output - target) ^ 2) = 2 * (output - target)
        d_error_wrt_output = 2 * (output - target)
        
        # Derivative of output with respect to weighted sum -> d(sigmoid(wsum)) = d_sigmoid(wsum) (as given in the defined function)
        d_output_wrt_wsum = d_sigmoid(wsum)
        
        # Derivative of wsum with respect to bias -> d(np.dot(input_vector, weights) + bias) = 1
        d_wsum_wrt_bias = 1
        
        # Derivative of wsum with respect to weights -> d(np.dot(input_vector, weights) + bias) = (0 * weights) + (1 * input_vector)
        d_wsum_wrt_weightvector = (0 * self.weight_vector) + (1 * input_vector)

        # Calculate derivatives (gradients) of error with respect to bias and weights and make corresponding changes to the values
        d_error_wrt_bias = d_error_wrt_output * d_output_wrt_wsum * d_wsum_wrt_bias
        d_error_wrt_weightvector = d_error_wrt_output * d_output_wrt_wsum * d_wsum_wrt_weightvector

        # Adjust each value by a percentage of the gradient, the size of which is determined by the learning rate
        self.bias = self.bias - (d_error_wrt_bias * self.learning_rate)
        self.weight_vector = self.weight_vector - (d_error_wrt_weightvector * self.learning_rate)

# Obtain training and testing data
train_list = []
train_list_output = []
test_list = []
test_list_output = []

can_train = True
try:
    fp = open("ann-train.data", "r")
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        # Omit any data with missing values
        if (len(data) == 22):
            # Convert all strings into floats
            for i in range(len(data)):
                data[i] = float(data[i])
            # Adjust the final value into a simple yes/no (1/0)
            if (data[-1] == 3):
                data[-1] = 0
            else:
                data[-1] = 1
            # Save the input and output in lists
            train_list.append(np.asarray(data[:-1]))
            train_list_output.append(data[-1])
    fp.close()
except:
    print("File 'ann-train.data' not available, the NN will not be able to train.")
    can_train = False
    
try:
    fp = open("ann-test.data", "r")
    lines = fp.readlines()
    for line in lines:
        data = line.split()
        # Omit any data with missing values
        if (len(data) == 22):
            # Convert all strings into floats
            for i in range(len(data)):
                data[i] = float(data[i])
            # Adjust the final value into a simple yes/no (1/0)
            if (data[-1] == 3):
                data[-1] = 0
            else:
                data[-1] = 1
            # Save the input and output in lists
            test_list.append(np.asarray(data[:-1]))
            test_list_output.append(data[-1])
    fp.close()
    
    # Define Neural Network
    nn = Neural_Network(INPUT_SIZE, LEARNING_RATE)

    # Part 1: Initial Testing
    print(f"Initial Weights: {nn.weight_vector}")
    print(f"Initial Bias: {nn.bias}")
    print()

    print("Testing Neural Network...")
    num_correct = 0
    total = 0

    for i in range(len(test_list)):
        # Keep track of the total number of tests and the correct predictions
        total += 1
        if (nn.predict(test_list[i]) == test_list_output[i]):
            num_correct += 1

    # Calculate and output the accuracy for the entire test
    accuracy = num_correct / total
    print(f"Accuracy: {accuracy * 100}%")
    print()

    if (can_train):
        # Part 2: Training
        # Keep track of the correct/incorrect predictions for future analysis
        print("Training Neural Network...")
        X = []
        Y = []

        # Run through the entire training set, adjusting weights/bias for each one
        for i in range(len(train_list)):
            nn.adjust_weights_bias(train_list[i], train_list_output[i])
            # Build coordinate points, where X is the index and Y is 1/0 if the prediction is correct/incorrect
            X.append(i)
            if (nn.predict(train_list[i]) == train_list_output[i]):
                Y.append(1)
            else:
                Y.append(0)

        print("Done!")
        # Create a plot to demonstrate the progression of the NN
        print("Displaying result graph. (Proceed to Testing by closing the graph)")
        plt.plot(X, Y)
        plt.title("Successful Predictions (Training)")
        plt.show()
        print()

        # Part 3: Final Testing
        print(f"Final Weights: {nn.weight_vector}")
        print(f"Final Bias: {nn.bias}")
        print()

        print("Testing Neural Network...")
        num_correct = 0
        total = 0

        for i in range(len(test_list)):
            # Keep track of the total number of tests and the correct predictions
            total += 1
            if (nn.predict(test_list[i]) == test_list_output[i]):
                num_correct += 1

        # Calculate and output the accuracy for the entire test
        accuracy = num_correct / total
        print(f"Accuracy: {accuracy * 100}%")
    else:
        print("Training file not present, no further progress can be made.")

except:
    print("File 'ann-test.data' not available, no data can be tested.")



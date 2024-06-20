import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Define activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Loss function and its derivative
def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-8))

def cross_entropy_derivative(predictions, labels):
    return predictions - labels

# Regularization functions
def l2_regularization(weights, lambda_param):
    return 0.5 * lambda_param * np.sum(weights ** 2)

# Define layers
class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        h, w, d = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, d = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3)) + self.biases

        return relu(output)

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_biases = np.zeros(self.biases.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                d_L_d_biases[f] += d_L_d_out[i, j, f]

        self.filters -= learn_rate * d_L_d_filters
        self.biases -= learn_rate * d_L_d_biases

        return None

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, d = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.pool_size):(i * self.pool_size + self.pool_size),
                                  (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, d = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, d))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.pool_size + i2, j * self.pool_size + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class DenseLayer:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(2.0/input_len) # Xavier initialization
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input = input
        input = input.flatten()
        self.last_input_shape = input.shape
        totals = np.dot(input, self.weights) + self.biases
        return relu(totals)

    def backward(self, d_L_d_out, learn_rate, lambda_param=0.001):
        d_L_d_w = np.dot(self.last_input[np.newaxis].T, d_L_d_out[np.newaxis])
        d_L_d_b = d_L_d_out
        d_L_d_input = np.dot(d_L_d_out, self.weights.T).reshape(self.last_input_shape)

        # Add L2 regularization to weights gradient
        d_L_d_w += lambda_param * self.weights

        # Update weights and biases
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        return d_L_d_input

class BatchNormLayer:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)

    def forward(self, input, training=True):
        if training:
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.input_normalized = (input - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.input_normalized + self.beta

    def backward(self, d_L_d_out, learn_rate):
        N, D = d_L_d_out.shape
        self.gamma -= learn_rate * np.sum(d_L_d_out * self.input_normalized, axis=0)
        self.beta -= learn_rate * np.sum(d_L_d_out, axis=0)
        d_input_normalized = d_L_d_out * self.gamma
        d_var = np.sum(d_input_normalized * (self.last_input - self.mean) * -0.5 * (self.var + self.epsilon)**(-1.5), axis=0)
        d_mean = np.sum(d_input_normalized * -1 / np.sqrt(self.var + self.epsilon), axis=0) + d_var * np.sum(-2 * (self.last_input - self.mean), axis=0) / N
        return d_input_normalized / np.sqrt(self.var + self.epsilon) + d_var * 2 * (self.last_input - self.mean) / N + d_mean / N

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape)
            return input * self.mask
        else:
            return input * (1 - self.dropout_rate)

    def backward(self, d_L_d_out):
        return d_L_d_out * self.mask

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

# Define VGG network structure
class VGG:
    def __init__(self):
        self.layers = [
            ConvLayer(64, 3, 3),  # input depth = 3 (RGB image)
            ConvLayer(64, 3, 64),
            MaxPoolLayer(2),
            ConvLayer(128, 3, 64),
            ConvLayer(128, 3, 128),
            MaxPoolLayer(2),
            ConvLayer(256, 3, 128),
            ConvLayer(256, 3, 256),
            ConvLayer(256, 3, 256),
            MaxPoolLayer(2),
            ConvLayer(512, 3, 256),
            ConvLayer(512, 3, 512),
            ConvLayer(512, 3, 512),
            MaxPoolLayer(2),
            ConvLayer(512, 3, 512),
            ConvLayer(512, 3, 512),
            ConvLayer(512, 3, 512),
            MaxPoolLayer(2),
            DenseLayer(512 * 1 * 1, 4096),  # Adjust according to final output size
            DenseLayer(4096, 4096),
            DenseLayer(4096, 10)  # 10 classes for CIFAR-10
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learn_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learn_rate)
        return grad

    def train(self, x, y, learn_rate=0.001, epochs=1, optimizer=None, batch_size=32):
        if optimizer is None:
            optimizer = SGD(lr=learn_rate)
        
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward pass
                outputs = np.array([self.forward(xi) for xi in x_batch])

                # Compute loss and gradient
                loss = np.mean([cross_entropy_loss(output, yi) for output, yi in zip(outputs, y_batch)])
                grads = np.array([cross_entropy_derivative(output, yi) for output, yi in zip(outputs, y_batch)])

                # Backward pass
                for grad in grads:
                    self.backward(grad, learn_rate)

                print(f'Epoch {epoch + 1}/{epochs}, Batch {i // batch_size + 1}/{len(x) // batch_size + 1}, Loss: {loss:.4f}')

# Load CIFAR-10 dataset using torchvision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../dataset/cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Extract and preprocess training data
train_data = []
train_labels = []

for images, labels in trainloader:
    for i in range(len(images)):
        train_data.append(images[i].numpy())
        train_labels.append(np.eye(10)[labels[i]])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Initialize VGG model and train
vgg = VGG()
vgg.train(train_data, train_labels, learn_rate=0.001, epochs=10, batch_size=128)

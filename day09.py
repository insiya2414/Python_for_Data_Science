'''Q46 Python function to tokenize a sentence into words (split by spaces, removing punctuation)'''
# Tokenize: To split a sentence into individual words (tokens), while ignoring punctuation and converting everything to lowercase for uniformity.

import re # Python’s Regular Expressions library, which helps match patterns in strings.
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())
# text.lower()
    # Converts the entire text to lowercase so that "Hello" and "hello" are treated the same.
# re.findall(r'\b\w+\b', text.lower())
    # This uses a regular expression to find all the words:
    # \b: Word boundary — ensures we match entire words only.
    # \w+: One or more word characters (letters, digits, and underscores).
    # \b: Ends the word boundary.

text = "Hello, how are you?"
tokens = simple_tokenizer(text)
print(tokens)
#  Output:['hello', 'how', 'are', 'you']


'''Q47 Implement a simple NER function using regular expressions to extract names, locations, and dates'''
import re

def simple_ner(text):
    names = re.findall(r'\b[A-Z][a-z]*\b', text)
    locations = re.findall(r'\b(?:New York|Paris|London)\b', text)  
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)  
    return {'names': names, 'locations': locations, 'dates': dates} # returns a dictionary with three types of entities

text = "John went to New York on 12/12/2020"
entities = simple_ner(text)
print(entities)
# Output: {'names': ['John'], 
#        'locations': ['New York'], 
#        'dates': ['12/12/2020']}   


'''Q48 function to convert a color image into grayscale'''
# A grayscale image contains only shades of gray—no color. Each pixel is a single intensity value (0–255), unlike color images where each pixel has 3 values (R, G, B).

import cv2 # cv2 is OpenCV, a popular library for computer vision tasks.

def to_grayscale(image_path):
    image = cv2.imread(image_path) # Reads the image using OpenCV. The image is loaded in BGR format (Blue, Green, Red), which is OpenCV's default.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Converts the image from BGR to grayscale using OpenCV’s cvtColor function.
# Internally, OpenCV uses the weighted sum formula:
# Gray = 0.299 × 𝑅 + 0.587 × 𝐺 + 0.114 × 𝐵
# This mimics human perception, as we’re more sensitive to green than blue.

'''Q49  custom loss function that calculates the mean squared error but with a twist: 
it penalizes the model more if the predicted values are too large'''
import tensorflow as tf # using TensorFlow to define and compute the loss function

def custom_loss(y_true, y_pred): # This function will be used during training to compare true labels (y_true) and model predictions (y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred)) # Standard MSE: the average of the squares of the errors (differences between actual and predicted values)
    penalty = tf.reduce_mean(tf.square(y_pred)) * 0.01  # Penalize large predictions, 0.01 is a hyperparameter controlling the strength of the penalty
    return mse + penalty # The final loss is a combination of:
                            # Error term (MSE), # Penalty term (extra cost if predictions are large)

# y_true = [1.0, 2.0]
# y_pred = [10.0, 12.0]
# MSE will be large because predictions are far off.
# The penalty will make the loss even higher due to the large values in y_pred.
# This encourages the model to predict closer values and avoid unnecessary magnitude.

# 🧠 What's the purpose of a loss function?
# When you're training a machine learning model, like a neural network, the loss function tells the model how bad its predictions are.
# The goal is to minimize this loss so that predictions get closer to the real answers.

# 🔍 What is Mean Squared Error (MSE)?
# MSE is a popular loss function used for regression problems. It looks at the difference between actual and predicted values, squares them (so they're all positive), and averages them.

# Example:
# True	Predicted	Error	Squared Error
# 2	        3	     -1	         1
# 4	        6	     -2	         4
# Avg: 2.5

# The model uses this 2.5 number to adjust itself and improve.
# This encourages the model to predict closer values and avoid unnecessary magnitude.

'''Q50 Implement a custom activation function that combines ReLU and sigmoid. Use it in a neural network model'''
# 🔍 What are ReLU and Sigmoid?
# ReLU (Rectified Linear Unit):
# ReLU(x) = max(0, x)
# Keeps positive values as they are
# Turns negative values to 0
# Popular because it's simple and effective

# Sigmoid:
# Sigmoid(x) = 1 / (1 + exp(-x))
# Outputs values between 0 and 1
# Smooth and squashes values
# Often used in binary classification

# 🤔 Why combine them?
# Non-linearity from ReLU (help with learning complex patterns)
# Squashing effect of sigmoid (keep outputs bounded and smooth)

# By multiplying them:
# combined = ReLU(x) * Sigmoid(x)
# ReLU removes negatives (like a gate)
# Sigmoid smooths and limits the positive values

# This custom activation has characteristics of both:
# ✅ Keeps values positive
# ✅ Limits them in scale
# ✅ Is smooth and differentiable

import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomActivation(Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs) * tf.sigmoid(inputs)
# A custom Keras Layer is created
# Inside it, the activation function is defined
# It applies both ReLU and Sigmoid to the input and multiplies them

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=CustomActivation(), input_shape=(784,))
])
# Input layer has 784 features (e.g., flattened 28x28 image like MNIST)
# The dense layer has 128 neurons
# Each neuron's output is passed through your custom activation function


# 🧠 1. What is MNIST?
# MNIST (Modified National Institute of Standards and Technology database) is:
# A dataset of handwritten digits.
# 60,000 training images and 10,000 test images.
# Each image is 28x28 pixels and grayscale (black & white).
# The goal: Recognize which digit (0–9) is written in the image.

# 📸 Example:
# Image of a handwritten "5" → Your model should predict 5
# It's often used to learn and test image classification in machine learning and deep learning.

# 🤖 2. What is Keras?
# Keras is:
# A high-level deep learning API (Application Programming Interface).
# Built on top of TensorFlow, a powerful ML library by Google.
# It allows you to easily build, train, and test neural networks.
# Think of Keras like a simple tool that lets you build models quickly, like assembling LEGO blocks.

# 🔗 3. What is a Neural Network?
# A neural network is:
# A series of connected “nodes” or neurons arranged in layers.
# Each neuron takes input → processes it → passes it to the next layer.
# Inspired by how human brains process information.

# 📦 Structure:
# Input Layer → Hidden Layers → Output Layer
# Each connection has a weight, and the network learns by adjusting those weights using data.

# ⚡ 4. What is a Neuron?
# In neural networks, a neuron is:
# A computational unit.
# It takes numbers in (input), multiplies them by weights, adds bias, and passes the result through an activation function.

# Example of what one neuron does:
# output = activation(weight * input + bias)
# Think of a neuron as a tiny calculator that helps the model make decisions.

# 🏗️ 5. What is a Dense Layer?
# A dense layer (also called fully connected layer) means:
# Every neuron in one layer is connected to every neuron in the next layer.
# It’s the most common type of layer in neural networks.

# In Keras:
# tf.keras.layers.Dense(128, activation='relu')
# 128 neurons
# Each neuron applies ReLU to its inputs

# 🎚️ 6. What is an Activation Function?
# An activation function decides whether a neuron should be activated or not.
# Why do we need it?
# Without it, a neural network becomes just a big linear equation.
# With it, we add non-linearity → meaning the model can learn complex patterns.

# Common examples:

# Activation	Description
# ReLU	       Keeps positive values, zeros out negatives
# Sigmoid	   Squashes input to a range of 0 to 1
# Softmax	   Used in final layer for classification (turns values into probabilities)

# 🧱 7. What are Keras Layers?
# Keras provides many types of layers to build your model:
# Dense – Fully connected layers
# Conv2D – For image convolution
# Dropout – To prevent overfitting
# Flatten – Converts 2D to 1D
# Custom layers – You can also build your own!
# Each layer transforms the data and passes it to the next.

# Example model:
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# 🧠 How It All Fits Together
# Imagine this pipeline:
# Input: 28x28 image from MNIST → Flatten to 784 numbers.
# First Dense Layer: 128 neurons, uses ReLU.
# Second Dense Layer: 10 neurons (for 10 digits), uses Softmax.
# Output: Model predicts a digit from 0 to 9.

# ⚙️ Keras works with TensorFlow
# Keras is built on top of TensorFlow (which does the heavy work). You can think of:
# TensorFlow = Engine
# Keras = Dashboard/Interface to control the engine
# Keras is a tool (a Python library) that helps you build and train neural networks easily.


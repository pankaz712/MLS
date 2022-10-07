### first practicle lab for ML specialization

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

plt.style.available

plt.style.use('ggplot')

# x_train - input variable (size in 1000 sqaure feet)
# y_train - target (price in 1000s of dollars)

x_train = np.array([1, 2])
y_train: ndarray = np.array([300, 500])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples

print(f"x_train.shape:{x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is : {m}")
len(x_train)

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# plotting the data

# plot data points
plt.scatter(x_train, y_train, marker='x', c='r')
# adding title
plt.title('Housing Prices')
# set the y-axis label
plt.ylabel('Price (in 1000s of dollars) ')
# set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

# creating model function

w = 200
b = 100

print(f"w: {w}")
print(f"b: {b}")


# creating a function

def compute_model_output(x, w, b):
    '''
    Computes the prediction of a linear model
    @param x: x (ndarray (m,)) : Data, m examples
    @param w: w (scalar) : model parameters
    @param b: b (scalar) : model parameters
    @return:  y (ndarray (m,)) : target values
    '''

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

print(compute_model_output.__doc__)

# call the function and plot the output
tmp_f_wb = compute_model_output(x_train, w, b,)

# plot the model predictions

plt.plot(x_train,tmp_f_wb,c='b',label='Our Prediction')

# plot the data points

plt.scatter(x_train,y_train,marker='x',c='r',label = 'Actual Values')

# set the title

plt.title("Housing Prices")

# set the y axis label

plt.ylabel('Price (in 1000s of dollars)')

# set the x axis label

plt.xlabel('Size (1000 sqft)')

plt.legend()
plt.show()

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")


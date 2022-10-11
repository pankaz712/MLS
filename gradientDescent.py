import copy
import math

import matplotlib.pyplot as plt
import numpy as np

# dataset

x_train = np.array([1.0, 2.0, 3.0, 5.0])
y_train = np.array([300.0, 500.0, 800.0, 1000])

plt.scatter(x_train, y_train)


# function to compute cost

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2

    total_cost = 1 / (2 * m) * cost

    return total_cost


# function compute gradient

def compute_gradient(x, y, w, b):
    """
    @param x: Data m examples
    @param y: target values
    @param w: model paramater
    @param b: mddel paramater
    @return:  dj_dw : gradient of the cost function wrt w
              dj_db : gradient of the cost function wrt b
    """

    # number of training examples

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])

        dj_db = dj_db + dj_db_i
        dj_dw = dj_dw + dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# let us test the above two functions

compute_cost(x=x_train, y=y_train, b=100, w=100)  # cost of 46250
compute_gradient(x=x_train, y=y_train, w=100, b=100)  # -925.0,-275.0

# lets plot

w_values = np.array(range(100, 700, 50))
b_value = 100

# w_values[2]
# list(range(len(w_values)))

costList = []
gradientList = []
for i in range(len(w_values)):
    print(i)
    cost_i = compute_cost(x=x_train, y=y_train, w=w_values[i], b=b_value)
    costList.append(cost_i)
    gradient_i = compute_gradient(x=x_train, y=y_train, w=w_values[i], b=b_value)
    gradientList.append(gradient_i)

print(costList)
print(gradientList)


def Extract(lst):
    """
    @param lst: list of lists
    @return: first element of each sublist
    """
    return [item[0] for item in lst]


slope_w = Extract(gradientList)  # extracting slope wrt to w

# plotting cost and gradient for values of w as we have fixed b

plt.plot(w_values, costList, c='b', label='Cost')
plt.twinx()
plt.plot(w_values, slope_w)


# gradient is 0 and cost is minimum at w = 200

# gradient descent implementation to find optimal values of w and b
# will calculate cost and gradient at each step

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps
    with learning rate alpha

    @param x: Data, m examples
    @param y: target values
    @param w_in: initial value of w parameter
    @param b_in: initial value of b parameter
    @param alpha: Learning rate
    @param num_iters: number of iterations to run gradient descent
    @param cost_function: function to call to produce cost
    @param gradient_function: function to call to produce gradient
    @return:
    w : Updated value of paramater after running gradient descent
    b : Updated value of paramater after running gradient descent
    J_history : History of cost values
    p_history : History of params [w,b]
    """

    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # calculate the gradient and update the parameters using gradient function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # update parameters

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # save cost J at each iteration

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print cost at intervals 10 times

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iterations {i:4}: Cost {J_history[-1]:0.2e}",
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}",
                  f"w: {w:0.3e}, b: {b:0.5e}")

    return w, b, J_history, p_history


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost');
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step');
ax2.set_xlabel('iteration step')
plt.show()

# plotting fit

y_hat = []

for i in range(len(x_train)):
    y_i = w_final * x_train[i] + b_final
    y_hat.append(y_i)

# plot the model predictions

plt.plot(x_train, y_hat, c='b', label='Our Prediction')

# plot the data points

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

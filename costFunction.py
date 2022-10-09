import numpy as np
import matplotlib.pyplot as plt
plt.style.available
plt.style.use('ggplot')


# lets create a function that is the true data gen process

def true_func(x):
    '''
    @param x: input number
    @return: reutns y mapped by f(x) = intercept + B1*x
    '''
    # s = np.random.normal(mu, sigma, 1)
    y = 500 + (300 * x)
    y = int(y)
    return y

testReturn = true_func(x=1000)
x_input = np.random.normal(3000, 500, 1000) # generating x training data
x_input = x_input.astype(int)

v_true_func = np.vectorize(true_func)
y_input = v_true_func(x_input)
randError = + np.random.normal(0, 1000, len(y_input)) # adding a random error
y_input = y_input + randError
y_input = y_input.astype(int)

# plotting histograms

plt.hist(x_input)
plt.hist(y_input)

# plotting scatterplot

plt.scatter(x_input, y_input)


# cost function
# f(x) = w(x) + b
# parameters are w,b
# m is the number of datapoints in training set ie pair of x and y
# cost function  J(w,b) = 1/(2m) * (sum(f(x) - y)^2 for training data points in m)

def compute_cost(x, y, w, b):
    '''
    @param x: 1d array of inputs x in training data
    @param y: 1d array of inputs y in test data
    @param w: model parameter
    @param b: model parameter
    @return: total cost for using w,b as the parameters
    '''
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])
        cost = cost.astype(float)
        cost = pow(cost, 2)
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    total_cost = round(total_cost, 0)
    return total_cost


# running manual checks

# predictions using values of w

def compute_predictions(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

w = 300
b = 500
tmp_f_wb = compute_predictions(x_input, w, b, )

# plot the model predictions
plt.plot(x_input, tmp_f_wb, c='b', label='Our Prediction')
# plot the data points
plt.scatter(x_input, y_input, marker='x', c='r', label='Actual Values')
err = tmp_f_wb - y_input
plt.hist(err)

wGuess = list(range(100, 700, 50))  # guess range for the value of parameter w

costList = []
for i in range(len(wGuess)):
    print(i)
    cost_i = compute_cost(x=x_input, y=y_input, w=wGuess[i], b=500)
    costList.append(cost_i)
print(costList)

# plotting the cost against value of w, cost minimizes at 300 which is the true value of w
plt.plot(wGuess, costList, c='b', label='Cost')

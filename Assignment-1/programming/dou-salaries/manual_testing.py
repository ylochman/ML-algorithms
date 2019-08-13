import numpy as np
import pandas as pd
np.random.seed(42)

x = np.array([1, -0.5, 3, 1])
X = np.array([
    [1, -0.5, 3, 1],
    [2, 8, -0.33, 5],
    [0, 0, 0, 0]
])
y = np.array([40, 100, 12])
theta = np.array([2., 5, 7, 9])
eps = 0.001

def assertFloatEquals(a, b):
    assert np.abs(a - b) < eps
    
def assertArrayEquals(a, b):
    a = np.array(a)
    b = np.array(b)
    assert a.shape == b.shape
    assert np.all(np.abs(a - b) < eps)
    
def assertTrue(statement):
    assert statement

expected = 29.5
actual = predict_linear(theta, x)
assert actual == expected

expected = [29.5, 86.69, 0]
actual = (predict_linear(theta, X))
assertArrayEquals(actual, expected)

weights = get_example_weights(X, x, tau=5)
assertTrue(weights.shape[0] == X.shape[0])

expected = [1.000, 0.134, 0.798]
actual = get_example_weights(X, x, tau=5)
assertArrayEquals(actual, expected)

weights = np.ones(X.shape[0])
expected = 71.901
actual = cost_function(theta, X, y, weights)
assertFloatEquals(actual, expected)

weights = np.array([0.5, 0.1, 0.28])
expected = 18.860
actual = cost_function(theta, X, y, weights)
assertFloatEquals(actual, expected)

weights = np.ones(X.shape[0])
grad = cost_function_gradient(theta, X, y, weights)
assertTrue(grad.shape == theta.shape)

weights = np.ones(X.shape[0])
expected = [-37.12, -101.23, -27.108, -77.05]
actual = cost_function_gradient(theta, X, y, weights)
assertArrayEquals(actual, expected)

weights = np.array([0.5, 0.1, 0.28])
expected = [-7.912, -8.023, -15.311, -11.905]
actual = cost_function_gradient(theta, X, y, weights)
assertArrayEquals(actual, expected)

grad = np.zeros(theta.shape[0])
theta_new = update_model_weights(theta, learning_rate=1, cost_gradient=grad)
assertArrayEquals(theta_new, theta)

grad = np.array([1.35, -0.89, 0.16, 0.98])
expected = [0.65, 5.89, 6.84, 8.02]
actual = update_model_weights(theta, learning_rate=1, cost_gradient=grad)
assertArrayEquals(actual, expected)

grad = np.array([1.35, -0.89, 0.16, 0.98])
expected = [1.730, 5.178, 6.968, 8.804]
actual = update_model_weights(theta, learning_rate=0.2, cost_gradient=grad)
assertArrayEquals(actual, expected)
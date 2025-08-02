# ProfitPredictor

Absolutely! Below is a rewritten version of the `README.md` where the **important technical content remains intact**, but the tone has been shifted to sound like **you** are describing the lab yourself—clear, direct, and from a personal/explanatory point of view, rather than like a student instruction prompt.

---

# 📊 Linear Regression: Predicting Restaurant Profits

This project implements **Linear Regression with one variable** using **Gradient Descent** to predict restaurant profits based on population data. The goal is to estimate potential monthly profits for new locations, based on existing data from other cities.

---

## 🧠 What This Covers

I used this lab to:

* Build a linear regression model from scratch.
* Understand and implement the cost function.
* Apply batch gradient descent to learn the parameters.
* Visualize how well the model fits the training data.
* Use the model to make predictions for new city populations.

---

## 🧰 Libraries and Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *  # helper functions
import copy
import math
```

---

## 🗂️ Dataset Details

The dataset includes:

* `x_train`: City population (in 10,000s)
* `y_train`: Monthly profit (in \$10,000s)

```python
x_train, y_train = load_data()
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("First few x_train values:", x_train[:5])
print("First few y_train values:", y_train[:5])
```

---

## 📈 Data Visualization

I started by visualizing the dataset using a scatter plot to better understand the relationship between population and profit:

```python
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per City")
plt.xlabel("Population of City (in 10,000s)")
plt.ylabel("Profit (in $10,000)")
plt.show()
```

---

## 📘 Model Overview

The model is a basic linear regression equation:

$$
f_{w,b}(x) = wx + b
$$

The cost function used to measure error is:

$$
J(w, b) = \frac{1}{2m} \sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

---

## 💰 Cost Function Implementation

Here’s the implementation for the cost function:

```python
def compute_cost(x, y, w, b): 
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i])**2
    total_cost = total_cost / (2 * m)
    return total_cost
```

Example usage:

```python
initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(f"Cost at initial w: {cost:.3f}")  # Expect around 75.203
```

---

## 🔁 Gradient Descent Logic

To minimize the cost, I used gradient descent:

$$
\begin{aligned}
w &:= w - \alpha \cdot \frac{\partial J}{\partial w} \\
b &:= b - \alpha \cdot \frac{\partial J}{\partial b}
\end{aligned}
$$

Gradient computation:

```python
def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
```

---

## 🧪 Running Batch Gradient Descent

```python
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    m = len(x)
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history
```

Training:

```python
initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01

w, b, _ = gradient_descent(x_train, y_train, initial_w, initial_b, 
                           compute_cost, compute_gradient, alpha, iterations)
print(f"w, b found by gradient descent: {w}, {b}")
```

---

## 📉 Visualizing the Fit

```python
predicted = w * x_train + b

plt.plot(x_train, predicted, c='b')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per City")
plt.xlabel("Population (in 10,000s)")
plt.ylabel("Profit (in $10,000)")
plt.show()
```

---

## 📍 Predictions

Using the final model parameters, I made predictions for two cities:

```python
predict1 = 3.5 * w + b
predict2 = 7.0 * w + b

print(f"For population = 35,000, predicted profit: ${predict1 * 10000:.2f}")
print(f"For population = 70,000, predicted profit: ${predict2 * 10000:.2f}")
```

Expected output:

```
For population = 35,000, predicted profit: $4519.77  
For population = 70,000, predicted profit: $45342.45
```

---

## ✅ Summary

This was a hands-on implementation of linear regression using gradient descent. It was a great introduction to:

* How a cost function works
* How to optimize using gradients
* Making predictions with a trained model

This same approach can scale to multiple features and even more advanced models later on.

---

## 📁 File Structure

```
linear-regression-lab/
├── README.md
├── utils.py
├── linear_regression.py
└── data/
    └── city_profits.csv (optional)
```

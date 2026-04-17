# Linear Regression Mathematics

This document explains the mathematical foundations of the simple linear regression implementation used in this project.

## 1. The Hypothesis Function
The hypothesis function $h_\theta(x)$ is the linear equation that our model uses to predict the output $y$ (e.g., price) for a given input feature $x$ (e.g., mileage).

$$h_\theta(x) = \theta_0 + \theta_1 x$$

- $\theta_0$: The y-intercept.
- $\theta_1$: The slope of the line (weight of the feature $x$).

## 2. Data Normalization (Z-Score Standardization)
To ensure the Gradient Descent algorithm converges efficiently, the input features and outputs are normalized to have a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$.

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

- $\mu$: The mean (average) of the feature.
- $\sigma$: The standard deviation of the feature.

## 3. The Cost Function (Mean Squared Error)
The cost function $J(\theta_0, \theta_1)$ measures the average squared difference between the model's predictions and the actual data points. Our goal is to minimize this cost.

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

- $m$: The total number of training examples.
- $x^{(i)}$: The $i$-th input feature.
- $y^{(i)}$: The $i$-th actual output value.

*Note: We divide by $2m$ instead of $m$ to make the calculation of the derivative cleaner, as the $2$ from the exponent will cancel out.*

## 4. Gradient Descent
Gradient Descent is an iterative optimization algorithm used to find the parameters ($\theta_0$ and $\theta_1$) that minimize the cost function. It updates the parameters by taking steps proportional to the negative of the gradient (partial derivative) of the cost function.

**Update Rules:**

Repeat until convergence:
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$
$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} ((h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)})$$

- $\alpha$: The learning rate, which controls the size of the steps taken towards the minimum.
- $:=$ represents assignment (updating the value simultaneously for all parameters).

## 5. Denormalization (Making Predictions)
When predicting a value for a new, unscaled input, we must first normalize the input using the training data's $\mu_x$ and $\sigma_x$. The model then outputs a normalized prediction, which must be denormalized back to its original scale using the training data's $\mu_y$ and $\sigma_y$.

$$y_{actual} = (y_{predicted\_scaled} \times \sigma_y) + \mu_y$$
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "bffaae54822766603c06f7199ffe8797", "grade": false, "grade_id": "cell-23f9a13fcbe0410a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Exercise Sheet No. 4
#
# ---
#
# > Machine Learning for Natural Sciences, Summer 2023, Jun.-Prof. Pascal Friederich
# > 
# > Deadline: May 15th 2023, 8am
# >
# > Container version 1.0.0
# >
# > Tutor: chen.zhou@kit.edu
# >
# > **Please ask questions in the forum/discussion board and only contact the Tutor when there are issues with the grading**
#
#
# ---
#
# **Topic**: 
# This exercise sheet (24 points in total) will focus on the math basics for ML. You will implement another simple ML algorithm, **linear regression** (LR), instead of decision tree to work on the same task you have already seen in exercise 02. **Gradient decent**, **mean square error loss function**, and **mean absolute error function** are covered in this exercise. And you will learn to use **ridge regression** to control overfitting.
# -


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "82ea78fe0c8959187f922be45be12b78", "grade": false, "grade_id": "cell-6754d0c5a478283c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Linear Regression
# In assignment 2, we have seen the use of decision tree for iris species classification. Now, let's try to solve the same problem with another simple machine learning technique: linear regression (LR).   
#
# As you may have already learned from the lecture, linear regression uses a linear combination of features to predict a target. In our case we can use a linear combination of the four flower descriptors to predict the species.
#
# In this assignment, you will learn to implement the Loss function and Gradient Decent for optimization. Then, we will see that even small model like LR can overfit, and how regularization (ridge regression) can help against this problem.
# -

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import random
from functools import wraps
warnings.filterwarnings('ignore')


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c85a1f4f71c05ff0d236a0fed190d64f", "grade": false, "grade_id": "cell-1cb644d0633a97e5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Preprocessing
# Let's start with preprocessing the dataset, as already learned from assignment 2.

# +
def log_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"\n{func.__name__}:")
        return result
    return wrapper

def log_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"\tshape: {result.shape}")
        return result
    return wrapper

def log_columns(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"\tcolumns: {result.columns.values}")
        return result
    return wrapper

@log_columns
@log_shape
@log_name
def load(df, path):
    """Loads the dataset from path."""
    df = pd.read_csv(path)
    return df

@log_columns
@log_shape
@log_name
def convert_to_categorical(df, col_name: str):
    df[col_name] = df[col_name].astype('category')
    return df

@log_columns
@log_shape
@log_name
def add_class_labels(df):
    df['class'] = df['species'].cat.codes
    return df


# -

df = pd.DataFrame()
df = (
    df.pipe(load, 'iris.csv')
      .pipe(convert_to_categorical, 'species')
      .pipe(add_class_labels)
)

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "bacac97568c793d09cb905d237ed2332", "grade": false, "grade_id": "cell-0b250c3f733fc1f4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Plot the data
# It is always a good practice to have some inspection on the dataset. This provides us with information about the data such as its structure, range, outliers etc., which may help on the design of ML algorithm. There are numerous ways to visualize data, and introduced here is the one called "Pairplot", which displays pairwise relationships in a dataset (you probably have already seen it in lecture slides). Pairplot can be implemented easily with the [seaborn](https://seaborn.pydata.org/generated/seaborn.pairplot.html?highlight=pairplot#seaborn.pairplot) library. Here we used it to show the pairwise relationships among iris features.

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "dfc92cc47a6425740e9b9cda2fe97bee", "grade": false, "grade_id": "cell-a75562b66630b58d", "locked": true, "schema_version": 3, "solution": false, "task": false}
sns.pairplot(df.loc[:,:'species'], hue= 'species')
plt.show()

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3e843b7e6fe84c67a0bf4215fba3843a", "grade": false, "grade_id": "cell-d45fb09151e8b752", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Question**: out of these combinations of features, which pair has the strongest linear correlation relationship?
#
# **1.** sepal_length and petal_width    
# **2.** sepal_width and petal_length    
# **3.** petal_length and petal_width    
# **4.** sepal_width and petal_length    
# **5.** sepal_length and sepal_width
#
# (To learn more about the linear correlation, please refer to [here](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient))
#
# Assign the number of your choice to the variable `A`:

# + deletable=false nbgrader={"cell_type": "code", "checksum": "ce8e995fe0828a8ce02423c932289142", "grade": false, "grade_id": "Correlation_Plot-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 1: select the correct choice. (1 point in total)

A = (
int(3)
#raise NotImplementedError()
)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "53240d3ef4f7bb5dd4d3b57fbc060a0a", "grade": true, "grade_id": "Correlation_Plot-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
assert type(A) == int # Please make sure your input is an int.

# Hidden test below
# 1 point for correct answer

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1663c643186725844cbb68b76e2e4847", "grade": false, "grade_id": "cell-ebcde626c67fdc9b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Generate training/validation set
# Now, for our machine learning task, let's extract feature matrix `X` and label matrix `Y` from the dataframe `df`. Please implement the `split_X_Y` function that returns feature `X` as a $150 \times 4$ numpy array, as well as label `Y` as a $150 \times 1$ numpy array. `X` should contain values from column "sepal_length", "sepal_width", "petal_length" and "petal_width" (**Please make sure to follow this order for grading purpose**). `Y` should be the value of column "class". You may use methods such as `.reshape()` to reshape an array, and attributes like `.values` to access values in columns of the data frame.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "c6fb8e677fe4d9701263e3cafbbf7055", "grade": false, "grade_id": "Train_Test_Split-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 2: Implement split_X_Y. (3 points in total)

def split_X_Y(df):
    """
    split the dataframe into feature matrix X and label matrix Y
    
    Args:
        df (pandas.DataFrame): shape (150, 6)
    
    Returns:
        X (numpy.ndarray): array containing values of 4 features of iris. Shape (150, 4) 
        Y (numpy.ndarray): array containing target value in the column "class". Shape (150, 1)
    """
    X = None # please update this in your solution
    Y = None # please update this in your solution
    X = np.array(df.iloc[:, :-2].values) 
    Y = np.array(df.iloc[:, -1].values).reshape(-1, 1)
    #raise NotImplementedError()
    return X, Y

split_X_Y(df)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d2bf8ea124290286762fd1726e8f5aaf", "grade": true, "grade_id": "Train_Test_Split-test_1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# feature/target split
X, Y = split_X_Y(df)

# 1 point for correct shape of X and Y
assert X.shape == (150, 4)
assert Y.shape == (150, 1)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ffcb73f8b72cb7d999e21a8a08083014", "grade": true, "grade_id": "Train_Test_Split-test_2", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 2 points: values in X and Y are checked

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "7d02b5b63fd7bc759dac1cfb11ee1d6c", "grade": false, "grade_id": "cell-af56efe9d9027cdf", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We have learned that the linear model tries to fit the function:
# \begin{align}
# y = \omega^T x + \omega_0 = \sum^n_{i=1} \omega_i x_i + \omega_0
# \end{align}
# Where $\omega$ is the weight matrix and $\omega_0$ is the bias term. Let $x_0 = 1$, this function can be rewritten into:
# \begin{align}
# y =  \sum^n_{i=1} \omega_i x_i + \omega_0 x_0 = \sum^n_{i=0} \omega_i x_i = X \Omega^T
# \end{align}
# $\Omega$ can be initialized randomly, and optimized later through training. In our case $\Omega$ has five elements - the first one for the bias and rest four for the weights of the four features.
#
# **Please note that $\Omega$ is written as a row vector in many text books for display convenience and is actually a column vector during implementation. That's why we have $\Omega^T$ in the equation to denote the convertion. Here we use `numpy.T` to make our code and equation consistent.**

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7d06186b74d4b27daa9e6defa279d973", "grade": false, "grade_id": "cell-faa32eb08af53092", "locked": true, "schema_version": 3, "solution": false, "task": false}
# random initialization for weights and bias
np.random.seed(0)
omega = np.random.randn(1,5)
omegaT = omega.T
omegaT

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d5177e914af51aedb7102d87477dfac3", "grade": false, "grade_id": "cell-75bfd7154053f864", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now let's contract $x_0 = 1$ to our feature matrix `X` to obtain the input matrix for LR model. Please do that by stacking a new column to `X` as the first colunm with all values equal to $1$. This may be easily implemented with the numpy method `.hstack()` . You may find more details [here](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html).

# + deletable=false nbgrader={"cell_type": "code", "checksum": "fde24ecada6404395fa40527231be465", "grade": false, "grade_id": "Add_bias_term_to_X-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 3: stack to X a new column with value 1. (2 points in total)

# Update feature matrix X with x0 = 1
X = (
np.hstack((np.ones((X.shape[0], 1)), X))
#raise NotImplementedError()
)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2c438304677e91f8215632f604681a67", "grade": true, "grade_id": "Add_bias_term_to_X-test_1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point for correct shape of X after stacking
assert X.shape == (150, 5) # now the shape of X should change from (150, 4) to (150, 5)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7109659f913446f9ea2736a77fe03038", "grade": true, "grade_id": "Add_bias_term_to_X-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point: values in X are checked
assert (X[:, 0] == 1).all()

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ce376689dff772ce6037f0b200f48210", "grade": false, "grade_id": "cell-3063ec8d8014755b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# In case you are interested, there's a "closed-form solution" that estimates the best parameter set $\Omega^*$ by solving the equation:
# \begin{align}
# \Omega^* = (X^TX)^{-1}X^Ty
# \end{align}
# Watch this [video](https://www.coursera.org/lecture/ml-regression/approach-1-closed-form-solution-G9oBu) from coursera for detailed explanation.

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5de70719b7511a3ceeff8793f6dcebbe", "grade": false, "grade_id": "cell-3b6704681c9de592", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Gradient Decent
# For more complex tasks, a closed form solution often doesn't exist and hence we will introduce another approach here, aka Gradient Decent (GD) algorithm, to solve the linear regression. Gradient Decent works by updating the weight vector incrementally after each epoch. Read more [here](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent).

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3e098ddae9e88235ebf9fbb6776910db", "grade": false, "grade_id": "cell-2aac0b9bb5d8eb89", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Firstly, let's do a train-valid split for our dataset, as we have already seen in Assignment 2. The `x_train` and `y_train` will be our training set, and `x_val`, `y_val` will be our validation set. Just to refresh your memory, training set is used for training to improve our model performance, while validation set tests the model with unseen data (recall from the lecture that they should be drawn from the same distribution). Testing our model with validation set is crucial especially when we want to know the generalization ability of the model. You will see more details later in the Overfitting & Ridge regression section.

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "50b9863a6dc10fa6275c9f973cfbc578", "grade": false, "grade_id": "cell-08de0d05761681ae", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Here we randomly shuffle the dataset and take $120$ $(80\%)$ data points as training set. The rest $30$ $(20\%)$ are used as validation set. There's also a useful method `sklearn.model_selection.train_test_split()` from scikit-learn for the same purpose. You can find its documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
# -

idx = list(range(0, 150))
random.shuffle(idx)
x_train = X[idx[:120]]
y_train = Y[idx[:120]]
x_val = X[idx[120:]]
y_val = Y[idx[120:]]


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9bc929fdd450b6e159d816771093413c", "grade": false, "grade_id": "cell-a0ed23a25e5d3cd1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Cost function
# The mean squared error (mse) is a common loss function that measures the average squared difference between predict and real values. Here, it is defined as:
# \begin{align}
# MSE = \frac{1}{2n}\sum^n_{i=1}(\text{y_pred}_i - \text{y_real}_i)^2
# \end{align}
# The $\frac{1}{2}$ in equation is just for convenience when computing the gradient (see next step). Recall that $y = X \Omega^T$, so we have:
# \begin{align}
# MSE = \frac{1}{2n}\sum^n_{i=1}(x_i \Omega^T - \text{y_real}_i)^2
# \end{align}
# Please implement the `mean_square_error()` that returns MSE. Methods/attribute you may need are `numpy.sum()` and `numpy.dot()`.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "6ea0f5da2f681fd28bc04ad03ac410ec", "grade": false, "grade_id": "MSE-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 4: implement MSE function. (2 points in total)

def mean_square_error(x, y, omegaT):
    """
    return the mean suqared error.
    
    Args:
        x (numpy.ndarray): numpy array of features (n data points, m features). Shape (n, m)
        y (numpy.ndarray): numpy array of corresponding targets. Shape (n, 1)
        omegaT (numpy.ndarray): weight vector (transposed). Shape (m, 1)
        
    Returns:
        mse (numpy.float64 or float): mean squared error.
    """
    mse = None # please update this in your solution
    #mse = np.mean((y - x.dot(omegaT))**2)
    mse = (1/(2*y.shape[0])) * np.sum(np.power((x.dot(omegaT) - y),2))
    #raise NotImplementedError()
    return mse

mean_square_error(x_train, y_train, omegaT) 

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2484a2406e271b051433fbbc93f0292e", "grade": true, "grade_id": "MSE-test_1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point: a simple test to check if your mse function works
a = np.array([[1.0, 2.0], [2.0, 2.0]])
b = np.array([[1.0], [2.0]])
w = np.array([[1.0, 1.0]]).T
result = mean_square_error(a, b, w)
assert result - 2.0 <= 0.01
assert type(result) == np.float64 or type(result) == float

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "49de407807de736a7cbe2c3d04ee985a", "grade": true, "grade_id": "MSE-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 1 point: further tests to check the output of the mse function

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3e31a05f980305a18350dba0e9f8ced1", "grade": false, "grade_id": "cell-0c61b6eda0caba8e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# After each epoch $t$, the Gradient Decent algorithm updates the weight vector in the direction of the negative gradient in order to reduce the cost function. The gradient is simply the partial derivative of mean squared error to the weight.
#
# You can deduce the derivative yourself for practice. Just run the following cell to render the answer:

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "80780dfa68eb98ce6a088765fdc79edc", "grade": false, "grade_id": "cell-4107228f483c16f2", "locked": true, "schema_version": 3, "solution": false, "task": false}
from IPython.display import display, Markdown

display(Markdown("\\begin{align}"
                 "\\frac{\\partial MSE}{\\partial \\Omega^T} &= "
                 "\\frac{1}{n} \\sum^n_{i=0}(x_i \\Omega^T - \\text{y_real}_i) x_i \\\\"
                 "\\Omega^T(t+1) &= \\Omega^T(t) - \\alpha \\frac{\\partial MSE}{\\partial \\Omega^T(t)}"
                 "= \\Omega^T(t) - \\frac{\\alpha}{n} \\sum^n_{i=0}(x_i \\Omega^T - \\text{y_real}_i) x_i"
                 "\\end{align}"))


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "2a47eb5d06296ce22aff6e1e7b74e134", "grade": false, "grade_id": "cell-3b243d3fd3dba44a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The $\alpha$ is the learning rate that controls the step size for the update after each iteration.
#
# Please implement the `weight_update_function()` that returns updated $\Omega^T$. Methods/attribute you may need are `numpy.dot()` and `numpy.reshape()`.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "1b82e245c712d3c3c19fac43c72af170", "grade": false, "grade_id": "Gradient_Descent-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 5: implement weight update function. (2 points in total)

def weight_update_function(x, y, omegaT, alpha):
    """
    return updated set of weights
    
    Args:
        x (numpy.ndarray): numpy array of features (n data points, m features). Shape (n, m)
        y (numpy.ndarray): numpy array of corresponding targets. Shape (n, 1)
        omegaT (numpy.ndarray): weight vector (transposed). Shape (m, 1)
        alpha (float): learning rate.
        
    Returns:
        omega_updated (numpy.ndarray): the updated weight vector. Shape (m, 1)
    """
    omega_updated = None # please update this in your solution
    omega_updated = omegaT - (alpha/y.shape[0]) * np.sum((x.dot(omegaT) - y) * x, axis=0).reshape(-1,1)
    #raise NotImplementedError()
    return omega_updated


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c225c91805a25ef61fb1523bdaa6c474", "grade": true, "grade_id": "Gradient_Descent-test_1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 points: check the output value and shape of your weight_update_function
result = weight_update_function(a, b, w, 0.001)
assert abs(np.mean(result) - 0.996) <= 0.01
assert result.shape == w.shape


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5945f61970177f021873074c8f573712", "grade": true, "grade_id": "Gradient_Descent-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 1 point: further tests to check the output of the weight_update_function function

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "88fb4084875e771a0f50675db6e2b1dd", "grade": false, "grade_id": "cell-7425ef1050a146d1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Training & plot
# To monitor the training process, here we use mean absolute error (mae):
# \begin{align}
# MAE = \frac{1}{n}\sum^n_{i=1} |\text{y_pred}_i - \text{y_real}_i|
# \end{align}
# Please implement the `mean_absolute_error()` that returns MAE. Methods/attribute you may need are `numpy.sum()`, `numpy.dot()` and `numpy.abs()`.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "ef7cea6643b284d747982d4a7cb13ec0", "grade": false, "grade_id": "MAE-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 6: implement mean absolute function. (2 points in total)

def mean_absolute_error(x, y, omegaT):
    """
    return the mean absolute error
    
    Args:
        x (numpy.ndarray): numpy array of features (n data points, m features). Shape (n, m)
        y (numpy.ndarray): numpy array of corresponding targets. Shape (n, 1)
        omegaT (numpy.ndarray): weight vector (transposed). Shape (m, 1)
        
    Returns:
        mse (numpy.float64 or float): mean absolute error.
    """
    mae = None # please update this in your solution
    mae = (1/y.shape[0]) * np.sum(np.abs(np.dot(x,omegaT) - y))
    #raise NotImplementedError()
    return mae


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "3bdf8639f9007b75d9a38bbb3617d6f2", "grade": true, "grade_id": "MAE-test_1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 points: check the output value and shape of your mean_absolute_error function
result = mean_absolute_error(a, b, w)
assert np.round(result)  == 2.0
assert type(result) == np.float64 or type(result) == float

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "eb976ece48b75f53773fd868fccb0bbe", "grade": true, "grade_id": "MAE-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 1 point: further tests to check the output of the mean_absolute_error function

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0cc44d69ab7bd4ebe63f3a7d333609ee", "grade": false, "grade_id": "cell-851877d35d2dd051", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For the last step before training the model, let's set up the learning rate and number of iterations. We use `J_train` and `J_val` to record mean absolute errors for each epoch of the training and validation processes.
# -

epochs = 15000 # number of updates to weights, aka omegaT
alpha = 0.001 # learning rate
J_train = np.zeros(epochs) # MAE of training process
J_val = np.zeros(epochs) # MAE of validation process

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "21cc4278a129505b0d3151588367fce1", "grade": false, "grade_id": "cell-13d06de7e4e3c1f5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now, let's train our model by updating `omegaT` with the `weight_update_function()` and record mean absolute errors through `mean_absolute_error()` for the training/validation process.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "8572967d641fe3314adfe303c13d4a19", "grade": false, "grade_id": "Train_Loop-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 7: implement the training process. (3 points in total)

np.random.seed(0)
omegaT = np.random.randn(1,5).T # initialize weight/bias matrix randomly

# begin training
for i in range(epochs):
    """
    for each epoch, record the mae for training and validation process
    use the weight_update_function to update the omega
    """
    J_val[i] = None # please update this in your solution
    J_train[i] = None # please update this in your solution
    # don't forget to update the omegaT
    J_val[i] = mean_absolute_error(x_val, y_val, omegaT)
    J_train[i] = mean_absolute_error(x_train, y_train, omegaT)
    omegaT = weight_update_function(x_train, y_train, omegaT, alpha)
    #raise NotImplementedError()

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "69cdefa6ea64ee1e37474a7606e4d6a0", "grade": true, "grade_id": "Train_Loop-test_1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# 2 point: check the results of the training process 
assert J_train[-1] > 0
assert np.std(J_val[-10:]) <= 1e-3 # the training process should converge


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "36c844b8f2a397e04ecf34196b70c5e7", "grade": true, "grade_id": "Train_Loop-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden tests below
# 1 point: further tests for the training/validation results

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9355fea4e3327f217dcc560b627f4b45", "grade": false, "grade_id": "cell-ae74de811a5efbf3", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Here the `plot_training_curve()` is implemented to visualize the training process.
# -

def plot_training_curve(MAE_train, MAE_val, epochs):
    """Plot the mean absolute error for training/validation process"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(np.arange(epochs), MAE_train, label='Training')
    ax.plot(np.arange(epochs), MAE_val, label='Validation')
    ax.set_ylim([0,1])
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Epochs")
    ax.set_title("Mean Absolute Error vs Epochs")
    ax.legend(loc='upper right')
    plt.show()


plot_training_curve(J_train, J_val, epochs)

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "66afd65476359b4065568e97a474b940", "grade": false, "grade_id": "cell-8f5ede23cbae5e06", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Question:** You may have noticed that we have implemented the `mean_square_error()` function but seemed to never use it in the training process. Please think about the reason and write it down here.
# Hint: double check the deduce process of the Gradient Decent.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "6693fad6d883f633fc4c1980fc0d2557", "grade": false, "grade_id": "mse_question-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 8: answer the question. (1 point in total)

your_answer = (
    '''
    Explicitly computing the Mean Squared Error (MSE) is unnecessary, as it suffices to determine
    the gradient of the MSE functions with respect to the weights. By selecting the direction of
    steepest descent, the MSE can be minimized automatically without the need for explicit calculation.
    '''
)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8d07b9f2b307b30217aa34ec7ed61b52", "grade": true, "grade_id": "mse_question-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point for the answer
assert type(your_answer) == str
assert len(your_answer) > 1


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8743b24d2e9f9e65372b94b56895060d", "grade": false, "grade_id": "cell-d1d5ee5a2c54fcc1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Accuracy
# Now, with the optimized weight vector, let's implement `prediction()` to get the predicted class labels from our linear model. You may need `numpy.dot()` for the computation, and `numpy.round()` to round up the results into integers (because our class labels are integer 0, 1, 2).

# + deletable=false nbgrader={"cell_type": "code", "checksum": "03334e304acb0d257af24f02b17a339c", "grade": false, "grade_id": "Prediction-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 9: implement prediction function. (1 point in total)

def prediction(x, omegaT):
    """
    Return predicted labels.
    
    Args:
        x (numpy.ndarray): the feature matrix (n data points, m features). Shape (n, m)
        omegaT (numpy.ndarray): optimized weight vector (transposed). Shape (m, 1) 
        
    Returns:
        y_pred (numpy.ndarray): predicted labels. Shape (n, 1)
    """
    y_pred = None # please update this in your solution
    y_pred = np.abs(np.round(np.dot(x,omegaT),0))
    return y_pred


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "26b03364bb1ea08c392487446db8e6e9", "grade": true, "grade_id": "Prediction-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point
Y_pred = prediction(X, omegaT)
assert Y_pred.shape == Y.shape


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "23cadf0bc3dbc578e618cd3b67711cc8", "grade": false, "grade_id": "cell-3ff8c071f6a86c8b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# With `Y_pred`. Please implement `accuracy()` that calculates the prediction accuracy.
#
# Hint: prediction accuracy is computed by counting the number of predictions that match the real values, and then devide by the number of total instances.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "44509117dab6f43bd0e45da3d5dd5ca5", "grade": false, "grade_id": "Accuracy_function-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 10: implement accuracy function. (1 point in total)

def accuracy(y_pred, y):
    """
    Compute the accuracy of prediction.
    
    Args:
        y_pred (numpy.ndarray): the predicted class labels for n data points. Shape (n, 1)
        y (numpy.ndarray): the real class labels. Shape (n, 1)
        
    Returns:
        acc (numpy.float64 or float): calculated accuracy in float.
    """
    acc = None # please update this in your solution
    acc = np.sum(y_pred == y)/y.shape[0]
    return acc


# -

acc = accuracy(Y_pred, Y)
acc

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f393e6ee564eda293ee59f324c2e1469", "grade": true, "grade_id": "Accuracy-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
assert type(acc) == np.float64 or type(acc) == float

# Hidden test below
# 1 point: test the output value of the accuracy() function

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e2b70debd6883360e007b365656c4e64", "grade": false, "grade_id": "cell-cfd646f9a9655148", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Visualize predictions vs ground truth
# For the last step, let's visualize our predictions v.s. the real labels.
# -

def plot_predict_vs_real(Y_pred, Y):
    """Plot the predictions vs ground truch"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(np.arange(1, 151, 1), Y_pred, label='Predictions')
    ax.plot(np.arange(1, 151, 1), Y, label='Ground Truch', color='red')
    ax.legend(loc="upper left")
    plt.yticks(np.arange(0, 3, 1))
    plt.show()


plot_predict_vs_real(Y_pred, Y)


# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "77a6d3a61915f65eb5c6d77f51c827ba", "grade": false, "grade_id": "cell-165d84e42ab77f0d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Please feel free to play around with the model and training process. You may adjust the learning rate and number of epochs to see how the trainig curve changes (do not forget to re-initialize the weight vector). You may also try different train-validation split ratio to see the effect.
#
# You may notice that the training results differ a lot with different hyperparameters (learning rate, omega initialization, etc.). This is one limitation of such simple ML model. Later in this semester, you will see and implement more sophisticated ML algorithms that can make more stable and accurate predictions.

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "28746d6c2e44b1f4af8c15eb64e4157c", "grade": false, "grade_id": "cell-926be4a613646f05", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Overfitting & Ridge regression

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b8cd31d89c809cdb62e872fdbe6d3f75", "grade": false, "grade_id": "cell-59d2a351bd90a39f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# When finishing this assignment, you may (and may not, depending on the training set setup) sometimes observe the gap between the training curve and validation curve (validation error higher than training error) that cannot be diminished by increasing the number of epochs. This phenomenon is known as "overfitting". It happens when the model gets too complex so that it fits the training set perfectly, but loses the generalization ability towards unseen data from validation/test set.

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5c4351520639ec67d353bb9104358a0c", "grade": false, "grade_id": "cell-0f409bfd47dc63d7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Ridge regression, also known as L2 regularization, is a useful technique to restrict our model from getting too complicated and reduce the effect of overfitting. It works by simply adding a penalty term to the cost function. In our case, the penalty term is $||\Omega||_2^2$ (the square of the $L^2$ norm). It prefers lower absolute values of weights thus reduce the model complexity. For more information, please refer to [here](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization).

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ff6433bf2ccf391f99fb8f06cab328f8", "grade": false, "grade_id": "cell-a2fef35e4cc07408", "locked": true, "schema_version": 3, "solution": false, "task": false}
# By introducting the penalty term, our cost function becomes:
# \begin{align}
# MSE = \frac{1}{2}(\frac{1}{n}\sum^n_{i=1}(\text{y_pred}_i - \text{y_real}_i)^2 + \lambda ||\Omega||_2^2)
# \end{align}
# $\lambda$ is the coefficient to adjust the regularization effect.

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5a17349b83bfd3bbf61de6e864080d53", "grade": false, "grade_id": "cell-6e8209261b08ce72", "locked": true, "schema_version": 3, "solution": false, "task": false}
# While the mean absolute error calculation stays the same, the weight update function becomes:
# \begin{align}
# \Omega = \Omega - \alpha \frac{\partial MSE}{\partial \Omega} = \Omega - \alpha(\frac{1}{n} \sum^n_{i=0}(x_i \Omega^T - \text{y_real}_i) x_i + \lambda ||\Omega||_2) 
# \end{align}
# Now, please implement the new cost function `mean_square_error_ridge()` and `weight_update_function_ridge()` with the penalty term. You may use `mean_square_error()` and `weight_update_function()` that are implemented earlier as references. You may also need `np.square` and `np.sqrt` to calculate square and square root.

# + deletable=false nbgrader={"cell_type": "code", "checksum": "7b91e085e897bbca3e5687634a43e285", "grade": false, "grade_id": "Ridge_function-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 11: implement L2 regularization. (2 points in total)

def mean_square_error_ridge(x, y, omegaT, lam):
    """
    return the mean suqared error.
    
    Args:
        x (numpy.ndarray): numpy array of features (n data points, m features). Shape (n, m)
        y (numpy.ndarray): numpy array of corresponding targets. Shape (n, 1)
        omegaT (numpy.ndarray): weight vector (transposed). shape (m, 1)
        lam (float): ridge regression coefficient.
        
    Returns:
        mse (numpy.float64 or float): mean squared error.
    """
    mse = None # please update this in your solution
    mse = 1/2*(1/y.shape[0]) * (np.sum(np.square(np.dot(x,omegaT) - y) + lam * np.square(np.linalg.norm(np.transpose(omegaT)))))
    #raise NotImplementedError()
    return mse


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "511ac04243ed6fa720369bd73b9be6fb", "grade": true, "grade_id": "Ridge_function-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point: check the output value
a = np.array([[1.0, 2.0], [2.0, 2.0]])
b = np.array([[1.0], [2.0]])
w = np.array([[1.0, 1.0]]).T
assert np.round(mean_square_error_ridge(a, b, w, 0.1), 1) == 2.1


# + deletable=false nbgrader={"cell_type": "code", "checksum": "94b9a89f91bf418522f8a1a480d025e6", "grade": false, "grade_id": "Ridge_update_function-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
def weight_update_function_ridge(x, y, omegaT, alpha, lam):
    """
    return updated set of weights
    
    Args:
        x (numpy.ndarray): numpy array of features (n data points, m features). Shape (n, m)
        y (numpy.ndarray): numpy array of corresponding classes. Shape (n, 1)
        omegaT (numpy.ndarray): weight vector (transposed). Shape (m, 1)
        alpha (float): learning rate.
        lam (float): ridge regression coefficient.
        
    Returns:
        omega_updated (numpy.ndarray): the updated weight vector (transposed). Shape (m, 1)
    """
    omega_updated = None # please update this in your solution
    n = x.shape[0]
    omega = np.transpose(omegaT)

    omega_updated = omega - alpha* (1/n * np.sum((np.dot(x,omegaT) - y) * x , axis=0) + lam * np.linalg.norm(omega))
    omega_updated = np.transpose(omega_updated)
    
    return omega_updated


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "feb712f502586a6f603867a328caa7c1", "grade": true, "grade_id": "Ridge_update_function-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# 1 point: check the output value
assert abs(np.mean(weight_update_function_ridge(a, b, w, 0.001, 0.1)) - 0.99) <= 0.01
assert w.shape == weight_update_function_ridge(a, b, w, 0.001, 0.1).shape

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "228944becea614c63a4c7c5a6d4551e2", "grade": false, "grade_id": "cell-32d88540dd3063f2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now, let's look at a very unbalanced training set. In this example, only 25 instances are used as training set ($17\%$), and the number of class 0 is significantly higher than both class 1 and 2.
# -

import pickle
with open('iris_overfit', 'rb') as f:
    x_train, x_val, y_train, y_val = pickle.load(f)

fig, ax = plt.subplots()
ax.scatter(np.arange(0, len(y_train), 1), y_train)
plt.yticks(np.arange(0, 3, 1))
ax.set_xlabel("data index")
ax.set_ylabel("class label")
ax.set_title("Class distribution of training set")
plt.show()

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "09963cd951855227a5e7c42432197740", "grade": false, "grade_id": "cell-cdee0339af40f024", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For the training, let's do the normal linear regression and ridge regression side by side, with `J_train`, `J_val` recording linear regression training/validation MAE, and `J_train_ridge`, `J_val_ridge` recording ridge regression MAE. Please implement them in the same `for` loop.
# -

np.random.seed(0)
omegaT = np.random.randn(5,1)
epochs = 3000 # number of updates to the weight
alpha = 0.0005 # learning rate
lam = 0.5 # coefficient of L2 penalty
J_train = np.zeros(epochs) # record MAE of training process (without L2 penalty)
J_val = np.zeros(epochs) # record MAE of validation process (without L2 penalty)
J_train_ridge = np.zeros(epochs) # record MAE of training process (with L2 penalty)
J_val_ridge = np.zeros(epochs) # record MAE of validation process (with L2 penalty)

# + deletable=false nbgrader={"cell_type": "code", "checksum": "10c232ad8a4dfc226833ee2dd9ce921e", "grade": false, "grade_id": "Train_Loop_Ridge-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 12: implement training process with L2 regularization. (3 points in total)

omega_norm = np.copy(omegaT) # weight vector for training without L2 penalty
omega_ridge = np.copy(omegaT) # weight vector for ridge regression

for i in range(epochs):
    # for each epoch, record the mae for training and validation process without using ridge regression
    # use the weight_update_function to update the omega_norm
    J_val[i] = None # please update this in your solution
    J_train[i] = None # please update this in your solution
    
    J_val[i] = mean_absolute_error(x_val, y_val, omega_norm)
    J_train[i] = mean_absolute_error(x_train, y_train, omega_norm)
    omega_norm = weight_update_function(x_train, y_train, omega_norm, alpha)
    
    # Now, let's try to optimize omega using ridge regression
    # use the weight_update_function_ridge to update the omega_ridge
    J_val_ridge[i] = None # please update this in your solution
    J_train_ridge[i] = None # please update this in your solution
    
    J_val_ridge[i] = mean_absolute_error(x_val, y_val, omega_ridge)
    J_train_ridge[i] = mean_absolute_error(x_train, y_train, omega_ridge)
    omega_ridge = weight_update_function_ridge(x_train, y_train, omega_ridge, alpha, lam)   
    
    #raise NotImplementedError()

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "af678bb3c62b3e741c7e21b95262f235", "grade": true, "grade_id": "Train_Loop_Ridge-test_1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# 2 point: check the results of the training process
assert J_train_ridge[-1] > 0
assert np.std(J_val_ridge[-10:]) <= 1e-3 # the model should converge after training


# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d6b45c5a65a271a06b1b94c5914d8500", "grade": true, "grade_id": "Train_Loop_Ridge-test_2", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 1 point: further tests of the training results

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6b13bdefa45e339152180be31a55445b", "grade": false, "grade_id": "cell-edcd9970ea4609f4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now we plot the training curves again, but this time on a log-log scale to better see the differences:
# -

def plot_training_curve_ridge(MAE_train, MAE_val, MAE_train_ridge, MAE_val_ridge, epochs):
    """Plot the mean absolute error for training/validation process"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].loglog(np.arange(epochs), MAE_train, label='Training')
    ax[0].loglog(np.arange(epochs), MAE_val, label='Validation')
    ax[0].set_ylim([0,1])
    ax[0].set_ylabel("Mean Absolute Error")
    ax[0].set_xlabel("Epochs")
    ax[0].set_title("Without L2 Regularization")
    ax[0].legend(loc='upper right')
    
    ax[1].loglog(np.arange(epochs), MAE_train_ridge, label='Training')
    ax[1].loglog(np.arange(epochs), MAE_val_ridge, label='Validation')
    ax[1].set_ylim([0,1])
    ax[1].set_ylabel("Mean Absolute Error")
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("With L2 Regularization")
    ax[1].legend(loc='upper right')
    plt.show()


plot_training_curve_ridge(J_train, J_val, J_train_ridge, J_val_ridge, epochs)

Y_pred = prediction(X, omega_norm)
Y_pred_ridge = prediction(X, omega_ridge)
acc = accuracy(Y_pred, Y)
acc_ridge = accuracy(Y_pred_ridge, Y)
print(acc)
print(acc_ridge)

plot_predict_vs_real(Y_pred, Y)
plot_predict_vs_real(Y_pred_ridge, Y)

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "da6badb3722e34bf4e12dc1d1b4f15af", "grade": false, "grade_id": "cell-11f395039356b11f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Yes or No question:** from the plots/accuracy calculations, do you think the ridge regression improved the learning performance?

# + deletable=false nbgrader={"cell_type": "code", "checksum": "bafd88bdf89f683fda4d33892185fe92", "grade": false, "grade_id": "Ridge_question-answer", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Task 13: answer the question. (1 point in total)

Answer = (
    "Yes"
    #raise NotImplementedError()
)

# + deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e29c84e92b3739c4e52045c0376b86be", "grade": true, "grade_id": "Ridge_question-test", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden test below
# 1 point for your answer

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5055c03f08fd98b5f02ae7aab78e47f8", "grade": false, "grade_id": "cell-298117388b266b67", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Scipy Optimize

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "06d235aa081fe52b39148d8369007e09", "grade": false, "grade_id": "cell-1e16607a3299687d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Rather than handcraft the Gradient Decent algorithm, there are libraries that can do the optimization task automatically. The `minimize()` function provided by SciPy library is a good example. It takes as input the name of the function that needs to be minimized, the initial point where the search starts and (optionally) the name of a specific search algorithm. It returns a `OptimizeResult` object that contains details of the optimization result. Below is an example of using `minimize()` function to optimize our linear model.

# +
from scipy.optimize import minimize

np.random.seed(0)
omega = np.random.randn(1,5)

def mean_square_error(omega):
    return (1/(2*len(X)) * np.sum((np.dot(X, omega.reshape(1,5).T) - Y) ** 2 ))


res = minimize(mean_square_error, omega, method='L-BFGS-B')
print(res)
omega_op = np.array(res.x).reshape(5,1)
# -

Y_pred = prediction(X, omega_op)
acc = accuracy(Y_pred, Y)
acc

plot_predict_vs_real(Y_pred, Y)

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "02ec63fee222f86896d40a010e15a67f", "grade": false, "grade_id": "cell-11e0b1c3d98cf009", "locked": true, "schema_version": 3, "solution": false, "task": false}
# You can find a good tutorial of SciPy optimization [here](https://machinelearningmastery.com/function-optimization-with-scipy/).  
#
# If you have more questions, please refer to the SciPy [documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html).
#
# For the introduction of the "L-BFGS-B" search algorithm using here, please read [this](http://sepwww.stanford.edu/data/media/public/docs/sep117/antoine1/paper_html/node6.html).
#
# ***If you want to play around with this library, please do this in a separate notebook that is not submitted!***

# + [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "080ee7ded5103ae39977e143e7202939", "grade": false, "grade_id": "cell-4c22e42798de2a83", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Thank you very much for participating the exercise! I hope you find it helpful to understand math basics in machine learning!
#
# Please let us know if you have any thought/comment/suggestion to this exercise by taking the exercise evaluation. Your feedback is valuable to us and is highly appreciated!
# -




# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "87b1431e25aa9fa7e9ea168152b7a436", "grade": false, "grade_id": "cell-6fa1bcf98952e899", "locked": true, "schema_version": 3, "solution": false, "task": false}
#
# # Exercise Sheet No. 6
#
# ---
#
# > Machine Learning for Natural Sciences, Summer 2023, Jun.-Prof. Pascal Friederich, pascal.friederich@kit.edu
# > 
# > Deadline: June 5th 2023, 8:00 am
# >
# > Container version 1.0.1
# >
# > Tutor: patrick.reiser@kit.edu
# >
# > **Please ask questions in the forum/discussion board and only contact the Tutor when there are issues with the grading**
# ---
#
# ---
#
# **Topic**: This exercise sheet will focus on feed-forward neural networks, their implementation and training, as well as an application to materials science.
#

# %% [markdown]
# Please add here your group members' names and student IDs. 
#
# Names: 
#
# IDs:

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9fc6c67a5b72b907a3c10371fa0e5abe", "grade": false, "grade_id": "cell-3236a919b585de02", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Application
#
# Next week you will start learning about applications of NNs in materials science. To give you a little insight, we will use an application in materials science already here:
#
# # Organic Solar Cells
# For organic materials to become semi-conducting, electrons must be delocalized in the molecule. For electrons to be delocalized, a high level of conjugation is necessary:  
# When single and double bonds are alternating in an organic molecule, electrons can move. When we think about an aromatic ring, like benzene, it is not defined where the double bonds would form, so they can move around the ring and are delocalized along the whole aromatic system:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ef4251792d2543a6430fb65b29792637", "grade": false, "grade_id": "cell-1cd6d0e1688214d5", "locked": true, "schema_version": 3, "solution": false, "task": false}
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

mol = Chem.MolFromSmiles('c1ccccc1')
mol

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "914c87520a2072c223eab86a4b8cc844", "grade": false, "grade_id": "cell-ad8bd72c4fa018c6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# These electrons have higher energies and are part of what is called the highest occupied molecular orbit (HOMO). This is basically the equivalent of the valence band in classical semiconductors.  
#
# The equivalent of the conduction band is the lowest level at higher energies that is unoccupied or the lowest unoccupied molecular orbit (LUMO). The gap between those two levels is the bandgap of organic semiconductors.
#
# For organic photovoltaic cells this bandgap needs to be small enough so that visible light can excite an electron from the HOMO to the LUMO. This requires a high level of conjugation and hence aromatic systems (Figure 1)
#
# <a title="Alevina89, CC BY-SA 3.0 &lt;https://creativecommons.org/licenses/by-sa/3.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Homo-lumo_gap.tif"><img width="512" alt="Homo-lumo gap" src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Homo-lumo_gap.tif/lossless-page1-607px-Homo-lumo_gap.tif.png"></a>
# <p style="text-align:center;font-size:80%;font-style:italic">
#     Figure 1: Conjugation induced LUMO reduction.
# </p>
#
# For the development of organic semiconductors, HOMO and LUMO can be simulated by density functional theory (DFT) using different levels of theory. You will learn about this in later lectures. All you need to know now, is that depending on the level of theory  - or better the details of the approximations taken - the calculation of properties like HOMO and LUMO from the molecule can take hours.
#
# This is a problem for high-throughput screening if one wants to discover new materials. It is extremely costly to evaluate e.g. 100,000 molecules for their properties with methods that are precise enough. Hence, one usually tries to do a detailed simulation only on a subset of molecules and then train a ML-model to predict the properties of interest on the labeled data.
#
# ## Dataset
# The dataset contains the simulated LUMO and HOMO values for 51,247 organic molecules. Additionally, it contains 63 molecule descriptors that were calculated using [rdkit](https://www.rdkit.org/docs/index.html):

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "08f95d1380082c2200df08828984c752", "grade": false, "grade_id": "cell-868d8be608b5077e", "locked": true, "schema_version": 3, "solution": false, "task": false}
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_hdf('OPV.h5')
df.describe()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "633636d6cac62ef0484bb5972c74ef3d", "grade": false, "grade_id": "cell-42ef56ae22c7077a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The dataframe has a two-level column index so you can index the labels by calling `df['labels']`:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4d0e91dfe3c2384ab4e53d11aa8dbdeb", "grade": false, "grade_id": "cell-996e09cffe1ea828", "locked": true, "schema_version": 3, "solution": false, "task": false}
df['labels'].hist(bins=100)
plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3a5a3615b53471b9b911b0ddb1e7641d", "grade": false, "grade_id": "cell-cff397671bc615ac", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Linear regression benchmark
# For a first shot we can train a ridge regression to predict the `labels` from the `mol_descriptors`.
#
# This time we will use `sklearn`. Additionally we will also pre-process the features with a standard scaler to shift them to zero mean and scale them to unit variance.
#
# As already mentioned, the calculation of DFT properties can be very costly. Hence, we want to have a model that extrapolates well to unseen data. Hence, we will use only 20% of the dataset for training and test on 80%.
#
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4144c4650fe99794fb819743ad51fc90", "grade": false, "grade_id": "cell-200cc37a5b72eb23", "locked": true, "schema_version": 3, "solution": false, "task": false}
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df['mol_descriptors'].values
y = df['labels'].values

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ebbd6b8923aa3df805a0742916705d33", "grade": false, "grade_id": "cell-5c747b57ec6724fa", "locked": true, "schema_version": 3, "solution": false, "task": false}
# First use the [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to scale the features (`X`) to mean of 0 and unit variance:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "eea93bf92b9f5d8b68c9c2c751580c46", "grade": false, "grade_id": "cell-c9f2df20a8d0cf01", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Assign a StandardScaler object to scaler and obtain the scaled fatures as X_scaled
scaler = None
X_scaled = None

# YOUR CODE HERE
raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2c8a5ab885363529eebab69020604548", "grade": true, "grade_id": "Scaler-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Scaler - 1 point

assert isinstance(scaler, StandardScaler), "The scaler should be an instance of the sklearn StandardScaler"
np.testing.assert_almost_equal(np.mean(X_scaled), 0, 10)
np.testing.assert_almost_equal(np.var(X_scaled, axis=0).sum(), 63.)
# Possible hidden tests

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "2c3b0c0a2f1b58b77004de8fae20133b", "grade": false, "grade_id": "cell-d0ed310e0c9ef260", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next we use the [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test_split#sklearn.model_selection.train_test_split) to split of 20% of `X_scaled` and `y` as train set and use the rest as test set:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6a78eed9979bc872c50db84a260c8a5c", "grade": false, "grade_id": "cell-6e293b67656d2a39", "locked": true, "schema_version": 3, "solution": false, "task": false}
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=0)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b29110c13e1d3ff7772eed7a5d1796a4", "grade": false, "grade_id": "cell-eabac20dab75b13d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now use the [`Ridge()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge) model to fit a ridge regression with standard parameters to the train data, and assign the predictions of the fitted model on the test data to `y_pred`:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "2e3006652180e3013f24e9b708880ca4", "grade": false, "grade_id": "cell-b1b48084492b70d2", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Instantiate a Ridge() model as model, fit it and assign the predictions to y_pred
model = None
y_pred = None

# YOUR CODE HERE
raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e122ed242136664f92b33a73a71321f0", "grade": false, "grade_id": "cell-69dfde6ff482f06b", "locked": true, "schema_version": 3, "solution": false, "task": false}
r2_lumo_ridge = r2_score(y_test[:, 0], y_pred[:, 0])
r2_homo_ridge = r2_score(y_test[:, 1], y_pred[:, 1])
print(f'R2 LUMO: {r2_lumo_ridge}\nR2 HOMO: {r2_homo_ridge}')

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "438eb3187e157479740bea551785acab", "grade": true, "grade_id": "Ridge_Regression-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Ridge Regression - 1 point

assert y_pred.shape[0] == y_test.shape[0]
assert y_pred.shape[1] == y_test.shape[1]
assert r2_lumo_ridge > 0.70
assert r2_homo_ridge > 0.78
# Possible hidden tests

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1f58d0ad84f10801c84341ddf6c449e2", "grade": false, "grade_id": "cell-372ca63030fe0b3d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now we can plot the correlations of the the true and predicted values:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "356572c32d7c3285dcb85abbc39f2542", "grade": false, "grade_id": "cell-0a06841aa21a0f40", "locked": true, "schema_version": 3, "solution": false, "task": false}
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.2, s=5)
axs[0].plot([-6, -1], [-6, -1], 'k')
axs[0].set_title(f'R2 LUMO: {r2_lumo_ridge}')
axs[0].set_xlabel('true LUMO')
axs[0].set_ylabel('predicted LUMO')
axs[0].set_xlim([-6, -1])
axs[0].set_ylim([-6, -1])

axs[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.2, s=5)
axs[1].plot([-9, -3], [-9, -3], 'k')
axs[1].set_title(f'R2 HOMO: {r2_homo_ridge}')
axs[1].set_xlabel('true HOMO')
axs[1].set_ylabel('predicted HOMO')
axs[1].set_xlim([-9, -3])
axs[1].set_ylim([-9, -3])

plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f47be03e3b8b9db3a3c4ce79daa55bd1", "grade": false, "grade_id": "cell-b045f7ffe8f279f8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# As you can see, the ridge regression learns something, but the predictions are still way off from the true values.
#
# Hence, we will apply a non-linear model, namely a:
#
# # Neural Network
#
# ## Inspiration from Neuroscience
# In this week's lecture you saw that feedforward neural networks can be thought of as 'function approximation machines' [Goodfellow et al. 2016](https://www.deeplearningbook.org/contents/mlp.html).
# Some ideas of todays artificial neural networks are loosely inspired by neuroscientific models of biological networks.
# For a different perspective we can try to make a connection to models of biological neural networks (while remembering the goal of todays neural networks is not to get a model of the brain but rather to approximate some function).
#
# The fundamental units of the brain are neurons which can also be thought of as information messengers. Neurons receive, transmit and transform information in the form of electrical impulses.
# A neuron integrates information from its upstream neurons at the dendrites, creating a post-synaptic potential. 
# At the axon hillock this potential is encoded into a spike train of action potentials. These are sent down the axon until they reach a synapse of an axon terminal. At the synapse the spike train is again decoded into a pre-synaptic potential and depending on the incoming spike frequency more or less neurotransmitters are released to pass the signal on to the next neuron (Figure 2).
# <p style="text-align:center;font-size:80%;font-style:italic">
#     <img src="https://images.topperlearning.com/topper/tinymce/imagemanager/files/Synapse_between_Neurons.jpg", width="70%">
#     <br>
#     Figure 2: Simplified sketch of a neuron.
# </p>
#
#
# ## Forward Pass
#
# For todays artificial neural networks this was simplified in the following way:  
# Instead of simulating spikes, we only simulate a spike frequency as a continuous value. Depending on the post-synaptic potential, this frequency can be higher or lower. E.g. neurons do not respond at all until a specific post-synaptic potential is reached and also have a maximum spike-frequency. This implies two things:  
# **1. The transformation from post-synaptic potential to spike-frequency is non-linear. (activation function)**  
# **2. Each neuron has a different base activity. (bias)**
#
# In the following mathematical definitions, lower-case letters denote scalars, while upper-case letters denote vectors or matrices. No dot or $\cdot$ denotes a dot product of vectors or matrices.
#
# ### Single neuron
# Given a subjected neuron $l$ receives inputs from an upstream layer $k$ and the upstream layer consists of three neurons with the outputs $x_1, x_2, x_3$ via connections with weights $\theta_{1,l}, \theta_{2,l}, \theta_{3,l}$ (Figure 3), we can compute the weighted sum as state $h_l(X)$ of the subjected neuron by linear algebra as the product of the row vector $X_k$ and the column vector $\Theta_{k,l}$. Additionally, we add a bias $b$ to the neuron.
#
# \begin{align}
# h_l(X) &= X_k \cdot \Theta_{k,l} + b_l\\
# &=
# \begin{bmatrix}
#     x_1 & x_2 & x_3
# \end{bmatrix} 
# \cdot
# \begin{bmatrix}
#     \theta_{1,l} \\ \theta_{2,l} \\ \theta_{3,l}
# \end{bmatrix}
# + b_l\\
# &=x_1 \theta_{1,l} +  x_2 \theta_{2,l} + x_3 \theta_{3,l} + b_l
# \end{align}
#
# And the output or activation $a_l(x)$ of the subjected neuron in terms of spike-frequency is computed using a non-linear activation function $\sigma()$:
#
# \begin{align}
# a_l(X) &= \sigma(h_l(X))
# \end{align}
#
# <div>
# <img src="attachment:ff_b1.png" width="30%"/>
# </div>
# <p style="text-align:center;font-size:80%;font-style:italic">
# Figure 3: A neuron layer k with three neurons has feed-forward connections to a single neuron l.
# </p>

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1b73b00204c130c25d426c5b774fd2c1", "grade": false, "grade_id": "cell-65de2b486b4e7331", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Neuron Layer
# Now given that the subjected $l$ is not only a single neuron but e.g. a layer of two neurons (Figure 4), the notation is the same with the only difference that we add a column to the weight column vector $\Theta_{k,l}$ changing it to a $3 \times 2$ dimensional matrix to calculate the $1 \times 2$ dimensional row state vector $H_l(X)$:
#
# \begin{align}
#     H_l(X) &= X_k \cdot \Theta_{k,l} + B_l\\
#     &=
#     \begin{bmatrix}
#         x_1 & x_2 & x_3
#     \end{bmatrix} 
#     \cdot
#     \begin{bmatrix}
#         \theta_{1,1} & \theta_{1,2} \\
#         \theta_{2,1} & \theta_{2,2} \\
#         \theta_{3,1} & \theta_{3,2} 
#     \end{bmatrix}
#     +
#     \begin{bmatrix}
#         b_1 & b_2 
#     \end{bmatrix} \\
#     &=
#     \begin{bmatrix}
#         \theta_{1,1} x_1 + \theta_{1,2} x_2 + \theta_{1,3} x_3 + b_1 &
#         \theta_{2,1} x_1 + \theta_{2,2} x_2 + \theta_{2,3} x_3 + b_2
#     \end{bmatrix}
# \end{align}
#
# And the activation of the layer:
#
# \begin{align}
# A_i(X) &= \sigma(H_i(X))
# \end{align}
#
# <div>
# <img src="attachment:ff_b2.png" width="30%"/>
# </div>
# <p style="text-align:center;font-size:80%;font-style:italic">
# Figure 4: A neuron layer j with three neurons has feed-forward connections to a neuron layer i with two neurons.
# </p>
#
# For the bias there exist two approaches:
# 1. One can see the bias as an additional input with a constant $1$ from the previous layer and a learnable weight vector.
# 2. Adding the bias simply as row vector that added to the neuron state $H(x)$.
#
# Here we use approach 2.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "077dafa4e12dc6c4e8af45f09d1543e3", "grade": false, "grade_id": "cell-635ae832ad50c9c8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Batches
# Usually data is processed in batches, so not only one single sample is processed at a time but multiple samples as this is much faster when calculated using vectorization. In our case, samples are rows with the length of the features. So when we pass 10 samples of our features (the `mol_descriptors`), we pass a matrix of shape $10 \times 63$. This already fits well with our current notation and it will return a $10 \times 32$ state matrix.
#
#
# ### Weight initialization
# To compute anything, the weights must be different from $0$. There are several approaches to initialize weights. We will use the approach of [Glorot et Al., 2010](http://proceedings.mlr.press/v9/glorot10a.html).
# The initial weights are drawn form a uniform distribution $U(-z, z)$ with the limit $z$ calculated as:
# \begin{align}
#     z &= \sqrt{\frac{6}{k+l}}
# \end{align}
# ...with input length $k$ and output length $l$.
#
#
# Just to get an idea how this all works in python we will build a layer with 32 neurons and feed all `mol_descriptors`.  
# Initialize the weight matrix `theta` using numpy functions:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5fbecf8fb397c63e44ce1eb5bffa3f56", "grade": false, "grade_id": "cell-6da9b09109577066", "locked": true, "schema_version": 3, "solution": false, "task": false}
sample = X_train[0:10]
n_inputs = sample.shape[1]
n_outputs = 32

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "e12c0498043d82f9f0df88e867892a89", "grade": false, "grade_id": "cell-31a3010d3f198150", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Initialize the weights theta after Glorot et Al. 2010 (glorot uniform):
theta = None
# YOUR CODE HERE
raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "249e1c571db8c4153d3bfd53d467a62e", "grade": true, "grade_id": "Weights-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Weights - 1 point

assert theta.shape[0] == n_inputs, "Your weight shape doesn't match!"
assert theta.shape[1] == n_outputs, "Your weight shape doesn't match!"

# Hidden asserts for the limits of the uniform distribution

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0a8d40b239a972514487a815cce117a4", "grade": false, "grade_id": "cell-6f705f2c4c9570d4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now calculate h(x) for the sample using your weights:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "9b6f77c2eaeb20c4ad68cd763fa03328", "grade": false, "grade_id": "cell-e47a93a7f3cc2876", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Initialize the bias row vector with zeros:
b = None
# YOUR CODE HERE
raise NotImplementedError()

# Calculate h(x) using sample, theta and b
# The bias won't change anything but it will check for correct shapes.
h = None
# YOUR CODE HERE
raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6dc5b9220c0b0d8f8f843afe4da33f68", "grade": true, "grade_id": "States-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# States - 2 points

assert h.shape[0] == sample.shape[0]
assert h.shape[1] == n_outputs
# Possible hidden asserts

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "53548aa2148e92a3ccbca80706782f95", "grade": false, "grade_id": "cell-1b37830e571f872c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Activation functions
#
# Our model will be a regression model. For the output of the model we need to fit the unscaled `y`. Hence we need an unbound linear activation function for the output:
#
# \begin{align}
#     linear(H(x)) &= H(x)
# \end{align}
#
# As non-linear activation function for the hidden layers we will use the ReLu function:
#
# \begin{align}
#     relu(H(x)) &= 
# \begin{cases} 
#     0~\text{ if }~h_l(x) \leq 0\\
#     h_l(x)~\text{ if }~h_l(x)>0
# \end{cases}
# \end{align}
#
# The ReLu has the biological motivation of having a minimal threshold until it starts outputting values different from 0. Additionally, it yields sparse activations in the network, which is usually favorable, and its gradient is trivial to calculate.
#
# Please calculate the ReLu of your previously calculated `h`:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "9ab1acb5fc88c13ad757486c80348314", "grade": false, "grade_id": "cell-5c8a3e28ebedb119", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Calculate the activation a by computing the ReLu of h
a = None
# YOUR CODE HERE
raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2b6cf6d2a764b2c0113a0678acb0e9cf", "grade": true, "grade_id": "ReLu-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# ReLu - 1 point

assert np.min(a) >= 0
# Possible hidden asserts

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8a6a2a28475def9fb6d582bd318b32b0", "grade": false, "grade_id": "cell-b2b2af32ce0ca084", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Backpropagation
#
# ### Weights
# From the lecture recall the following chain to get the gradient of the error $J$ with respect to the weights $\Theta$ of the output layer:
#
# \begin{align}
# \frac{\partial J}{\partial \Theta} &= \underset{(1)}{\frac{\partial J}{\partial a}} 
#                                       \underset{(2)}{\frac{\partial a}{\partial h}}
#                                       \underset{(3)}{\frac{\partial h}{\partial \Theta}} \\
# \end{align}
#
# For our case the error is the MSE defined as:
# \begin{align}
# J &= \frac{1}{2m} \sum_{i=1}^m \left( a^{(i)} - y^{(i)} \right)^2
# \end{align}
# ...for $m$ samples $i$.
#
# Everything else is given. Now calculate the partial derivatives for a single sample $i$ of:  
# **(1) the MSE  
# (2) the linear and ReLu functions  
# (3) $h(x)$**  
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "650a0f838bd12eee53c46634e7114de0", "grade": false, "grade_id": "cell-c639ca9b3959931f", "locked": true, "schema_version": 3, "solution": false, "task": false}
from IPython.display import display, Markdown

display(Markdown(r"\begin{align}"
                 r"\frac{\partial \text{MSE}}{\partial a^{(i)}} &= \frac{1}{m} \left( a^{(i)} - y^{(i)} \right) \tag{1}\\"
                 r"\frac{\partial \text{ReLu}^{(i)}}{\partial h^{(i)}} &= "
                 r"\begin{cases} "
                 r"0~\text{ if }~h^{(i)}(x) \leq 0\\"
                 r"1~\text{ if }~h^{(i)}(x)>0"
                 r"\end{cases} \tag{2}\\"
                 r"\frac{\partial \text{linear}^{(i)}}{\partial h^{(i)}} &= 1 \tag{2}\\"
                 r"\frac{\partial h^{(i)}}{\partial \theta^{(i)}} &= x^{(i)} \tag{3}"
                 r"\end{align}"))

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1ab2acada8e49cf1b058b83d633c227a", "grade": false, "grade_id": "cell-c47108b07d99fbf0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now let's take the above calculated `a` as output and calculate some random true labels `y_` for testing:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "984ef23cfd5fda32d981d71af0c46445", "grade": false, "grade_id": "cell-1b8e39a0adf8c3a9", "locked": true, "schema_version": 3, "solution": false, "task": false}
y_ = np.random.uniform(low=0, high=3, size=[10, 32])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "422b66439135a32895755ec4840a7dfd", "grade": false, "grade_id": "cell-00bda07a4a44647d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next we can use the derivatives from above to calculate the gradient for one layer:
#
# \begin{align}
# \frac{\partial J}{\partial \Theta} &= \underset{(1)}{\frac{\partial J}{\partial a}} 
#                                       \underset{(2)}{\frac{\partial a}{\partial h}}
#                                       \underset{(3)}{\frac{\partial h}{\partial \Theta}} \\
# \end{align}

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "92779ac0363dc94a79abde8b4453a8f7", "grade": false, "grade_id": "cell-d04d29795455d3b4", "locked": false, "schema_version": 3, "solution": true, "task": false}
# We start from the back:
# (3) is trivial
dh_dtheta = None
# YOUR CODE HERE
raise NotImplementedError()

# (2) for ReLu: Here you need a reference to h and maybe two steps
da_dh = None
# YOUR CODE HERE
raise NotImplementedError()

# (1) m is the batch size!
dJ_da = None
# YOUR CODE HERE
raise NotImplementedError()

dJ_dTheta = np.dot(dh_dtheta.T, (dJ_da * da_dh))
# The error gradient with respect to the weights and the shape of the weights should agree:
print(dJ_dTheta.shape, theta.shape)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4234c64dbac518d3e8e61520d3f745b2", "grade": true, "grade_id": "dh_dtheta-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# dh dtheta - 1 point

assert dh_dtheta.shape[0] == 10
assert dh_dtheta.shape[1] == 63
# Hidden test for the content of dh_dtheta

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d11c554e8aecdf1be6ebfc92d4f1637a", "grade": true, "grade_id": "da_dh-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# da dh - 1 point

assert da_dh.shape[0] == 10
assert da_dh.shape[1] == 32
# Hidden test for the content of drelu_dh

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "bee76a01284e330535afe1d7ba5b6d95", "grade": true, "grade_id": "dJ_da-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# dJ dh - 1 point

assert dJ_da.shape[0] == 10
assert dJ_da.shape[1] == 32
# Hidden test for the content of dJ_da

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c3b2f296fac8196ac43a71c7f3b13a99", "grade": false, "grade_id": "cell-ba0b36989b2f7672", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We can then update the weights for the next step using:
# \begin{align}
#     \Theta_{t+1} &= \Theta_t - \alpha \cdot \frac{\partial J}{\partial \Theta} \\
# \end{align}
# ...with learning rate $\alpha$.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6a723e4505dbb32f796f5f0d187891f0", "grade": false, "grade_id": "cell-b7e8e14e2d041ed8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Bias 
#
# Besides the weights we also need to fit the bias. The bias can be derived in the same manner, only that instead of $x$ the input is $1$. We simply multiply $1$ with the bias weights. Therefore (3) only for the bias collapses to:
# \begin{align}
# \frac{\partial h^{(i)}_b}{\partial \theta^{(i)}_b} &= x^{(i)}_b = 1\tag{3}
# \end{align}

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "72d713aef4e8a41b69704987b4972425", "grade": false, "grade_id": "cell-25e02e4c3321aa4d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next we will build a simple feed forward network.
#
# Recall from the lecture (Figure 5):
#
# <div>
# <img src="attachment:backprop_1.png" width="80%"/>
# </div>
# <p style="text-align:center;font-size:80%;font-style:italic">
# Figure 5: Backpropagation.
# </p>
#
# For backpropagation we are still missing the one element connecting the layers, namely $\frac{\partial a^2}{\partial a^1}$:
# \begin{align}
# a^2 &= \Theta^2 a^1 \\
# \frac{\partial a^2}{\partial a^1} &= \Theta^2
# \end{align}
#
# With this and the above derived equations we can calculated the gradient of the first (hidden) layer weights regarding the output error as:
# \begin{align}
# \frac{\partial J}{\partial \Theta^1} &= \underbrace{\frac{\partial J}{\partial a^2}
#                                                     \frac{\partial a^2}{\partial a^1}}_\text{(a.)}
#                                         \underbrace{\frac{\partial a^1}{\partial h^1}
#                                                     \frac{\partial h^1}{\partial \Theta^1}}_\text{(b.)} \\
# \end{align}
# **The part (a.) can be calculated in the second (output) layer and is returned as upstream gradient.**
#
# **In the first (hidden) layer we can then use this gradient and combine it with part (b.) to obtain the gradient for the weights of layer 1 with respect to the output error.**

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8074640cf9b5e7032a04ff486ef892db", "grade": false, "grade_id": "cell-336b1a80f598e3c7", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Let's put this into work by building a feed forward network for our regression task with:
#
# 1. The 63 `mol_descriptors` as input $x^{(i)}$
# 2. One hidden layer $1$ with 32 neurons and a ReLu activation: `class HiddenLayer`
# 3. One output layer $2$ with two outputs (LUMO and HOMO) with a linear (no) activation function: `class OutputLayer`
#
# Both have a forward pass, which calculates the output of the layer, a backward pass which calculates the gradients and an update function that updates the weights using the gradients and the learning rate.
#
# Only the `OutputLayer` has to return the part (a.) from above to then pass it on to the backward pass of the `HiddenLayer`.
#
# We start implementing from the bottom up with the `OutputLayer` (layer 2):
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "be9b4aba1c94e576bb2071f20a48c15e", "grade": false, "grade_id": "cell-5cec7f87a6783db2", "locked": false, "schema_version": 3, "solution": true, "task": false}
class OutputLayer:

    def __init__(self, n_inputs: int, n_outputs: int):
        # Initialize (n_inputs X n_outputs)-dimensional weight matrix self.theta with a
        # Glorot et Al. 2010 uniform initialization:
        self.theta = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Initialize the bias vector self.b to zeros:
        self.b = None
        # YOUR CODE HERE
        raise NotImplementedError()

    def forward(self, input_vector):
        self.input = input_vector
        # Compute the states h(x) as self.h
        self.h = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # As this is a linear layer a(x) = h(x)
        self.a = self.h
        return self.a

    def backward(self, y_predicted, y_true):
        # HINT: as we do things backwards you might have to transpose some matrices

        # partial derivative of the states with respect to the weights
        dh2_dtheta2 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # partial derivative of activations with respect to the states
        # One as we have a linear/no activation for the regression output
        da2_dh2 = 1

        # partial derivative of the error (MSE) with respect to the acivation
        # infer the batch size from the input shape for normalization
        dJ_da2 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Gradient of the weights with respect to the error
        # for the weight updates for this layer:
        self.dJ_dTheta2 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Gradient of the bias with respect to the error
        # Recall using dh2_db2 = 1 and handle the batch size by summation
        self.dJ_db2 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # The downstream gradient for layer 1.
        # employing da2_da1 = theta
        downstream_gradient = None
        # YOUR CODE HERE
        raise NotImplementedError()
        return downstream_gradient

    def update(self, learning_rate):
        # You don't need to change this
        # HINT:
        # If your model gets worse instead of better make sure you calculate the correct MSE
        self.theta = self.theta - learning_rate * self.dJ_dTheta2
        self.b = self.b - learning_rate * self.dJ_db2


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "01dbdad3a58c5a41b09b3aadef59de4a", "grade": false, "grade_id": "cell-a2f72c3745dfbb62", "locked": true, "schema_version": 3, "solution": false, "task": false}
X_sample = X_train[0:100, :]
y_sample = y_train[0:100, :]

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "47b1bbc1e90b5ad8a1412d3dbbf97303", "grade": true, "grade_id": "Output_Forward_Pass-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Output Forward Pass - 1 point

l2 = OutputLayer(X_sample.shape[1], y_sample.shape[1])
y_pred_sample = l2.forward(X_sample)

assert y_sample.shape[0] == y_pred_sample.shape[0]
assert y_sample.shape[1] == y_pred_sample.shape[1]

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f0c70551912c6b9a0e4549980fdc7477", "grade": true, "grade_id": "Output_Backward_Pass-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Output Backward Pass - 2 points

downstream_gradient = l2.backward(y_pred_sample, y_sample)

assert downstream_gradient.shape[0] == X_sample.shape[0]
assert downstream_gradient.shape[1] == X_sample.shape[1]

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "37456bb228a6b562df835641760c5e12", "grade": true, "grade_id": "Output_Weight_Update-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Output Weight Update - 2 points

l2 = OutputLayer(X_sample.shape[1], y_sample.shape[1])
y_pred_before = l2.forward(X_sample)

for i in range(100):
    y_pred_sample = l2.forward(X_sample)
    l2.backward(y_pred_sample, y_sample)
    l2.update(0.05)

y_pred_after = l2.forward(X_sample)
r2_before = r2_score(y_sample, y_pred_before)
r2_after = r2_score(y_sample, y_pred_after)

assert r2_before < r2_after


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b095fdf2cfd03d569292a1962be9d662", "grade": false, "grade_id": "cell-677fcd4d04d26c49", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next is the `HiddenLayer`. There are mainly two main differences you have to keep in mind here:
# 1. You have to work with the ReLu activation, so $\frac{\partial a}{\partial h}$ isn't $1$ anymore.
# 2. We use the `upstream_gradient` for the backward pass, provided by the `backward` method of the `OutputLayer`.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "20168340cbd6cdf2c11f9034135b25c7", "grade": false, "grade_id": "cell-10128093607d268a", "locked": false, "schema_version": 3, "solution": true, "task": false}
class HiddenLayer:

    def __init__(self, n_inputs: int, n_outputs: int):
        # Initialize the weight matrix self.theta with a
        # Glorot et Al. 2010 uniform initialization:
        self.theta = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Initialize the bias vector to zeros:
        self.b = None
        # YOUR CODE HERE
        raise NotImplementedError()

    def forward(self, input_vector):
        self.input = input_vector
        # Compute the states h(x) as self.h
        self.h = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Compute the activations a(x) as self.a with ReLu activation
        self.a = None
        # YOUR CODE HERE
        raise NotImplementedError()
        return self.a

    def backward(self, upstream_gradient):
        # HINT: as we do things backwards you might have to transpose some matrices

        # Gradient of the states with respect to the inputs (trivial):
        dh1_dtheta1 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Gradient of the activations with respect to the states:
        # Remember to apply ReLu here
        da1_dh1 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Gradient of the error with respect to the weights:
        # Now we can finally use the upstream gradient...
        self.dJ_dTheta1 = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # And for the bias, similar to the output layer but this time
        # using the upstream gradient:
        self.dJ_db1 = None
        # YOUR CODE HERE
        raise NotImplementedError()

    def update(self, learning_rate):
        # You don't need to change this
        self.theta = self.theta - learning_rate * self.dJ_dTheta1
        self.b = self.b - learning_rate * self.dJ_db1


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ebf949dc6657a84671c041459ed04a50", "grade": false, "grade_id": "cell-8c9fba61d9a7e611", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We create a sample from the existing data that matches the output dimension 
# and also shift it above zero, so it is learnable with ReLu:

n_hidden = 32

X_sample = X_train[0:100, :]
y_sample = np.array([y_train[i * 100:i * 100 + 100, 0] for i in range(n_hidden)]).T
y_sample = y_sample - y_sample.min()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "0cd3ab232551bde363a738e85c024eaa", "grade": true, "grade_id": "Hidden_Forward_Pass-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Hidden Forward Pass - 1 point

l1 = HiddenLayer(X_sample.shape[1], n_hidden)

y_pred_sample = l1.forward(X_sample)
assert y_pred_sample.shape[0] == X_sample.shape[0]
assert y_pred_sample.shape[1] == n_hidden
assert y_pred_sample.min() >= 0

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "160e25a8ac1a0fa136f6a857abd9c703", "grade": true, "grade_id": "Hidden_Weight_Update-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Hidden Weight Update - 2 points

l1 = HiddenLayer(X_sample.shape[1], n_hidden)
y_pred_before = l1.forward(X_sample)

for i in range(100):
    y_pred_sample = l1.forward(X_sample)
    downstream_gradient = 1 / 100 * (y_pred_sample - y_sample)
    l1.backward(downstream_gradient)
    l1.update(0.05)

y_pred_after = l1.forward(X_sample)
r2_before = r2_score(y_sample, y_pred_before)
r2_after = r2_score(y_sample, y_pred_after)

assert r2_before < r2_after

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "83f716735540190bf138fe343af3ca71", "grade": false, "grade_id": "cell-354dc7f5d7680449", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## NN Training
#
# Okay now let's see whether we can be better with our network than the benchmark ridge regression.
#
# As said before we will train our network in batches (`batch_size`) and for `n_epochs`.  
# Per epoch we therefore have to pass `X_train.shape[0] // batch_size` batches.  
# For each epoch we randomly shuffle the dataset. Using `np.random.permutation(X_train.shape[0])` we generate a randomly shuffled index. Indexing X and y with slices of this shuffled index, creates differently shuffled batches for each epoch.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5e9496a196e31d24b70904c2356e32c8", "grade": false, "grade_id": "cell-db017859cd6974c3", "locked": true, "schema_version": 3, "solution": false, "task": false}
n_hidden = 64
lr = 0.05
n_epochs = 60
batch_size = 100
n_batches = X_train.shape[0] // batch_size

l1 = HiddenLayer(X_train.shape[1], n_hidden)
l2 = OutputLayer(n_hidden, y_train.shape[1])

y_pred_before = l2.forward(l1.forward(X_test))

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "479d999efc2dd6a968ae20950766ad2a", "grade": false, "grade_id": "cell-5e91f098bcd19cff", "locked": false, "schema_version": 3, "solution": true, "task": false}
for epoch in range(n_epochs):
    permutation = np.random.permutation(X_train.shape[0])

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train[permutation[start:end]]
        y_batch = y_train[permutation[start:end]]

        # Get the predictions
        y_pred = None
        # YOUR CODE HERE
        raise NotImplementedError()

        # Do the backward pass of both layers and update the weights
        # YOUR CODE HERE
        raise NotImplementedError()

    y_pred = l2.forward(l1.forward(X_test))
    if epoch % 4 == 0:
        print(f"Epoch {epoch}: Test R2 = {r2_score(y_test, y_pred)}")

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "512a47b90437c3dd95c5aa3b998738f8", "grade": true, "grade_id": "Full_Training-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
# Full Training - 2 points

y_pred_after = l2.forward(l1.forward(X_test))

r2_before = r2_score(y_test, y_pred_before)
r2_after = r2_score(y_test, y_pred_after)

assert r2_before < r2_after

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "650b24c3f3b868c6f8ea6835b063f429", "grade": false, "grade_id": "cell-aa9f5444c49aa249", "locked": true, "schema_version": 3, "solution": false, "task": false}
y_pred = l2.forward(l1.forward(X_test))

r2_lumo_nn = r2_score(y_test[:, 0], y_pred[:, 0])
r2_homo_nn = r2_score(y_test[:, 1], y_pred[:, 1])
print(f'R2 LUMO: {r2_lumo_nn}\nR2 HOMO: {r2_homo_nn}')

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "769d31191675e9386f0c2cf068095f59", "grade": true, "grade_id": "Bonus_Beating_Ridge-1", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Bonus Beating Ridge - 1 point

print(f'\tRidge\t\t\tNN\nLUMO\t{r2_lumo_ridge}\t{r2_lumo_nn}\nHOMO\t{r2_homo_ridge}\t{r2_homo_nn}')

assert r2_lumo_nn > r2_lumo_ridge and r2_homo_nn > r2_homo_ridge

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ba0d17f177b2dd43f1e8a233e262d2c7", "grade": false, "grade_id": "cell-1d228c998d797378", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Quite possibly you were able to beat the ridge regression at that point, only with one hidden ReLu layer. Of course the hyperparameters of both models haven't been optimized yet, so you can get even better. But please do this in a separate notebook ;)
#
# Let's have a look at the final plots:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "327ca0e965f5286f79ad47af9ac2d9a4", "grade": false, "grade_id": "cell-afdaac610e54e883", "locked": true, "schema_version": 3, "solution": false, "task": false}
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.2, s=5)
axs[0].plot([-6, -1], [-6, -1], 'k')
axs[0].set_title(f'R2 LUMO: {r2_lumo_nn}')
axs[0].set_xlabel('true LUMO')
axs[0].set_ylabel('predicted LUMO')
axs[0].set_xlim([-6, -1])
axs[0].set_ylim([-6, -1])
axs[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.2, s=5)
axs[1].plot([-9, -3], [-9, -3], 'k')
axs[1].set_title(f'R2 HOMO: {r2_homo_nn}')
axs[1].set_xlabel('true HOMO')
axs[1].set_ylabel('predicted HOMO')
axs[1].set_xlim([-9, -3])
axs[1].set_ylim([-9, -3])

plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "4fbdcb385dc9820f3017fac00e1b0e90", "grade": false, "grade_id": "cell-1e05cf72bc62162b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **You finished the Exercise!**
#
# Next time we will start using Tensorflow and Keras to build neural networks.  
#
# Here is a little example for our application. Be aware that Tensorflow does things slightly different internally, so it might give results that differ from your own implementation.
#
# You can use this example as a start to experiment with own implementations **in another notebook**:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "14d7e5ef4dc6529a3b70adc19b5c24c3", "grade": false, "grade_id": "cell-b41ed4cb02d0788f", "locked": true, "schema_version": 3, "solution": false, "task": false}
from tensorflow import keras
from tensorflow.keras import layers

n_hidden = 64
lr = 0.05
# A low number of epochs so the nb doesn't slow down during grading
n_epochs = 5
batch_size = 100

inputs = keras.Input(shape=(X.shape[1],))
hidden1 = layers.Dense(n_hidden, activation='relu')(inputs)
outputs = layers.Dense(2, activation='linear')(hidden1)
model = keras.Model(inputs=inputs, outputs=outputs, name="simple_ff")
print(model.summary())

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.SGD(),
    metrics=["MSE"],
)

model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs)

y_pred = model.predict(X_test)
r2_lumo_keras = r2_score(y_test[:, 0], y_pred[:, 0])
r2_homo_keras = r2_score(y_test[:, 1], y_pred[:, 1])
print(f'R2 LUMO: {r2_lumo_keras}\nR2 HOMO: {r2_homo_keras}')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.2, s=5)
axs[0].plot([-6, -1], [-6, -1], 'k')
axs[0].set_title(f'R2 LUMO: {r2_lumo_keras}')
axs[0].set_xlabel('true LUMO')
axs[0].set_ylabel('predicted LUMO')
axs[0].set_xlim([-6, -1])
axs[0].set_ylim([-6, -1])
axs[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.2, s=5)
axs[1].plot([-9, -3], [-9, -3], 'k')
axs[1].set_title(f'R2 HOMO: {r2_homo_keras}')
axs[1].set_xlabel('true HOMO')
axs[1].set_ylabel('predicted HOMO')
axs[1].set_xlim([-9, -3])
axs[1].set_ylim([-9, -3])

plt.show()

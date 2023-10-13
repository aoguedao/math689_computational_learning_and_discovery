# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: sciml-edu
#     language: python
#     name: python3
# ---

# # Homework #06

# +
import secrets
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import stats
# -

data_path = Path().resolve().parent / "data"

# ## Bayes

# Random seed for reproducibility
# secrets.randbits(128) # 208905213533139122735706682150229709525
rng = np.random.default_rng(208905213533139122735706682150229709525)
indices_train = rng.choice(5000, 2500, replace=False)
indices_test = rng.choice(800, 400, replace=False)
flag_full_dataset = False  # If it is True it will use full train and test datasets 

train_list = []  # Auxiliary list of train datasets
for f in data_path.glob("train*.txt"):
    raw_data = np.loadtxt(f) if flag_full_dataset else np.loadtxt(f)[indices_train, :]  # Sample or full dataset
    target = raw_data[:, [0]]  # Target values, i.e. digit 
    features = (raw_data[:, 1:] / raw_data[:, 1:].max(axis=1, keepdims=True)).astype(bool).astype(int)  # Pixels values, normalization and then cast to 0 and 1
    train_list.append(np.hstack((target, features)))  # Add to the temp list
train_data = np.vstack(train_list)  # Concatenate train datasets
train_data.shape

# Similar to train dataset
test_list = []
for f in data_path.glob("test*.txt"):
    raw_data = np.loadtxt(f) if flag_full_dataset else np.loadtxt(f)[indices_test, :]
    target = raw_data[:, [0]]
    features = (raw_data[:, 1:] / raw_data[:, 1:].max(axis=1, keepdims=True)).astype(bool).astype(int)
    test_list.append(np.hstack((target, features)))
test_data = np.vstack(test_list)
test_data.shape

# Split datasets into features matrices and target vectors
X_train = train_data[:, 1:]
y_train = train_data[:, 0].astype(int)
X_test = test_data[:, 1:]
y_test = test_data[:, 0].astype(int)

# +
y_train_unique, y_train_count = np.unique(y_train, return_counts=True)  # Get unique targets values (digits) and its count
prob_v_dict = dict(zip(y_train_unique.astype(int), y_train_count / y_train.size))  # Dictionary with $prob(v_j)$

m = 1
p1 = 1 / 2  # Values can be 0 or 1
p2 = 1 / 4  # Values can be 00, 01, 10, 00

prob_a_0_dict = {}  # Dictionary with $p(a_i | v_j)$ when $a_i$ = 0 for each $v_j$
prob_a_1_dict = {}  # Dictionary with $p(a_i | v_j)$ when $a_i$ = 1 for each $v_j$
prob_pairs_dict = {}  # Dictionary of dictrionaries of each combination $p(a_i | a_{i+1}, v_j)$ for each $v_j$.
for j in y_train_unique:
    X_train_target_j = X_train[y_train == j]
    prob_a_0_dict[j] = ((X_train_target_j == 0).sum(axis=0) + m * p1) / (X_train_target_j.shape[0] + m)
    prob_a_1_dict[j] = ((X_train_target_j == 1).sum(axis=0) + m * p1) / (X_train_target_j.shape[0] + m)
    prob_pairs_dict[j] = {}
    for k in range(X_train_target_j.shape[1] - 1):
        X_train_target_j_k = X_train_target_j[:, k:k+2]
        pairs, pairs_count = np.unique(X_train_target_j_k, axis=0, return_counts=True)
        denom = (X_train_target_j_k.shape[0] + m)
        d = {(0, 0): m * p2 / denom, (0, 1): m * p2 / denom, (1, 0): m * p2 / denom, (1, 1): m * p2 / denom}  # Dummy
        for pair in range(pairs.shape[0]):
            d[tuple(pairs[pair])] = (pairs_count[pair] + m * p2) / denom
        prob_pairs_dict[j][k] = d
# -

y_pred = np.empty(shape=X_test.shape[0])  # Vector of predictions
for i in range(X_test.shape[0]):  # Iterate over each sample
    X_test_i = X_test[i, :]
    prob_dict = {}  # Dictionary of $p(a_1, ..., a_n | v_j) p(v_j)$ for each $v_j$
    for j in y_train_unique:
        # p(a_n | v_j)
        if X_test_i[-1] == 0:
            prob = prob_a_0_dict[j][-1]
        else:
            prob = prob_a_1_dict[j][-1]
        # p(a_i | a_{i+1}, v_j)
        for k in range(X_test_i.shape[0] - 1):
            a_k = X_test_i[k]
            a_k_plus_1 = X_test_i[k]
            prob *= prob_pairs_dict[j][k][a_k, a_k_plus_1]
        prob_dict[j] = prob * prob_v_dict[j]
    y_pred[i] = int(max(prob_dict, key=prob_dict.get))  # Argmax

n = y_pred.size
error_h = sum(y_pred == y_test) / n
sigma = np.sqrt(error_h * (1 - error_h) / n)
print(f"Accuracy was {error_h:.2%} and with approximately 95% probability the true error lies in the interval [{error_h - 1.96 * sigma:.4f}, {error_h + 1.96 * sigma:.4f}]")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax
)
fig.tight_layout()
fig.savefig(f"belief_confusion_matrix_fulldata_{flag_full_dataset}.png", dpi=300)
fig.show()

# Confusion matrix - row normalization
fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    normalize="true",
    values_format=".2f",
    ax=ax
)
fig.tight_layout()
fig.savefig(f"belief_confusion_matrix_row_normalize_fulldata_{flag_full_dataset}.png", dpi=300)
fig.show()

# ## EM

em_filepath = data_path / "Alonso.txt"
x = np.loadtxt(em_filepath)
display(x.shape)
display(x.min())
display(x.max())

mu1_0, mu2_0 = -3, 3
sigma = 1


def plot_gaussians(x, mu1, mu2, filepath=None):
    x_min = np.min([stats.norm.ppf(0.01, loc=mu1, scale=sigma), stats.norm.ppf(0.01, loc=mu2, scale=sigma)])
    x_max = np.max([stats.norm.ppf(0.99, loc=mu1, scale=sigma), stats.norm.ppf(0.99, loc=mu2, scale=sigma)])
    x_linspace = np.linspace(x_min, x_max, 100)
    fig, ax = plt.subplots(figsize=(10, 4))
    gaussian1 = stats.norm.pdf(x_linspace, loc=mu1, scale=sigma)
    gaussian2 = stats.norm.pdf(x_linspace, loc=mu2, scale=sigma)
    ax.plot(x_linspace, gaussian1, color="red", label=rf"$\mu_1 = {mu1:.4f}$")
    ax.plot(x_linspace, gaussian2, color="green", label=rf"$\mu_2 = {mu2:.4f}$")
    ax.vlines(mu1, ymin=0, ymax=gaussian1.max(), color="red", linestyles="dashed")
    ax.vlines(mu2, ymin=0, ymax=gaussian2.max(), color="green", linestyles="dashed")
    ax.scatter(x, np.zeros_like(x), marker="x", label="Data")
    ax.legend()
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath, dpi=300)
    fig.show()


# +

tol = 1e-16
max_iterations = 100000

mu1 = mu1_0
mu2 = mu2_0
h = [mu1, mu2] 
h_array = [h]
for i in range(max_iterations):
    # E-step
    prob_x_mu1 = np.exp(-0.5 * ((x - mu1) / sigma) ** 2)
    prob_x_mu2 = np.exp(-0.5 * ((x - mu2) / sigma) ** 2)
    E_z1 = prob_x_mu1 / (prob_x_mu1 + prob_x_mu2)
    E_z2 = prob_x_mu2 / (prob_x_mu1 + prob_x_mu2)

    # M-step
    mu1_new = np.sum(E_z1 * x) / np.sum(E_z1) 
    mu2_new = np.sum(E_z2 * x) / np.sum(E_z2) 
    if np.linalg.norm([mu1 - mu1_new, mu2 - mu2_new], ord=2) < tol:
        break
    mu1 = mu1_new
    mu2 = mu2_new
    h_array.append([mu1, mu2])
print(i)
print(mu1, mu2)
# -

plot_gaussians(x, mu1_0, mu2_0, filepath=f"gaussian_mixture_init.png")

plot_gaussians(x, mu1, mu2, filepath="gaussian_mixture.png")

h_array = np.array(h_array)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h_array[:, 0], "-ro", label=rf"$\mu_1 = {mu1:.4f}$")
ax.plot(h_array[:, 1], "-go", label=rf"$\mu_2 = {mu2:.4f}$")
ax.legend()
fig.tight_layout()
fig.savefig("gaussian_mixture_convergence.png" , dpi=300)
fig.show()

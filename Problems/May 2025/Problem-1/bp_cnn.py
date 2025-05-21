import numpy as np

# -------------------------
# Convolution Forward Pass
# -------------------------
def conv_forward(X, W, b):
    """
    Perform 2D convolution over a 3D input (C x H x W)
    
    Parameters:
        X: Input feature map, shape (C, H, W)
        W: Convolution filters, shape (F, C, K, K)
        b: Bias vector, shape (F,)

    Returns:
        Z: Convolution output, shape (F, H-K+1, W-K+1)
    """
    C, H, Wt = X.shape
    F, _, K, _ = W.shape
    outH = H - K + 1
    outW = Wt - K + 1
    Z = np.zeros((F, outH, outW))
    
    for f in range(F):
        for i in range(outH):
            for j in range(outW):
                patch = X[:, i:i+K, j:j+K]
                Z[f, i, j] = np.sum(W[f] * patch) + b[f]
    return Z

# --------------------------
# Convolution Backward Pass
# --------------------------
def conv_backward(dZ, X, W):
    """
    Compute gradients for convolution layer.

    Parameters:
        dZ: Gradient w.r.t output Z, shape (F, outH, outW)
        X: Input feature map, shape (C, H, W)
        W: Convolution filters, shape (F, C, K, K)

    Returns:
        dX: Gradient w.r.t input X
        dW: Gradient w.r.t weights W
        db: Gradient w.r.t biases b
    """
    F, outH, outW = dZ.shape
    C, H, Wt = X.shape
    _, _, K, _ = W.shape

    dW = np.zeros_like(W)
    db = np.zeros(F)
    dX = np.zeros_like(X)

    for f in range(F):
        db[f] = np.sum(dZ[f])
        for i in range(outH):
            for j in range(outW):
                patch = X[:, i:i+K, j:j+K]
                dW[f] += dZ[f, i, j] * patch
                dX[:, i:i+K, j:j+K] += dZ[f, i, j] * W[f]
    return dX, dW, db

# -----------------------------
# ReLU Activation and Backward
# -----------------------------
def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    return dA * (Z > 0)

# --------------------------------
# Max Pooling Forward and Backward
# --------------------------------
def maxpool_forward(A, size=2, stride=2):
    """
    Apply 2x2 max pooling with stride 2

    Returns:
        P: Pooled output
        argmax: Indices of max values for backward pass
    """
    F, H, W = A.shape
    outH, outW = H // stride, W // stride
    P = np.zeros((F, outH, outW))
    argmax = {}

    for f in range(F):
        for i in range(outH):
            for j in range(outW):
                block = A[f, i*stride:(i+1)*stride, j*stride:(j+1)*stride]
                idx = np.unravel_index(np.argmax(block), block.shape)
                argmax[(f, i, j)] = idx
                P[f, i, j] = block[idx]
    return P, argmax

def maxpool_backward(dP, A, argmax, size=2, stride=2):
    """
    Backpropagate through max pooling using stored indices
    """
    F, H, W = A.shape
    dA = np.zeros_like(A)
    outH, outW = dP.shape[1:]

    for (f, i, j), idx in argmax.items():
        dA[f, i*stride + idx[0], j*stride + idx[1]] = dP[f, i, j]
    return dA

# -------------------
# Softmax + Loss
# -------------------
def softmax(s):
    e = np.exp(s - np.max(s))  # Stability fix
    return e / np.sum(e)

def cross_entropy_loss(ŷ, y):
    return -np.sum(y * np.log(ŷ))

# -------------------
# Example Forward + Backward
# -------------------
X = np.random.randn(3, 8, 8)  # Input: 3 channels, 8x8
W_conv = np.random.randn(4, 3, 3, 3)  # 4 filters, 3x3
b_conv = np.zeros(4)
U_fc = np.random.randn(5, 4 * 6 * 6)  # FC layer after pooling
c_fc = np.zeros(5)
y_true = np.zeros(5)
y_true[2] = 1  # Class 2 is the correct label

# FORWARD PASS
Z = conv_forward(X, W_conv, b_conv)     # Conv
A = relu(Z)                             # ReLU
P, argmax = maxpool_forward(A)         # MaxPool
p = P.reshape(-1)                      # Flatten
s = U_fc.dot(p) + c_fc                 # FC
ŷ = softmax(s)                          # Softmax
loss = cross_entropy_loss(ŷ, y_true)   # Loss

# BACKWARD PASS
ds = ŷ - y_true                         # dL/ds

dU = ds[:, None] * p[None, :]          # dL/dU

dc = ds                                 # dL/dc

dp = U_fc.T.dot(ds)                     # dL/dp
dP = dp.reshape(P.shape)               # Reshape

dA = maxpool_backward(dP, A, argmax)   # Through pooling
dZ = relu_backward(dA, Z)              # Through ReLU

dX, dW_conv, db_conv = conv_backward(dZ, X, W_conv)  # Through Conv
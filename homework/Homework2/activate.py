import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    for i in range(x.shape[0]):
        x[i] = max(x[i],0)
    return x

def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

vals = np.linspace(-10, 10, num=100, dtype=np.float32)
activation = sigmoid(vals)
fig = plt.figure(figsize=(12,6))
plt.plot(vals, activation)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks()
plt.ylim([-0.5, 1.5])
plt.show()


activation = tanh(vals)
print(vals)
print(activation)
fig1 = plt.figure(figsize=(12,6))
plt.plot(vals, activation)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks()
plt.xlim([-4,4])
plt.ylim([-1.5, 1.5])
plt.show()

activation = relu(vals)
fig1 = plt.figure(figsize=(12,6))
plt.plot(vals, activation)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks()
plt.xlim([-4,4])
plt.ylim([-0.5, 4])
plt.show()


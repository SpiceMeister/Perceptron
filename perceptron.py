import numpy as np
import matplotlib.pyplot as plt

# Set up formulas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


def costFunc(real, desired):
    return (real - desired) ** 2


def costDeriv(real, desired):
    return 2 * (real - desired)


# Set up array of training data [Input1, Input2, Input3, TargetOutput]
# If there is a 1 in the first input then the output should be a 1
training = np.array([[0,0,1, 0], [1,1,1, 1], [1,0,1, 1], [0,1,1, 0]])


# Initialize weights and learning rate
weights = 2 * np.random.random((3, 1)) - 1
learning_rate = .5
# Initialize cumulative costs and epochs for graphs/
costCum = []
epochCum = []
for epoch in range(5000):
    randInput = np.random.randint(len(training))
    sample = training[randInput]
    z = sample[0] * weights[0] + sample[1] * weights[1] + sample[2] * weights[2]
    a = sigmoid(z)
    target = sample[3]
    cost = costFunc(a, target)
    costCum.append(cost)
    epochCum.append(epoch)

    dCost_dReal = costDeriv(a, target)
    dReal_dZ = sigmoid_p(z)
    partialFormula = dCost_dReal * dReal_dZ

    dZ_dWeight1 = sample[0]
    dZ_dWeight2 = sample[1]
    dZ_dWeight3 = sample[2]

    dCost_dWeight1 = dZ_dWeight1 * partialFormula
    dCost_dWeight2 = dZ_dWeight2 * partialFormula
    dCost_dWeight3 = dZ_dWeight3 * partialFormula

    weights[0] -= learning_rate * dCost_dWeight1
    weights[1] -= learning_rate * dCost_dWeight2
    weights[2] -= learning_rate * dCost_dWeight3

userInput = [int(input("1: ")), int(input("2: ")), int(input("3: "))]
z = userInput[0] * weights[0] + userInput[1] * weights[1] + userInput[2] * weights[2]
y = sigmoid(z)
print(y)
plt.scatter(epochCum, costCum)
plt.show()



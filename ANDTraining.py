from random import choice
import numpy as np
from numpy import array, dot, random
import matplotlib.pyplot as plt
import time


unit_step = lambda x: 0 if x < 0 else 1

training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),
]

w = random.rand(3)
errors = []
eta = 0.2
n = 100

dataList = [elem1[0] for elem1 in training_data]
dataList = np.array(dataList)



a = [0, -w[0] / w[2]]
c = [-w[0]/w[1], 0]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set_title("Perceptron döntési felület")
plt.scatter(dataList[:, 1], dataList[:, 2])
plt.scatter(dataList[:, 2], dataList[:, 1])
plt.scatter(dataList[:, 1], dataList[:, 1])

for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x

    plt.plot([w[0], w[1]], 'r-', lw=2)
    plt.xlim((-0.1, 1.5))
    plt.ylim((-0.1, 1.5))
    print(w[1],w[2])

    fig.canvas.draw()
    time.sleep(1)
    fig.canvas.flush_events()
    




for x, _ in training_data:
    result = dot(x, w)
    
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
import numpy as np
import matplotlib.pyplot as plt
import time

H = [1, 0, 1, 1, 1, 1, 1, 0, 1] # H
I = [0, 1, 0, 0, 1, 0, 0, 1, 0] #I
train_data = np.array([
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
])

train_labels = np.array([
    [1, 0],
    [0, 1],
])

def hardlim(val):
    return 0 if val < 0 else 1

def perceptron_learning(data, labels):
    N, n = data.shape
    lr = .01
    w1 = np.random.randn(n, 1)
    w2 = np.random.randn(n, 1)
    
    E = 1

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    plt.scatter(H, H)

    #plt.scatter(I,I)
    
    while E != 0:
        E = 0
        
        for i in range(N):
            yi1 = hardlim(np.dot(data[i], w1))
            yi2 = hardlim(np.dot(data[i], w2))

            ei1 = labels[i][0] - yi1
            ei2 = labels[i][1] - yi2
            
            w1 += lr * ei1 * data[i].reshape(n, 1)
            w2 += lr * ei2 * data[i].reshape(n, 1)

            E += ei1 ** 2
            E += ei2 ** 2


            plt.plot([w1[0], w1[1]], 'r-', lw=2)
            plt.plot([w2[0], w2[1]], 'b-', lw=2)
            print(w1, w2)
            fig.canvas.draw()
            time.sleep(0.01)
            fig.canvas.flush_events()



perceptron_learning(train_data, train_labels)
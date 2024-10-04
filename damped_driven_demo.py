import numpy as np
import matplotlib.pyplot as plt
from integrators import RK4_V
import matplotlib.animation as ani
from time import sleep, time

def accel(x, v, g, R, b, A, k, m, t):
    p1 = (g / R) * np.sin(x)
    p2 = (1 / (m * R ** 2)) * (- b * v + A * np.cos(k * t))
    return - p1 + p2

R = 1
b = 0.5  # damping
m = 1
A = 1.15
k = 0.667
g = 1

T = 40
dt = 0.01

size = 100
d0 = 0.01

X = np.zeros([int(T / dt), size])
V = np.zeros([int(T / dt), size])

for t in range(1, size):
    X[0, t] = d0 * np.cos((t / size) * 2 * np.pi)
    V[0, t] = d0 * np.sin((t / size) * 2 * np.pi)



for i in range(size):
    for j in range(int(T / dt) - 1):
        t = j * dt
        V[j + 1, i], X[j + 1, i] = RK4_V(X[j, i], V[j, i], dt, accel, V[j, i], g, R, b, A, k, m, t)


for i in range(size):
    plt.scatter(X[:, i], V[:, i], s=2)
plt.show()


fig = plt.figure()
ax = fig.gca()

def animate(i):
    speed = 5
    ax.clear()
    ax.grid()
    ax.set_xlim([X[i * speed, 0] - 5 * d0, X[i * speed, 0] + 5 * d0])
    ax.set_ylim([V[i * speed, 0] - 5 * d0, V[i * speed, 0] + 5 * d0])
    ax.set_xlabel("theta(rad)")
    ax.set_ylabel("omega(rad/s)")
    speed = 5
    ax.plot(X[:, 0], V[:, 0], c = 'red')
    for n in range(size):
        ax.scatter(X[i * speed, n], V[i * speed, n], s=2, c = 'blue')
        

anim = ani.FuncAnimation(fig, animate, interval=0.02)
plt.show()

    

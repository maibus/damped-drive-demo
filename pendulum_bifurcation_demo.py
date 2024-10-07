import numpy as np
import matplotlib.pyplot as plt
from integrators import RK4_V
import matplotlib.animation as ani
from time import sleep, time

def accel(x, v, g, R, b, A, k, m, t):
    p1 = (g / R) * np.sin(x)
    p2 = (1 / (m * R ** 2)) * (- b * v + A * np.cos(k * t))
    return - p1 + p2

def sigmoid(x):
    return (1 + np.exp(-x)) ** -1

R = 1
b = 0.5  # damping
m = 1
A = 1.5  # standard chaotic value is 1.15
k = 0.667
g = 1

T = 1000
dt = 0.01

size = 1

X = np.zeros([int(T / dt), size])
V = np.zeros([int(T / dt), size])

sample_num = 11000
ran = [0, 1.5]
A_val = np.linspace(ran[0], ran[1], sample_num)

output = np.zeros([2, int(T / dt), sample_num])


t0 = time()
for n in range(sample_num):
    
    X = np.zeros([int(T / dt), size])
    V = np.zeros([int(T / dt), size])
    A = (n / sample_num) * (ran[1] - ran[0]) + ran[0]
    print(n, A)
    for i in range(size):
        for j in range(int(T / dt) - 1):
            t = j * dt
            V[j + 1, i], X[j + 1, i] = RK4_V(X[j, i], V[j, i], dt, accel, V[j, i], g, R, b, A, k, m, t)

    T_arr = np.linspace(0, T, int(T / dt))

    freq = np.fft.fftfreq(T_arr.shape[-1])
    sp = np.fft.fft(X[:, 0])
    norm = np.sqrt(np.real(sp) ** 2 + np.imag(sp) ** 2)
    output[0, :, n] = norm
    output[1, :, n] = freq
print(time() - t0)

bins = 1000
cut = 200


hist = np.zeros([cut, sample_num])
freq = np.zeros([cut, sample_num])

for n in range(sample_num):
    hist[:, n] = sigmoid(10 * output[0, 0:cut, n] / np.max(output[0, 0:cut, n]))
    freq[:, n] = output[1, 0:cut, n]

plt.imshow(hist, cmap='hot')

fig = plt.figure()
ax = fig.gca()

def animate(i):
    speed = 5
    ax.clear()
    #ax.set_xlim([0, 0.02])
    ax.scatter(freq[:, i], hist[:, i], s=2)
        

anim = ani.FuncAnimation(fig, animate, interval=0.02)
plt.show()

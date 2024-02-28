import numpy as np
from copy import deepcopy


def stupid_model(T):
    x = []
    for t in range(1, T + 1):
        theta = t * 3 * 2 * np.pi / T
        if (np.sin(theta) > 0 and np.cos(theta) > 0) or (
            np.sin(theta) < 0 and np.cos(theta) < 0
        ):
            x.append([np.cos(theta), np.sin(theta)])
        else:
            s = np.sign(np.sin(theta))
            x.append([s / (np.tan(theta) - 1), s + (s / (np.tan(theta) - 1))])
        x[-1][0] *= np.exp((t * 3 / T) * 0.005)
        x[-1][1] *= np.exp((t * 3 / T) * 0.005)

    x = np.array(x)
    x += np.random.normal(0, 0.001, size=(T, 2))
    return x


def basic_model(T):
    x = [[-1, -1]]
    v = [1, 0]
    dt = 0.005
    for t in range(T):
        x.append([x[-1][0] + v[0] * dt, x[-1][1] + v[1] * dt])
        if abs(x[-1][0]) < 1:
            continue
        elif x[-1][0] > 1:
            v[0] += -(x[-1][0] - 1) * dt
            v[1] += -x[-1][1] * dt
        elif x[-1][0] < -1:
            v[0] += -(x[-1][0] + 1) * dt
            v[1] += -x[-1][1] * dt

    x = np.array(x)
    x += np.random.normal(0, 0.001, size=(T + 1, 2))
    return x


def accelerating_model(T):
    x = [[-1, -1]]
    v = [1, 0]
    dt = 0.001
    for t in range(T):
        x.append([x[-1][0] + v[0] * dt, x[-1][1] + v[1] * dt])
        if abs(x[-1][0]) < 1:
            if x[-1][0] < 0:
                v[0] += 0.1*dt
            else:
                v[0] -= 0.1*dt
        elif x[-1][0] > 1:
            v[0] += -(x[-1][0] - 1) * dt
            v[1] += -x[-1][1] * dt
        elif x[-1][0] < -1:
            v[0] += -(x[-1][0] + 1) * dt
            v[1] += -x[-1][1] * dt

    x = np.array(x)
    x += np.random.normal(0, 0.001, size=(T + 1, 2))
    return x


def advanced_model(T):  ## don't use yet
    x = [[-0.8, -0.8]]
    v = [0.3, 0]
    dt = 0.005

    def get_track_limits(pos, direction):
        pos = deepcopy(pos)
        counter = 0
        while True:
            counter += 1
            pos[0] += direction[0] * dt * 0.1
            pos[1] += direction[1] * dt * 0.1

            if pos[0] > 1 and (pos[0] - 1) ** 2 + pos[1] ** 2 > 1:
                break
            if pos[0] < -1 and (pos[0] + 1) ** 2 + pos[1] ** 2 > 1:
                break
            if abs(pos[1]) > 1:
                break
        return counter / 10

    for t in range(T):
        front_dist = get_track_limits(x[-1], v)
        right_dist = get_track_limits(x[-1], [v[1], -v[0]])
        left_dist = get_track_limits(x[-1], [-v[1], v[0]])
        f_r_dist = get_track_limits(
            x[-1], [(v[0] + v[1]) / 2, (v[1] - v[0]) / 2]
        )
        f_l_dist = get_track_limits(
            x[-1], [(v[0] - v[1]) / 2, (v[1] + v[0]) / 2]
        )

        if f_r_dist < 50:
            v[0] -= v[1] * dt / 100
            v[1] += v[0] * dt / 100
        if f_l_dist < 50:
            v[0] += v[1] * dt / 100
            v[1] -= v[0] * dt / 100

        if front_dist < 60:
            v[0] *= 0.9
            v[1] *= 0.9
        elif front_dist > 200:
            v[0] *= 1.1
            v[1] *= 1.1

        if right_dist < 10:
            v[0] -= v[1] * dt / 3
            v[1] += v[0] * dt / 3
        if right_dist < 10:
            v[0] += v[1] * dt / 3
            v[1] -= v[0] * dt / 3

        x.append([x[-1][0] + v[0] * dt, x[-1][1] + v[1] * dt])

    x = np.array(x)
    x += np.random.normal(0, 0.001, size=(T + 1, 2))
    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = 100000
    data_a = accelerating_model(T)
    with open("dataset.csv", "w") as f:
        for i in data_a:
            f.write(", ".join([str(j) for j in i]) + "\n")

    fig, ax = plt.subplots(ncols=1)
    
    ax.plot(data_a[:,0])
    ax.plot(data_a[:,1])
    
    
    plt.show()
    
    plt.plot(data_a[:,0], data_a[:,1])
    
    plt.show()
    

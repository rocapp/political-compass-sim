import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(23)


def fun(t, x, *p):
    Auth = x[:, 0]
    Lib = x[:, 1]
    Left = x[:, 2]
    Right = x[:, 3]
    Happy = x[:, 4]
    Money = x[:, 5]
    Money_now = expit(Money[t]) - 0.5
    Happy_now = expit(Happy[t]) - 0.5
    Pop = x[:, 6]
    Pop_now = expit(Pop[t] - 0.5)
    return np.array([
        Auth[t] - p[0] * Auth[t]**3 / 9.0 + p[1] * 3.0 * Happy_now * expit(Auth[t])**3 - p[2] * Lib[t] * expit(Auth[t]),
        Lib[t] - p[3] * Lib[t]**3 / 9.0 + p[4] * 3.0 * Happy_now * expit(Lib[t])**3 - p[5] * Auth[t] * expit(Lib[t]),
        Left[t] - p[6] * Left[t]**3 / 9.0 + p[7] * 3.0 * Happy_now * expit(Left[t]-Right[t])**3 - p[8] * (Right[t] * expit(Left[t]-Right[t]))**3,
        Right[t] - p[9] * Right[t]**3 / 9.0 + p[10] * 3.0 * Happy_now * expit(Right[t]-Left[t])**3 - p[11] * (Left[t] * expit(Right[t]-Left[t]))**3,
        p[12] * Happy[t] + p[13] * (Money_now)**3 / 2.0 - p[14] * abs(Auth[t]+Lib[t]) * expit(0.1*(Auth[t]-Lib[t])**2-0.5) / 18.0 - p[15] * abs(Right[t]+Left[t]) * expit(0.1*(Right[t]-Left[t])**2-0.5) / 18.0 + p[16] * 1e-9 * np.random.uniform(),
        p[19] * Money[t] * expit(Pop[t]) * (1.0 - Money[t]) + p[18] * Money[t] * expit(Happy[t]) + np.random.uniform(-p[18], p[17]) * Pop_now * expit(Money[t]) / 23.0,
        p[20] * ( 1.0 + p[21] * (Money_now+Happy_now) * expit(Auth[t]*Right[t]) ) * Pop[t] * (1.0 - Pop[t])
    ], dtype=float)


def calc_pt(x):
    pt = np.array([ x[3] - x[2], x[0] - x[1] ], dtype=float)
    pt[pt < -1.0] = -1.0
    pt[pt > 1.0] = 1.0
    return pt


if __name__=='__main__':

    p = np.ones(22, dtype=float)

    p[19] = 1.0  # money growth rate
    p[20] = 0.5  # pop. growth rate

    Auth0 = 0.2
    Lib0 = 0.5
    Left0 = 0.5
    Right0 = 0.1
    
    Happy0 = 0.0
    Money0 = 0.3
    Pop0 = 0.2

    tvec = np.arange(100)
    x = np.zeros((tvec.size, 7), dtype=float)
    x[0, :] = [ Auth0, Lib0, Left0, Right0, Happy0, Money0, Pop0 ]
    startpt = calc_pt(x[0, :])
    print('x0', startpt)
    h = 1e-3
    for t in tvec[:-1]:
        # print(t)
        t0 = t
        t1 = t + 1.0
        while True:
            x[t+1, -1] = max(x[t+1, -1], 0.0)
            x[t+1, :] += h * fun(t, x, *p)
            t0 += h
            if t0 >= t1:
                break
    print('done.')
    
    endpt = calc_pt(x[-1, :])
    print('x1', endpt)

    df = pd.DataFrame(x, columns=['auth', 'lib', 'left', 'right', 'happy', 'money', 'pop'], index=tvec)
    ptx = calc_pt(x.T)

    plt.close('all')
    plt.ion()
    fig, axs = plt.subplots(nrows=2, ncols=1)
    df.plot(ax=axs[0])
    axs[1].axhline(0, xmin=-1, xmax=1, color='k', linestyle='--')
    axs[1].axvline(0, ymin=-1, ymax=1, color='k', linestyle='--')
    axs[1].plot(startpt[0], startpt[1], markersize=10.0, marker='x', color='g', label='start')
    axs[1].plot(endpt[0], endpt[1], markersize=10.0, marker='x', color='r', label='end')
    axs[1].plot(ptx[0], ptx[1], marker='.', color='b', alpha=0.7, label='trek')  # df.right - df.left, df.auth - df.lib
    axs[1].legend()
    axs[1].set_xlim([-1.1, 1.1])
    axs[1].set_ylim([-1.1, 1.1])
    

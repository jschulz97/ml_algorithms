import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


class FCM:

    def __init__(self, C: int, m: float):
        self.C = C  # type: int
        self.m = m  # type: float
        self.U = None  # type: Optional[np.ndarray]
        self.V = None  # type: Optional[np.ndarray]

    def fit(self, x: np.ndarray, max_iters: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float, float]:

        # Preallocate all arrays
        U = np.random.rand(self.C, x.shape[0])
        Um = np.power(U, self.m)
        V = np.empty((self.C, x.shape[1]), dtype=U.dtype)

        Vshape0_ones = np.ones(V.shape[0], dtype=U.dtype)
        Xshape1_ones = np.ones(x.shape[1], dtype=U.dtype)
        c_ones = np.ones(self.C, dtype=U.dtype)

        # Randomly initialize c-by-n membership matrix uniformly with rows summing to one
        U /= np.sum(U, axis=0)

        t = 0
        error = tol + 1  # initialize error large enough so that it allows first loop

        # Loop until either max iterations are reached or memberships don't change
        while t < max_iters and error > tol:
            V = np.divide(np.dot(Um, x), np.outer(np.sum(Um, axis=1), Xshape1_ones)) # shape: (c, p)
            d = (-2*np.dot(x, V.T) + np.sum(V**2, axis=1) + np.outer(np.sum(x**2, axis=1), Vshape0_ones)).T
            try:
                d = np.power(d, -1. / (self.m - 1))  # shape: (c, n)
            except Exception as e:
                print(e)
                print(d)
                print(self.m)
            U_new = np.divide(d, np.outer(c_ones, np.sum(d, axis=0)))  # shape: (c, n)
            error = np.linalg.norm(U_new - U)
            U = U_new.copy()
            Um = np.power(U, self.m)  # shape: (c, n)
            t += 1

        self.U = U
        self.V = V

        return U, V, t, error


    def predict(self, x: np.ndarray) -> np.ndarray:
        d = (-2 * np.dot(x, self.V.T) + np.sum(self.V ** 2, axis=1) + np.outer(np.sum(x ** 2, axis=1), np.ones(self.V.shape[0], dtype=self.U.dtype))).T
        d = np.power(d, -1. / (self.m - 1))  # shape: (c, n)
        return np.divide(d, np.outer(np.ones(self.C, dtype=self.U.dtype), np.sum(d, axis=0)))  # shape: (c, n)


def _toy_experiment():
    x = np.vstack((
        np.random.multivariate_normal(mean=(-1, -1), cov=np.eye(2) / 5, size=50),
        np.random.multivariate_normal(mean=(1, 1), cov=np.eye(2) / 5, size=50)
    ))
    labels = np.hstack((
        np.zeros(50),
        np.ones(50)
    ))
    colors = [u"#1f77b4", u"#ff7f0e", u"#2ca02c", u"#d62728", u"#9467bd", u"#8c564b", u"#e377c2", u"#7f7f7f", u"#bcbd22", u"#17becf"]

    m = 2
    C = 2
    fcm = FCM(C=C, m=m)
    U, V, iters, error = fcm.fit(x)

    x_min, y_min = np.amin(x, axis=0) - 1
    x_max, y_max = np.amax(x, axis=0) + 1
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    memberships = fcm.predict(np.dstack((X, Y)).reshape(-1, 2)).reshape((C, 100, 100))
    membership_plot = plt.imshow(np.amax(memberships, axis=0), extent=[x_min, x_max, y_max, y_min])

    plt.scatter(x[labels == 0, 0], x[labels == 0, 1], color="black", marker="+")
    plt.scatter(x[labels == 1, 0], x[labels == 1, 1], color="black", marker="+")
    for i in range(C):
        plt.scatter(V[i, 0], V[i, 1], color=colors[i], label="Centroid %d" % (i+1))
    plt.colorbar(membership_plot)
    plt.legend(loc="upper right")
    plt.title("Fuzzy c-Means Example")
    plt.show()


def _big_toy_experiment():
    x = np.vstack((
        np.random.normal(loc=1, scale=0.2, size=(5000, 9000)),
        np.random.normal(loc=-1, scale=0.2, size=(5000, 9000))
    ))
    labels = np.hstack((
        np.zeros(5000),
        np.ones(5000)
    ))
    colors = ["blue", "green", "red"]

    fcm = FCM(C=2, m=2)
    U, V, _, _ = fcm.fit(x)

    plt.scatter(x[labels == 0, 0], x[labels == 0, 1], color=colors[0])
    plt.scatter(x[labels == 1, 0], x[labels == 1, 1], color=colors[1])
    plt.scatter(V[:, 0], V[:, 1], color=colors[2])

    plt.show()


if __name__ == "__main__":
    _toy_experiment()

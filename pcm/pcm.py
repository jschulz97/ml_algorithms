import logging
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from .fcm import FCM


class PCM:

    def __init__(self, C: int, m: Union[Callable[[int], float], float], etas: Optional[np.ndarray] = None):

        if isinstance(m, int):
            m = float(m)

        # PCM parameters
        self.C = C                                              # type: int
        self.m = (lambda t: m) if type(m) == float else m       # type: Callable[[int], float]
        self.etas = etas                                        # type: Optional[np.ndarray]
        self.U = None                                           # type: Optional[np.ndarray]
        self.V = None                                           # type: Optional[np.ndarray]

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
        self.logger.addHandler(self.console_handler)


    def fit(self,
            x: np.ndarray,
            initialization: Union[str, Dict[str, np.ndarray]] = "fcm",
            max_iters: int = 100,
            tol: float = 1e-6,
            progress_callables: List[Callable[[dict], None]] = None
    ) -> (np.ndarray, np.ndarray, int, float):

        if progress_callables is None:
            progress_callables = []

        self.U, self.V, d_sq = PCM.initialize(x, self.C, self.m(0), initialization)

        Um = np.power(self.U, self.m(0))
        x_max = np.amax(x, axis=0)
        x_min = np.amin(x, axis=0)

        # Compute etas if not provided
        if self.etas is None:
            self.etas = np.divide(np.sum(np.multiply(Um, d_sq), axis=1), np.sum(Um, axis=1))
            #if alpha is None:
            #    alpha = 0.5 #np.min(np.amin(U, axis=1))
            #self.etas = np.sum(np.multiply(d_sq, U > alpha), axis=1) / np.sum(U > alpha, axis=1)

        # Initialization
        l = 0
        m = self.m(l)
        U_old = self.U.copy()
        error = np.inf

        for f in progress_callables:
            f(locals())

        # try:
        # PCM algorithm
        while l < max_iters and error > tol:
            self.U = np.divide(1., 1 + np.power(np.divide(d_sq, np.repeat(self.etas[:, np.newaxis], x.shape[0], axis=1)), 1./(m - 1)))
            Um = np.power(self.U, m)
            self.V = np.divide(np.matmul(Um, x), np.outer(np.sum(Um, axis=1), np.ones(x.shape[1])))
            d_sq = euclidean_distances(self.V, x, squared=True)

            # Loop condition updates
            error = np.linalg.norm(self.U - U_old, ord=2)
            U_old = self.U.copy()
            l += 1

            m = self.m(l)

            for f in progress_callables: f(locals())
        # except:
        #     print('PCM failure')

        return self.U, self.V, l, error


    def predict(self, x: np.ndarray, m: float) -> np.ndarray:
        d_sq = euclidean_distances(self.V, x, squared=True)
        etas = np.repeat(self.etas[:, np.newaxis], x.shape[0], axis=1)
        return np.divide(1, 1 + np.power(np.divide(d_sq, etas), 1./(m-1)))


    @staticmethod
    def initialize(x: np.ndarray, C: int, m: float = None, initialization: Union[str, Dict[str, np.ndarray]] = "fcm"):
        logger = logging.getLogger(__name__)
        if type(initialization) == dict:
            U = initialization["U_init"]
            V = initialization["V_init"]
            d_sq = euclidean_distances(V, x, squared=True)
            message = "[DEBUG] Using provided initialization"
        elif type(initialization) == str and initialization.lower() == "fcm":
            assert m is not None, "m must be specified for FCM initialization."
            U, V, _, _ = FCM(C=C, m=m).fit(x)
            d_sq = euclidean_distances(V, x, squared=True)
            message = "[DEBUG] Initializing PCM with FCM centroids"
        elif type(initialization) == str and initialization.lower() == "random":
            assert m is not None, "m must be specified for random initialization."
            V = np.random.rand(C, x.shape[1]) * (np.amax(x, axis=0) - np.amin(x, axis=0))
            d_sq = euclidean_distances(V, x, squared=True)
            U = np.divide(1., 1 + np.power(d_sq, 1. / (m - 1)))
            message = "[DEBUG] Uniform random centroid initialization."
        else:
            raise ValueError("Invalid initialization parameter.")
        if logger is not None:
            logger.debug(message)
        return U, V, d_sq


    @staticmethod
    def progress_plot_etas(ax: plt.Axes, locals_: Dict[str, Any]) -> None:
        etas = locals_["self"].etas
        etas_old = locals_["etas_old"]
        t = locals_["l"]
        if t > 0:
            for i in range(len(etas)):
                ax.plot([t-1, t], [etas_old[i], etas[i]], c="k", label="Cluster %d" % (i+1))
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Eta")
            ax.set_title("Eta values per cluster over time")


    @staticmethod
    def progress_plot_2d_centroids(ax: plt.Axes, locals_: Dict[str, Any], title: str = None, do_legend: bool = True, centroid_colors: List[str] = None) -> None:

        # This is stupid but it prevents my older code from breaking
        if "V" in locals_ and "V_override" in locals_:
            locals_["self"].V = locals_["V"]

        ax.scatter(locals_["x"][:, 0], locals_["x"][:, 1], c="k", marker="+", label="Data")
        for i in range(locals_["self"].V.shape[0]):
            ax.scatter(locals_["self"].V[i, 0], locals_["self"].V[i, 1], c="red" if centroid_colors is None else centroid_colors[i], marker="o", label="Cluster %d" % (i + 1))
        if title is not None:
            ax.set_title(title)
        if do_legend:
            ax.legend(loc="lower right")


    @staticmethod
    def progress_plot_decision(ax: plt.Axes, locals_: Dict[str, Any], do_colorbar: bool = True, title: str = None, region: (float, float, float, float) = None) -> None:

        # This is stupid but it prevents my older code from breaking
        if "V" in locals_ and "V_override" in locals_:
            locals_["self"].V = locals_["V"]
        if "etas" in locals_ and "etas_override" in locals_:
            locals_["self"].etas = locals_["etas"]

        if region is None:
            x_min, y_min = np.amin(locals_["x"], axis=0)
            x_max, y_max = np.amax(locals_["x"], axis=0)
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min, y_min = x_min - 0.1 * x_range, y_min - 0.1 * y_range
            x_max, y_max = x_max + 0.1 * x_range, y_max + 0.1 * y_range
        else:
            x_min, x_max, y_min, y_max = region


        X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        memberships = locals_["self"].predict(np.dstack((X, Y)).reshape(-1, 2), locals_["m"]).reshape((locals_["self"].C, 100, 100))
        membership_plot = ax.imshow(np.linalg.norm(memberships, axis=0), extent=[x_min, x_max, y_max, y_min])
        if do_colorbar:
            if hasattr(ax, "colorbar_handle"):
                ax.colorbar_handle.remove()
            ax.colorbar_handle = plt.colorbar(membership_plot, ax=ax)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if title is not None:
            ax.set_title(title)


if __name__ == "__main__":
    pass

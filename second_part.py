import numpy as np
import matplotlib.pyplot as plt
import time

from NiaPy.algorithms import BasicStatistics
from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Benchmark
from scipy.optimize import OptimizeResult
from scipy.optimize import rosen, minimize
from scipy.interpolate import Rbf

class noisy_rosenbrock(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, -3.0, 3.0)

    def function(self):
        def evaluate(D, sol):
            return rbf(*sol)
        return evaluate


def sgd(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.0024,
    mass=0.9,
    startiter=0,
    maxiter=500,
    callback=None,
    **kwargs
):
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):

        if callback and callback(x):
            break

        if jac == None:
            g = np.gradient((fun(benc)(len(x), x), fun(benc)(len(x), x+0.01)), 0.01)
        else:
            g = jac(x)

        velocity = mass * velocity - (1.0 - mass) * g
        x1 = x + learning_rate * velocity

    i += 1
    return OptimizeResult(x=x1, fun=fun(benc)(len(x), x), jac=g, nit=i, nfev=i, success=True)


if __name__ == "__main__":
    nruns = 10
    dim = 2

    x = np.linspace(-3, 3, 20)
    x, y = np.meshgrid(x, x)
    noise = np.random.uniform(size=(np.shape(x)))
    u = rosen((x, y)) + noise
    rbf = Rbf(x, y, u)

    benc = noisy_rosenbrock()
    benc.plot3d()

    Time = list()
    x_fss = list()
    y_fss = list()
    stats = np.zeros(nruns)
    print("Fish School Search")
    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=50, optType=OptimizationType.MINIMIZATION,
                            benchmark=noisy_rosenbrock())

        algo = FishSchoolSearch(NP=30, SI_init=0.5, SI_final=3, SV_init=0.3,
                                SV_final=7, min_w=0.1, w_scale=0.7)

        timer = time.perf_counter()
        best = algo.run(task)
        Time.append(time.perf_counter() - timer)

        stats[i] = best[1]
        x_fss.append(best[0][0])
        y_fss.append(best[0][1])

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))

    Time = list()
    x_sgd = list()
    y_sgd = list()
    stats = np.zeros(nruns)
    print("\nStochastic Gradient Descent")
    for i in range(nruns):
        x0 = np.random.uniform(size=dim)
        timer = time.perf_counter()
        res_sgd = minimize(noisy_rosenbrock.function, x0, method=sgd, jac=None)
        Time.append(time.perf_counter() - timer)

        stats[i] = res_sgd.fun
        x_sgd.append(res_sgd.x[0])
        y_sgd.append(res_sgd.x[1])

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))

    Time = list()
    x_spo = list()
    y_spo = list()
    stats = np.zeros(nruns)
    print("\nParticleSwarmAlgorithm")
    from NiaPy.algorithms.basic import ParticleSwarmAlgorithm

    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=50, optType=OptimizationType.MINIMIZATION,
                            benchmark=noisy_rosenbrock())

        algo = ParticleSwarmAlgorithm(NP=30, C1=2.0, C2=2.0, w=0.8, vMin=-1, vMax=1)

        timer = time.perf_counter()
        best = algo.run(task)
        Time.append(time.perf_counter() - timer)

        stats[i] = best[1]
        x_spo.append(best[0][0])
        y_spo.append(best[0][1])

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))

    x = np.linspace(min(x_spo + x_fss + x_sgd) - 0.3, max(x_spo + x_fss + x_sgd) + 0.3)
    y = np.linspace(min(y_spo + y_fss + y_sgd) - 0.3, max(y_spo + y_fss + y_sgd) + 0.3)
    x, y = np.meshgrid(x, y)
    plt.contourf(x, y, noisy_rosenbrock.function(benc)(dim, (x, y)))

    plt.scatter(x_spo, y_spo, label="SPO")
    plt.scatter(x_fss, y_fss, label="FSS")
    plt.scatter(x_sgd, y_sgd, label="SGD")

    plt.legend()
    plt.show()
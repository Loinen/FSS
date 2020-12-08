import numpy as np
import matplotlib.pyplot as plt
import time

from NiaPy.algorithms import BasicStatistics
from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Benchmark
from scipy.optimize import OptimizeResult
from scipy.optimize import rosen_der, minimize

class noisy_rosenbrock(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, -30, 30)

    def function(self):
        def evaluate(D, sol):
            val = sum(100.0*(sol[1:] - sol[:-1]**2.0)**2.0 + (1 - sol[:-1])**2.0)
            val += np.random.uniform()
            return val
        return evaluate

def noisy_rosenbrock_func(sol):
    val = sum(100.0 * (sol[1:] - sol[:-1] ** 2.0) ** 2.0 + (1 - sol[:-1]) ** 2.0)
    val += np.random.uniform()
    return val


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
        g = jac(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


if __name__ == "__main__":
    nruns = 10
    dim = 2

    Time = list()
    stats = np.zeros(nruns)
    print("Fish School Search")
    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=500, optType=OptimizationType.MINIMIZATION,
                            benchmark=noisy_rosenbrock())

        algo = FishSchoolSearch(NP=30, SI_init=0.5, SI_final=3, SV_init=0.3,
                                SV_final=7, min_w=0.1, w_scale=0.7)

        timer = time.perf_counter()
        best = algo.run(task)
        Time.append(time.perf_counter() - timer)

        stats[i] = best[1]

        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))

    Time = list()
    stats = np.zeros(nruns)
    print("\nStochastic Gradient Descent")
    for i in range(nruns):
        x0 = np.random.uniform(size=dim)
        timer = time.perf_counter()
        res_sgd = minimize(noisy_rosenbrock_func, x0, method=sgd, jac=rosen_der)
        Time.append(time.perf_counter() - timer)

        stats[i] = res_sgd.fun

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))

    Time = list()
    stats = np.zeros(nruns)
    print("\nParticleSwarmAlgorithm")
    from NiaPy.algorithms.basic import ParticleSwarmAlgorithm

    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=500, optType=OptimizationType.MINIMIZATION,
                            benchmark=noisy_rosenbrock())

        algo = ParticleSwarmAlgorithm(NP=30, C1=2.0, C2=2.0, w=0.8, vMin=-1, vMax=1)

        timer = time.perf_counter()
        best = algo.run(task)
        Time.append(time.perf_counter() - timer)

        stats[i] = best[1]

        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())
    print("Execution time ", np.mean(Time))
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from NiaPy import Runner
from NiaPy.algorithms import BasicStatistics
from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Rastrigin
from NiaPy.benchmarks import Rosenbrock
from NiaPy.benchmarks import Ackley


# np - размер популяции,
# параметры и критерии остановки: d - размер задачи, nfes - Number of function evaluations
# ngen - Number of generations or iterations
# refValue (Optional[float]): Reference value of function/fitness function.
# logger (Optional[bool]): Enable/disable logging of improvements.

# set params FSS - NP (Optional[int]) – Number of fishes in school.
# SI_init (Optional[int]) – Length of initial individual step.
# SI_final (Optional[int]) – Length of final individual step.
# SV_init (Optional[int]) – Length of initial volatile step.
# SV_final (Optional[int]) – Length of final volatile step.
# min_w (Optional[float]) – Minimum weight of a fish.
# w_scale (Optional[float]) – Maximum weight of a fish.

def start(benc, nruns, dim):
    benc.plot3d()
    fssTime = list()
    frfTime = list()
    psTime = list()

    stats = np.zeros(nruns)
    print("\nstarting FSS ", benc)
    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = FishSchoolSearch(NP=40, SI_init=0.3, SI_final=5, SV_init=0.3, SV_final=5, min_w=0.2, w_scale=0.8)
        timer = time.perf_counter()
        best = algo.run(task)
        fssTime.append(time.perf_counter() - timer)
        # task.plot()
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()
        # print("FSS", best[-1])
        # print(evals)  # print function evaluations
        # print(x_f)  # print values

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report
    print(algo.getParameters())
    print("time", np.mean(fssTime))

    stats = np.zeros(nruns)
    print("\nstarting FireflyAlgorithm", benc)
    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = FireflyAlgorithm(NP=40, alpha=0.2, betamin=0.2, gamma=0.5)
        timer = time.perf_counter()
        best = algo.run(task)  # возвращает наилучший найденный минимум
        frfTime.append(time.perf_counter() - timer)
        # task.plot()
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report
    print(algo.getParameters())
    print("time", np.mean(frfTime))

    stats = np.zeros(nruns)
    print("\nstarting ParticleSwarm", benc)
    for i in range(nruns):
        task = StoppingTask(D=dim, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.8, vMin=-1, vMax=1.5)
        # C1=2.0, C2=2.0, w=0.7, vMin=-1.5, vMax=1.5, c - когнитивный и социальный компонент
        # w - инерционный вес, v - минимальная и максимальная скорость
        timer = time.perf_counter()
        best = algo.run(task)  # возвращает наилучший найденный минимум
        psTime.append(time.perf_counter() - timer)
        # task.plot()
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report
    print(algo.getParameters())
    print("time", np.mean(psTime))


if __name__ == '__main__':
    runner = Runner(
        D=40,
        nFES=100,
        nRuns=2,
        useAlgorithms=[
            ParticleSwarmAlgorithm(),
            FishSchoolSearch(),
            FireflyAlgorithm()],
        useBenchmarks=[
            Ackley(),
            Rastrigin(),
            Rosenbrock()
        ]
    )

    # runner.run(export='dataframe', verbose=True) # Returns dictionary of results
    # unpickled_df = pd.read_pickle("./export/2020-12-02 13.57.04.047304.pkl")
    # pd.set_option('display.max_colwidth', -1)
    # print(unpickled_df)

    nruns = 10
    benc = Rosenbrock() # минимум 0
    start(benc, nruns, 2)
    benc = Rastrigin() # минимум 0, трудная задача, т.к. много локальных минимумов
    start(benc, nruns, 2)
    benc = Ackley() # минимум 0, еще больше локальных минимумов
    start(benc, nruns, 10)


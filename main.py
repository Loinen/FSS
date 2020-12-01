import numpy as np
import matplotlib.pyplot as plt
import time

import NiaPy as nia
from NiaPy.algorithms import BasicStatistics
from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Rastrigin
from NiaPy.benchmarks import Rosenbrock
from NiaPy.benchmarks import Ackley


# we will run Grey Wolf Optimizer for 5 independent runs (Rosenbrock)
# np - размер популяции, d - размер задачи, nfes - Number of function evaluations
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

def start(benc, nruns):
    benc.plot3d()
    fssTime = list()
    frfTime = list()
    psTime = list()

    stats = np.zeros(nruns)
    print("\nstarting FSS ", benc)

    for i in range(nruns):
        task = StoppingTask(D=2, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = FishSchoolSearch(NP=40, SI_init=0.3, SI_final=1, SV_init=0.3, SV_final=1, min_w=0.2, w_scale=0.8)
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
        task = StoppingTask(D=2, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = FireflyAlgorithm(NP=40)
        # alpha=1, betamin=1, gamma=2 - параметры светлячков, хз что значит
        timer = time.perf_counter()
        best = algo.run(task)  # возвращает наилучший найденный минимум
        frfTime.append(time.perf_counter() - timer)
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report
    print(algo.getParameters())
    print("time", np.mean(frfTime))

    stats = np.zeros(nruns)
    print("\nstarting ParticleSwarm", benc)
    for i in range(nruns):
        task = StoppingTask(D=2, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=benc)
        algo = ParticleSwarmAlgorithm(NP=40)
        # C1=2.0, C2=2.0, w=0.7, vMin=-1.5, vMax=1.5, c - когнитивный и социальный компонент
        # w - инерционный вес, v - минимальная и максимальная скорость
        timer = time.perf_counter()
        best = algo.run(task)  # возвращает наилучший найденный минимум
        psTime.append(time.perf_counter() - timer)
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report
    print(algo.getParameters())
    print("time", np.mean(psTime))


if __name__ == '__main__':
    # x = np.random.normal(size=1000)
    # x1 = np.random.normal(size=1000)
    # fss = nia.algorithms.basic.FishSchoolSearch(Name=['FSS', 'FishSchoolSearch'], school=100)
    # fss.gen_weight()
    # fss.setParameters(NP=100, SI_init=0.3, SI_final=1, SV_init=0.3, SV_final=1, min_w=0.3, w_scale=0.7)
    # gw = nia.algorithms.basic.GreyWolfOptimizer(Name=['GreyWolfOptimizer', 'GWO'], pop=10, task=rosen)
    # print(fss.getBest(X=x, X_f=x1), fss.bad_run())

    nruns = 5
    benc = Rosenbrock()
    start(benc, nruns)
    benc = Rastrigin()
    start(benc, nruns)
    benc = Ackley()
    start(benc, nruns)


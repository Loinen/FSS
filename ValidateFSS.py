import numpy as np
import matplotlib.pyplot as plt
import NiaPy as nia
from NiaPy.algorithms import BasicStatistics
from NiaPy.algorithms.basic import GreyWolfOptimizer
from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Rastrigin
from NiaPy.benchmarks import Rosenbrock
from NiaPy.benchmarks import Ackley


if __name__ == '__main__':
    x = np.random.normal(size=1000)
    x1 = np.random.normal(size=1000)
    fss = nia.algorithms.basic.FishSchoolSearch(Name=['FSS', 'FishSchoolSearch'], school=100)
    fss.gen_weight()
    fss.setParameters(NP=100, SI_init=0.3, SI_final=1, SV_init=0.3, SV_final=1, min_w=0.3, w_scale=0.7)
    # fss.runIteration(nia.benchmarks.rosenbrock, x, x, x1, 1.5, 1, 2, 3, 4)
    # gw = nia.algorithms.basic.GreyWolfOptimizer(Name=['GreyWolfOptimizer', 'GWO'], pop=10, task=rosen)
    print(fss.getBest(X=x, X_f=x1), fss.bad_run())
    print(fss.getParameters())
    #print(gw.getBest(X=x, X_f=x1), gw.bad_run())
    #print(gw.getParameters())
    print("starting FSS")

    # we will run Grey Wolf Optimizer for 5 independent runs (Rosenbrock)
    # np - размер популяции, d - размер задачи, nfes - Number of function evaluations
    # ngen - Number of generations or iterations
    # refValue (Optional[float]): Reference value of function/fitness function.
    # logger (Optional[bool]): Enable/disable logging of improvements.

    # for i in range(5):
    #     task = StoppingTask(D=2, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Rosenbrock())
    #     algo = GreyWolfOptimizer(NP=40)
    #     best = algo.run(task) # возвращает наилучший найденный минимум
    #     print(best[-1])
    nruns = 5
    stats = np.zeros(nruns)
    for i in range(nruns):
        task = StoppingTask(D=2, nGEN=100, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Rosenbrock())
        algo = FishSchoolSearch(NP=40, SI_init=0.3, SI_final=1, SV_init=0.3, SV_final=1, min_w=0.2, w_scale=0.8)
        # plt.plot(Rosenbrock())
        best = algo.run(task)
        # print("FSS", best[-1])
        # task.plot()
        stats[i] = best[1]  # save best
        evals, x_f = task.return_conv()
        print(evals)  # print function evaluations
        print(x_f)  # print values

    stat = BasicStatistics(stats)
    print(stat.generate_standard_report())  # generate report

    # main()

# set params FSS - NP (Optional[int]) – Number of fishes in school.
# SI_init (Optional[int]) – Length of initial individual step.
# SI_final (Optional[int]) – Length of final individual step.
# SV_init (Optional[int]) – Length of initial volatile step.
# SV_final (Optional[int]) – Length of final volatile step.
# min_w (Optional[float]) – Minimum weight of a fish.
# w_scale (Optional[float]) – Maximum weight of a fish.


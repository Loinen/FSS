from FSS import FSS
from ObjectiveFunction import *
from SearchSpaceInitializer import UniformSSInitializer, OneQuarterDimWiseSSInitializer
import numpy as np
import os
import NiaPy as nia
from NiaPy.algorithms.basic import GreyWolfOptimizer
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere
from NiaPy.benchmarks import rosenbrock


def create_dir(path):
    directory = os.path.dirname(path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


def main():
    search_space_initializer = UniformSSInitializer()
    file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + "Executions" + os.sep
    num_exec = 30
    school_size = 30
    num_iterations = 10000
    step_individual_init = 0.1
    step_individual_final = 0.0001
    step_volitive_init = 0.01
    step_volitive_final = 0.001
    min_w = 1
    w_scale = num_iterations / 2.0

    dim = 30

    regular_functions = [SphereFunction, RosenbrockFunction, RastriginFunction, SchwefelFunction,
                         GriewankFunction, AckleyFunction]

    regular_functions = [AckleyFunction]

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    cec_functions = [ ]

    for benchmark_func in regular_functions:
        func = benchmark_func(dim)
        run_experiments(num_iterations, school_size, num_exec, func, search_space_initializer, step_individual_init,
                        step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale, file_path)


def run_experiments(n_iter, school_size, num_runs, objective_function, search_space_initializer, step_individual_init,
                    step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale, save_dir):
    alg_name = "FSS"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {}"
    if save_dir:
        create_dir(save_dir)
        f_handle_cost_iter = open(save_dir + "/FSS_" + objective_function.function_name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = open(save_dir + "/FSS_" + objective_function.function_name + "_cost_eval.txt", 'w+')

    for run in range(num_runs):
        opt1 = FSS(objective_function=objective_function, search_space_initializer=search_space_initializer,
                   n_iter=n_iter, school_size=school_size, step_individual_init=step_individual_init,
                   step_individual_final=step_individual_final, step_volitive_init=step_volitive_init,
                   step_volitive_final=step_volitive_final, min_w=min_w, w_scale=w_scale)

        opt1.optimize()
        print (console_out.format(alg_name, objective_function.function_name, run+1, opt1.best_fish.cost))

        temp_optimum_cost_tracking_iter = np.asmatrix(opt1.optimum_cost_tracking_iter)
        temp_optimum_cost_tracking_eval = np.asmatrix(opt1.optimum_cost_tracking_eval)

        if save_dir:
            np.savetxt(f_handle_cost_iter, temp_optimum_cost_tracking_iter, fmt='%.4e')
            np.savetxt(f_handle_cost_eval, temp_optimum_cost_tracking_eval, fmt='%.4e')

    if save_dir:
        f_handle_cost_iter.close()
        f_handle_cost_eval.close()


def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)


if __name__ == '__main__':
    x = np.random.normal(size=1000)
    x1 = np.random.normal(size=1000)
    fss = nia.algorithms.basic.FishSchoolSearch(Name=['FSS', 'FishSchoolSearch'], school=100)
    fss.gen_weight()
    fss.setParameters(NP=100, SI_init=0.3, SI_final=1, SV_init=0.3, SV_final=1, min_w=0.3, w_scale=0.7)
    # fss.runIteration(nia.benchmarks.rosenbrock, x, x, x1, 1.5, 1, 2, 3, 4)
    gw = nia.algorithms.basic.GreyWolfOptimizer(Name=['GreyWolfOptimizer', 'GWO'], pop=10, task=rosen)
    print(fss.getBest(X=x, X_f=x1), fss.bad_run())
    print(fss.getParameters())
    print(gw.getBest(X=x, X_f=x1), gw.bad_run())
    print(gw.getParameters())
    print("starting FSS")

    # we will run Grey Wolf Optimizer for 5 independent runs
    for i in range(5):
        task = StoppingTask(D=5, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
        algo = GreyWolfOptimizer(NP=40)
        best = algo.run(task)
        print(best)
    # main()




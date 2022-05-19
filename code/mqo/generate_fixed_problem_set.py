import numpy as np
import math
import random
import csv
import argparse
import os

from sklearn.model_selection import train_test_split

import pickle

##################################
#Functions for problem generation#
##################################

def create_savings(n_queries, n_plans, savings_min=-20, savings_max=0):
    savings = {}
    for j in range(n_plans*n_queries):
        current_query = math.floor(j / n_plans)
        first_plan_next_query = (current_query + 1) * n_plans
        for i in range(first_plan_next_query, n_queries*n_plans):
            savings[j, i] = random.randint(savings_min, savings_max)
    return savings

def generate_problems(n_queries, n_plans, size, cost_min=0, cost_max=50):
    problems = []
    for i in range(size):
        problems.append((n_queries, np.random.randint(cost_min, cost_max, int(n_queries*n_plans)), create_savings(n_queries, n_plans)))
    return problems

def extract_values(dataset):
    values = []
    for row in dataset:
        values.append(
            np.concatenate((row[1].tolist(),list(row[2].values())))
            )
    return np.array(values)

def create_solution_set(problems):
    y = []
    y_complete = []
    for problem in problems:
        #### calculate totally cheapest option
        t_cost = []
        t_cost.append([problem[0]+problem[2]+problem[4], problem[0]+problem[3]+problem[5]])
        t_cost.append([problem[1]+problem[2]+problem[6], problem[1]+problem[3]+problem[7]])
        y.append(np.array(t_cost).argmin())
        #### calculate sorted ranking of cost
        t_total = {}
        t_total[0] = problem[0]+problem[2]+problem[4]
        t_total[1] = problem[0]+problem[3]+problem[5]
        t_total[2] = problem[1]+problem[2]+problem[6]
        t_total[3] = problem[1]+problem[3]+problem[7]
        y_complete.append(t_total)
    return y, y_complete

def extract_values(dataset):
    values = []
    for row in dataset:
        values.append(
            np.concatenate((row[1].tolist(),list(row[2].values())))
            )
    return np.array(values)


################
#Main functions#
################
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generates a pickle file with problems to solve")
    parser.add_argument('-s', '--size', help="Set how many problems to have", type=int, default=300)
    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    print('Problem Generator for 2x2 MQO')
    print('---------------------------------------------------')
    print('Creating {0} problems, with 2 queries, of which each has 2 plans'.format(args.size))
    problems = generate_problems(2, 2, args.size)
    problems_values = extract_values(problems)
    _, complete = create_solution_set(problems_values)
    x_train, x_test, y_train, y_test = train_test_split(problems_values, complete, test_size=0.25)
    y_test_ranked = []
    y_test_labels = []
    for sol in y_test:
        y_test_ranked.append(list({k: v for k, v in sorted(sol.items(), key=lambda item: item[1])}.keys()))
        y_test_labels.append(y_test_ranked[-1][0])
    
    data = {'x_train':x_train, 'x_test': x_test, 'y_train':y_train, 'y_test': y_test_labels, 'y_test_ranked': y_test_ranked}
    print('Generated all problems, saving them')
    try:
        os.makedirs('runs/data/')
    except:
        print('path already exists!')
    pickle.dump(data, open("runs/data/problems_with_solutions.p", "wb" ) )
    print('Finished execution.')
    print('---------------------------------------------------')



if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
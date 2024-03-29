from syslog import LOG_SYSLOG
from qiskit import *
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np
import math
import random
import csv
import argparse
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm


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

def scale_problems(problems, scale_min=-np.pi/4, scale_max=np.pi/4):
    scaled_problems = []
    scaler = MinMaxScaler((scale_min, scale_max))
    for problem in problems:
        scaled_problems.append(scaler.fit_transform(problem.reshape(-1,1)))
    return scaled_problems

def scale_set(set, scale_min=-np.pi/4, scale_max=np.pi/4):
    scaled = []
    scaler = MinMaxScaler((scale_min, scale_max))
    for entry in set:
        scaled.append(np.array(scaler.fit_transform(entry.reshape(-1, 1)).flatten()))
    return scaled

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




##################################
#Functions for circuit generation#
##################################
def uncertainity_principle(circuit):
    circuit.h(range(circuit.width()))
    circuit.barrier()

def cost_encoding(circuit):
    for i in range(circuit.width()):
        circuit.ry(-Parameter('c'+str(i)), i)
    circuit.barrier()

def same_query_cost(circuit):
    for i in range(0,circuit.width(),2):
        circuit.crz(-np.pi/4,i,i+1)
    circuit.barrier()

def savings_encoding(circuit):
        for i in range(int(circuit.width()/2)):
            circuit.crz(Parameter('s'+str(i)+str(int(circuit.width()/2))), i, int(circuit.width()/2))
            circuit.crz(Parameter('s'+str(i)+str(int(1+circuit.width()/2))), i, int(1+circuit.width()/2))
        circuit.barrier()

def rx_layer(circuit, weights):
    if len(weights) == 1:
        circuit.rx(weights[0], range(circuit.width()))
    else:
        for i, w in enumerate(weights):
            circuit.rx(w, i)
    circuit.barrier()

def ry_layer(circuit, weights):
    if len(weights) == 1:
        circuit.ry(weights[0], range(circuit.width()))
    else:
        for i, w in enumerate(weights):
            circuit.ry(w, i)
    circuit.barrier()

def rz_layer(circuit, weights):
    if len(weights) == 1:
        circuit.rz(weights[0], range(circuit.width()))
    else:
        for i, w in enumerate(weights):
            circuit.rz(w, i)
    circuit.barrier()

def create_circuit(n_queries, n_plans, scheme, weights):
    circuit = QuantumCircuit(n_queries*n_plans)
    params = [Parameter('w'+str(i)) for i in range(weights)]
    for module in scheme:
        if module == "h":
            uncertainity_principle(circuit)
        elif module == "c":
            cost_encoding(circuit)
        elif module == "s":
            savings_encoding(circuit)
        elif module == "x":
            rx_layer(circuit, params)
        elif module == "y":
            ry_layer(circuit, params)
        elif module == "z":
            rz_layer(circuit, params)
    return circuit


##############
#ML functions#
##############
def reorg_problems(problems):
    problems_reorg = []
    for a in problems:
        temp = []
        for b in a:
            temp.append(b[0])
        problems_reorg.append(temp)
    return problems_reorg

output_shape = 4
def parity(x):
    return x % 4

def callback(w, l):
    weights.append(w)
    loss.append(l)

def get_classifier(circuit, optimizer, quantum_instance, callback=callback):
    circuit_qnn = CircuitQNN(circuit=circuit,    
                         input_params=circuit.parameters[0:8],
                         weight_params=circuit.parameters[8:],
                         interpret=parity,
                         output_shape=output_shape,
                         quantum_instance=quantum_instance)

    # construct classifier
    return NeuralNetworkClassifier(neural_network=circuit_qnn,                                             
                                                optimizer=optimizer,
                                                callback=callback)

def fit_and_score(circuit_classifier, X_train, X_test, y_train, y_test, test_size=1/4):
    global weights
    global loss
    weights = []
    loss = []
    # fit classifier to data
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    circuit_classifier.fit(X_train, y_train)
    # return to default figsize
    score_train =  circuit_classifier.score(X_train, y_train)
    score_test =  circuit_classifier.score(X_test, y_test)
    print("Mean Accuracy training: " + str(score_train))
    print("Mean Accuracy testing: " + str(score_test))
    return score_train, score_test, weights[-1], loss



################
#Main functions#
################
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Runs the MQO Problem using ML optimization with different optimizers")
    parser.add_argument('-s', '--size', help="Set how many problems to have", type=int, default=300)
    parser.add_argument('-sh', '--shots', help="Shots for the simulation to run", type=int, default=1024)
    parser.add_argument('-i', '--iterations', help="How often each optimizer is run", type=int, default=13)
    parser.add_argument('-mi', '--maxiterations', help="Max iterations per optimizer", type=int, default=50)
    parser.add_argument('-g', '--gpu', help="Uses the gpu for the calculations", action="store_true")
    return parser.parse_args(argv)

def main(argv):

    args = parse_args(argv)
    print('Optimizer evaluation for MQO solving QCs')
    print('---------------------------------------------------')
    circuits = ['hcsx']
    weights = [1]
    

    print('Generating optimizers...')
    optimizers = []
    max_iterations = 150

    learning_rates = [0.005]
    beta_1s = [0.9]
    beta_2s = [0.85]
    noise_factors = [5e-08]
    ##AMSGRAD
    adams = []
    for lr in learning_rates:
        for b1 in beta_1s:
            for b2 in beta_2s:
                for noise in noise_factors:
                    adams.append(ADAM(maxiter=max_iterations, lr=lr, beta_1=b1, beta_2=b2, noise_factor=noise, amsgrad=True))
    optimizers.append(adams)

    optimizers_names = ['AMSGRAD']

    print('Generated {0} optimizers'.format(sum([len(a) for a in optimizers])))
    print('Loading existing problem collection')

    path = "runs/data/problems_with_solutions.p"
    data = pickle.load(open(path, "rb"))

    X_train = data['x_train']
    X_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    X_train = scale_set(X_train)
    X_test = scale_set(X_test)
    print('Generating all circuits')
    gen_circuits = []
    for w in weights:
        for c in circuits:
            gen_circuits.append(create_circuit(2, 2, c, w))
    print('Generated {0} circuits'.format(len(gen_circuits)))
    print('Setting up simulator')
    simulator = Aer.get_backend('qasm_simulator')
    if args.gpu:
        simulator.set_options(device='GPU')
    quantum_instance = QuantumInstance(simulator, shots=args.shots)
    print('Making sure path to save data exists...')
    try:
        os.makedirs('runs/experiments/optimizer_hyperparameters/')
    except:
        print('path already exists!')
    print('Running optimizer evaluation...')
    for i, opt_name in enumerate(optimizers_names):
        print('Evaluating optimizer ' + opt_name)
        optimizer_results = []
        for optimizer in optimizers[i]:
            print('Doing parameters {0} of {1}'.format(i, sum([len(a) for a in optimizers])))
            circuit_results = []
            for i, c in enumerate(gen_circuits):
                print('Evaluating circuit ' + str(i) + ' of ' + str(len(gen_circuits)))
                s_trains = []
                s_tests = []
                c_ws = []
                losses = []
                for n in range(args.iterations):
                    print('Doing crossfold iteration ' + str(n) + ' of ' + str(args.iterations))
                    classifier = get_classifier(c.copy(), optimizer=optimizer, quantum_instance=quantum_instance)
                    score_train, score_test, w, loss = fit_and_score(classifier, X_train, X_test, y_train, y_test)
                    s_trains.append(score_train)
                    s_tests.append(score_test)
                    c_ws.append(w)
                    losses.append(loss)
                circuit_results.append({'scoreTraining':s_trains, 'scoreTesting':s_tests, 'finalWeights':c_ws, 'losses':losses, 'circuit':c})
            optimizer_results.append({'optimizerName':opt_name, 'results':circuit_results, 'optimizerSettings':optimizer.settings})
        print('Saving pickle for optimizer ' + opt_name)
        pickle.dump(optimizer_results, open( "runs/experiments/optimizer_hyperparameters/amsgrad_final.p", "wb" ) )    
    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
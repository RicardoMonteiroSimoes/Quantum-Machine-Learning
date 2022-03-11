from qiskit import *
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

import numpy as np
import math
import random
import csv
import argparse
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib import cm

#### Circuit based functions 
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

def rx_layer(circuit):
    circuit.rx(np.pi/4, range(circuit.width()))
    circuit.barrier()

def create_circuit(n_queries, n_plans, scheme):
    circuit = QuantumCircuit(n_queries*n_plans)
    for module in scheme:
        if module == "h":
            uncertainity_principle(circuit)
        elif module == "c":
            cost_encoding(circuit)
        elif module == "s":
            savings_encoding(circuit)
        elif module == "x":
            rx_layer(circuit)
    return circuit

#### Running circuit
def run_circuit(x, circuit, shots):
    q_sim = Aer.get_backend("aer_simulator")
    results = []
    results_copy = []
    for problem in tqdm(x):
        qc = circuit.bind_parameters(problem.flatten())
        qc.measure_all()
        job = q_sim.run(transpile(qc, q_sim), shots=shots)
        res = job.result()
        results.append(res.get_counts(qc))
        results_copy.append(res.get_counts(qc).copy())
    return results, results_copy

#### Data saving function
def save_run_info(name, n_queries, n_plans, n_problems, n_shots, circuit, percentiles):
    f = open(name + "README.md", "w")
    f.write("# Information for this data run\r\r")
    f.write("Amount of queries: {0}\r\r".format(n_queries))
    f.write("Amount of plans per query: {0}\r\r".format(n_plans))
    f.write("Amount of problems: {0}\r\r".format(n_problems))
    f.write("Amount of shots: {0}\r\r".format(n_shots))
    f.write("<hr>\r\r")
    f.write("## Circuit:\r\r")
    f.write("![Circuit](circuit.png)\r\r")
    f.write("<hr>\r\r")
    f.write("## Percentile results:\r\r")
    f.write('```\r\r')
    for i, p in enumerate(percentiles):
        f.write('{:2.2%} percentile reached a distance of {} to the best solution\r\r'.format(p, i))
    f.write('```\r\r')
    f.write("<hr>\r\r")
    f.write("## Data:\r\r")
    f.write('[Problem Data](problems.csv)\r\r')
    f.write('[Measurements Data](measurements.csv)\r\r')
    f.close()
    circuit.draw("mpl", filename=name+"circuit.png")

def save_problem_data_to_csv(name, problems):
    with open(name + 'problems.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',)
        writer.writerow(['q0p0', 'q0p1', 'q1p0', 'q1p1', 's02', 's03', 's12', 's13'])
        for problem in problems:
            writer.writerow(np.array([problem[1],list(problem[2].values())]).flatten())


def save_measurements_to_csv(name, measurements):
    with open(name + 'measurements.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',)
        writer.writerow(['1010', '1001', '0110', '0101'])
        for measurement in measurements:
            writer.writerow([measurement['0101'], measurement['1001'], measurement['0110'], measurement['1010']])

#### Data based functions
def calculate_distance_percentiles(distances):
    total_count = sum(distances.values())
    percentiles = []
    for i in range(len(distances.keys())):
        print('{:2.2%} percentile reached a distance of {} to the best solution'.format(distances[i]/total_count, i))
        percentiles.append(distances[i]/total_count)
    return percentiles
    

def score_distance_results(results, solutions):
    x = []
    y = []
    for res, sol in zip(results, solutions):
        x.append(list({k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}.keys()))
        y.append(list({k: v for k, v in sorted(sol.items(), key=lambda item: item[1])}.keys()))

    x = np.array(x)
    y = np.array(y)
    distances = {}
    for r,s in zip(x,y):
        distance = np.where(s == r[0])[0][0]
        if not distance in distances:
            distances[distance] = 1
        else:
            distances[distance] += 1
    return distances


def score_results(results, solutions):
    x = []
    for res in results:
        x.append(max(res, key=res.get))
    return accuracy_score(x,solutions)

def parse_results(results):
    parsed_results = []
    for result in tqdm(results):
        remove_useless_keys(result)
        for i, key in enumerate(["0101", "1001", "0110", "1010"]):
            result[i] = result.pop(key, 0)
        parsed_results.append(result)
    return parsed_results

def remove_useless_keys(result):
    for key in ["0000","0001","0010","0100","1000","1101","1011","1100","0011","0111","1110","1111"]:
            if key in result:
                del result[key]

def parse_results_copy(results):
    for result in results:
        remove_useless_keys(result)

def create_solution_set(problems):
    y = []
    y_complete = []
    for problem in tqdm(problems):
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

def scale_problems(problems, scale_min=-np.pi/4, scale_max=np.pi/4):
    scaled_problems = []
    scaler = MinMaxScaler((scale_min, scale_max))
    for problem in tqdm(problems):
        scaled_problems.append(scaler.fit_transform(problem.reshape(-1,1)))
    return scaled_problems

def extract_values(dataset):
    values = []
    for row in tqdm(dataset):
        values.append(
            np.concatenate((row[1].tolist(),list(row[2].values())))
            )
    return np.array(values)

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
    for i in tqdm(range(size)):
        problems.append((n_queries, np.random.randint(cost_min, cost_max, int(n_queries*n_plans)), create_savings(n_queries, n_plans)))
    return problems

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Creates a static circuit to solve MQO problems")
    #parser.add_argument('-q', '--queries', help="Set how many queries", type=int, default=2)
    #parser.add_argument('-p', '--plans', help="Set how many plans per query", type=int, default=2)
    parser.add_argument('-s', '--size', help="Set how many problems to have", type=int, default=150)
    parser.add_argument('-sh', '--shots', help="Shots for the simulation to run", type=int, default=1024)
    parser.add_argument('-r', '--random', help="Random state", type=int)
    parser.add_argument('-n', '--name', help="Experiment name for data collection to files", default="dataset")
    parser.add_argument('-c', '--circuit', help="Design the circuit using chars. h -> uncertainity, c -> cost, s -> savings, b -> blocking savings, x -> rx layer", default="hcsx")
    parser.add_argument('-pc', '--printcircuit', help="Prints the circuit in the console", action="store_true")
    return parser.parse_args(argv)

def main(argv):
    print('Static MQO solver using Quantum Circuits')
    print('---------------------------------------------------')
    global args 
    args = parse_args(argv)
    print('Creating {0} problems, with {1} queries, of which each has {2} plans'.format(args.size, 2, 2))
    problems = generate_problems(2, 2, args.size)
    print('Extracting relevant problem values')
    problems_values = extract_values(problems)
    print('Scaling relevant problem values')
    problems_scaled = scale_problems(problems_values)
    print('Creating solution set, this might take some time')
    solution, complete_solution = create_solution_set(problems_scaled)
    print('Creating parameterized circuit for calculations')
    circuit = create_circuit(2, 2, args.circuit)
    if args.printcircuit:
        print(circuit)
    print('Running circuit')
    results, results_copy = run_circuit(problems_scaled, circuit.copy(), args.shots)
    print('Parsing results')
    results_parsed = parse_results(results)
    print('Comparing results to solution and calculating distances')
    accuracy = score_results(results_parsed, solution)
    distance_to_best = score_distance_results(results_parsed, complete_solution)
    percentiles = calculate_distance_percentiles(distance_to_best)
    print('Saving data')
    parse_results_copy(results_copy)
    save_run_info(args.name, 2, 2, args.size, args.shots, circuit, percentiles)
    save_problem_data_to_csv(args.name, problems)
    save_measurements_to_csv(args.name, results_copy)
    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
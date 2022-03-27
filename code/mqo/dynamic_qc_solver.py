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

def savings_encoding(problem, circuit):
    prev_i = 0
    for i, v in problem[2]:
        if prev_i != i:
            circuit.barrier()
        circuit.crz(Parameter('s'+str(i)+str(v)), i, v)
        prev_i = i
    circuit.barrier()


def rx_layer(circuit, weight):
    circuit.rx(weight, range(circuit.width()))
    circuit.barrier()

def create_circuits(problems, scheme, xweight=np.pi/4):
    circuits = []
    for problem in problems:
        circuit = QuantumCircuit(np.sum(problem[0]))
        for module in scheme:
            if module == "h":
                uncertainity_principle(circuit)
            elif module == "c":
                cost_encoding(circuit)
            elif module == "s":
                savings_encoding(problem, circuit)
            elif module == "x":
                rx_layer(circuit, xweight)
        circuits.append(circuit)
    return circuits

#### Running circuit
def run_circuits(problems, circuits, shots):
    q_sim = Aer.get_backend("aer_simulator")
    results = []
    results_copy = []
    for problem, circuit in tqdm(zip(problems, circuits)):
        qc = circuit.copy().bind_parameters(problem.flatten())
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

def save_problem_data_to_csv(name, problems, ordered_total_costs):
    with open(name + 'problems.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',)
        writer.writerow(['q0p0', 'q0p1', 'q1p0', 'q1p1', 's02', 's03', 's12', 's13', 'cheapest',' 2nd','3rd', 'most expensive'])
        for problem, cost in zip(problems, ordered_total_costs):
            writer.writerow(np.array([problem[1],list(problem[2].values()), cost.flatten()]).flatten())


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
    total_cost = []
    y = []

    for res, sol in zip(results, solutions):
        x.append(list({k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}.keys()))
        total_cost.append(list({k: v for k, v in sorted(sol.items(), key=lambda item: item[1])}.values()))
        y.append(list({k: v for k, v in sorted(sol.items(), key=lambda item: item[1])}.keys()))
    
    total_cost = np.array(total_cost)
    x = np.array(x)
    y = np.array(y)
    distances = {}
    for r,s in zip(x,y):
        distance = np.where(s == r[0])[0][0]
        if not distance in distances:
            distances[distance] = 1
        else:
            distances[distance] += 1
    return distances, total_cost


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
    for key in ["0101", "1001", "0110", "1010"]:
        if not key in result:
            result[key] = 0

def parse_results_copy(results):
    for result in results:
        remove_useless_keys(result)

#TODO change this so it can dynamically do a solution ranking
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

def extract_values(problems):
    values = []
    for row in tqdm(problems):
        values.append(
            np.concatenate((row[1].tolist(),list(row[2].values())))
            )
    return np.array(values)

def create_savings(n_queries, n_plans_per_query, savings_min=-20, savings_max=0):
    savings = {}
    for i in range(n_queries-1):
        for j in range(n_plans_per_query[i]):
            s = j + np.sum(n_plans_per_query[0:i], dtype=int)
            for a in range(i+1, n_queries):
                for b in range(n_plans_per_query[a]):
                    t = b + np.sum(n_plans_per_query[:a], dtype=int)
                    savings[s, t] = random.randint(savings_min, savings_max)
                    print(savings)

    return savings

def generate_problems(n_queries, plan, size, cost_min=0, cost_max=50):
    problems = []
    for i in tqdm(range(size)):
        if not plan:
            plan = np.random.randint(1,4, size=n_queries)
        problems.append((plan, np.random.randint(cost_min, cost_max, np.sum(plan)), 
        create_savings(n_queries, plan)))
    print(problems)
    return problems

def generate_measurement_keys(problems):
    combinational_keys = []
    for problem in tqdm(problems):
        n_qubits = np.sum(problem[0])
        binary_string = []
        for i, v in enumerate(problems[0][0]):
            if i == 0:
                for j in range(v):
                    binary_string.append('0'*j + '1' + '0'*(v-j-1))
            else:
                copy = []
                for x in binary_string:
                    for j in range(v):
                        copy.append(x + '0'*j + '1' + '0'*(v-j-1))
                binary_string = copy
        combinational_keys.append(binary_string)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Creates a static circuit to solve MQO problems")
    #parser.add_argument('-q', '--queries', help="Set how many queries", type=int, default=2)
    #parser.add_argument('-p', '--plans', help="Set how many plans per query", type=int, default=2)
    parser.add_argument('-s', '--size', help="Set how many problems to have", type=int, default=150)
    parser.add_argument('-q', '--queries', help="Set how many queries per problem to have", type=int, default=2)
    parser.add_argument('-qp', '--queryplans',  nargs='+', help="Can define a predetermined query plan that will be same for all problems. If not defined, it is random.", type=int)
    parser.add_argument('-sh', '--shots', help="Shots for the simulation to run", type=int, default=1024)
    parser.add_argument('-r', '--random', help="Random state", type=int)
    parser.add_argument('-n', '--name', help="Experiment name for data collection to files", default="dataset")
    parser.add_argument('-c', '--circuit', help="Design the circuit using chars. h -> uncertainity, c -> cost, s -> savings, b -> blocking savings, x -> rx layer", default="hcsx")
    parser.add_argument('-pc', '--printcircuit', help="Prints the circuit in the console", action="store_true")
    parser.add_argument('-xw', '--xweight', help="Set the weight of the x layer", type=float, default=np.pi/4)
    return parser.parse_args(argv)

def main(argv):
    print('Static MQO solver using Quantum Circuits')
    print('---------------------------------------------------')
    global args 
    args = parse_args(argv)
    print('Creating {0} problems, with {1} queries each'.format(args.size, args.queries))
    if not args.queries == len(args.queryplans):
        raise Exception('Amount of queries has to be equal to length of given query plan!')
    problems = generate_problems(args.queries, args.queryplans, args.size)
    print('Generating combinational measurement keys')
    measurement_keys = generate_measurement_keys(problems)
    print('Extracting relevant problem values')
    problems_values = extract_values(problems)
    print('Scaling relevant problem values')
    problems_scaled = scale_problems(problems_values)
    print('Creating solution set, this might take some time')
    solution, complete_solution = create_solution_set(problems_scaled)
    print('Creating parameterized circuit for calculations')
    circuits = create_circuits(problems, args.circuit, args.xweight)
    if args.printcircuit:
        print(circuits[0])
    print('Running circuit')
    results, results_copy = run_circuits(problems_scaled, circuits, args.shots)
    print('Parsing results')
    results_parsed = parse_results(results)
    print('Comparing results to solution and calculating distances')
    accuracy = score_results(results_parsed, solution)
    distance_to_best, ordered_total_costs = score_distance_results(results_parsed, complete_solution)
    percentiles = calculate_distance_percentiles(distance_to_best)
    #print('Saving data')
    #parse_results_copy(results_copy)
    #save_run_info(args.name, 2, 2, args.size, args.shots, circuit, percentiles)
    #save_problem_data_to_csv(args.name, problems, ordered_total_costs)
    #save_measurements_to_csv(args.name, results_copy)
    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
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
        circuit.crz(Parameter('s'+str(i)+str(v)), i, v)
    circuit.barrier()


def rx_layer(circuit, weight):
    circuit.rx(weight, range(circuit.width()))
    circuit.barrier()

def create_circuit(problem, scheme, xweight=np.pi/4):
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
    return circuit

#### Running circuit
def run_circuits(problems, circuit, shots):
    q_sim = Aer.get_backend("aer_simulator")
    results = []
    results_copy = []
    for problem in tqdm(problems):
        qc = circuit.bind_parameters(problem.flatten())
        qc = qc.reverse_bits()
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
    distances = {}
    x = []
    for res in results:
        x.append(list({k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}.keys()))
    
    for r,s in zip(x,solutions):
        print(r)
        print(r[0])
        print(s)
        distance = np.where(np.array(s) == r[0])[0][0]
        if distance in distances:
            distances[distance] += 1
    print(distances)

    return distances,


def score_results(results, solutions):
    x = []
    y = []

    for i, result in enumerate(results):
        x.append(max(result, key=result.get))
        y.append(solutions[i][0])
    return accuracy_score(x, y)*100

def parse_results(results, solution_keys_sorted):
    parsed_results = []
    for i, result in tqdm(enumerate(results)):
        temp = {}
        for key in solution_keys_sorted[i]:
            temp[key] = result.get(key, 0)
        parsed_results.append(temp)
    return parsed_results

def create_solution_set(problems):
    ranked_solution_keys = [] 
    classical_solution_ranking = []
    for problem in problems:
        savings = collect_savings_for_all_combinations(problem)
        total_cost = append_costs(savings, problem)
        savings_sorted = {k: total_cost[k] for k in sorted(total_cost, key=total_cost.get)}
        classical_solution_ranking.append(savings_sorted)
        ranked_solution_keys.append(generate_solution_keys(savings_sorted, sum(problem[0])))

    return ranked_solution_keys, classical_solution_ranking

def generate_solution_keys(costs, n_qubits):
    solution_keys_ranking = []
    for cost in costs:
        b = list('0'*n_qubits)
        for i in cost:
            b[i] = '1'
        solution_keys_ranking.append(''.join(b))
    return solution_keys_ranking

def append_costs(savings, problem):
    for a in savings:
        for b in a:
            savings[a] += problem[1][b]
    return savings


def collect_savings_for_all_combinations(problem):
    current_combinations = problem[2]
    while len(current_combinations) > np.prod(problem[0]):
        total_savings = {}
        for a in current_combinations:
            saves = current_combinations[a]
            for b in [z for z in problem[2] if z[0] == a[-1]]:
                c = list(a)
                c.append(b[-1])
                c = tuple(c)
                total_savings[c] = saves + sum([problem[2][x, b[-1]] for x in a ])
        current_combinations = total_savings
    return current_combinations

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
    return savings

def generate_problems(n_queries, plan, size, cost_min=0, cost_max=50):
    problems = []
    if not plan:
        plan = np.random.randint(1,4, size=n_queries)
    for i in tqdm(range(size)):
        problems.append((plan, np.random.randint(cost_min, cost_max, np.sum(plan)), 
        create_savings(n_queries, plan)))
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
    print('Creating solution set and keys, this might take some time')
    ranked_solution_keys, classical_solution_ranking = create_solution_set(problems)
    print('Creating parameterized circuit for calculations')
    circuit = create_circuit(problems[0], args.circuit, args.xweight)
    if args.printcircuit:
        print(circuit)
    print('Running circuit')
    results, results_copy = run_circuits(problems_scaled, circuit.copy(), args.shots)
    print('Parsing results')
    results_parsed = parse_results(results, ranked_solution_keys)
    print('Comparing results to solution and calculating distances')
    accuracy = score_results(results_parsed, ranked_solution_keys)
    print('Achieved accuracy of {:2}%'.format(accuracy))
    #distance_to_best, ordered_total_costs = score_distance_results(results_parsed, complete_solution)
    #percentiles = calculate_distance_percentiles(distance_to_best)
    #print('Saving data')
    #parse_results_copy(results_copy)
    #save_run_info(args.name, 2, 2, args.size, args.shots, circuit, percentiles)
    #save_problem_data_to_csv(args.name, problems, classical_solution_ranking)
    #save_measurements_to_csv(args.name, results_copy)
    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
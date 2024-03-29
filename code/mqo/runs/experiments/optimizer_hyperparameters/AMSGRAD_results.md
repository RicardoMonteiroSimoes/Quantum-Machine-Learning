# Results for hyperparameter check for optimizer AMSGRAD

Amount of queries: 2

Amount of plans per query: 2

Amount of problems: 150

Amount of shots: 1024

Amount of runs per parameters and circuit: 13

<hr>

## Circuits:

#### Circuit 0

![Circuit0](AMSGRAD_circuit_0.png)

#### Circuit 1

![Circuit1](AMSGRAD_circuit_1.png)

<hr>

## Result overview:

#### Mean Score Training C0 vs C1: 0.36 - 0.35

![total training accuracy](total_training_accuracy.svg)

#### Mean Score Testing C0 vs C1: 0.39 - 0.38

![total testing accuracy](total_testing_accuracy.svg)

### Sorted for best Testing Accuracy:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|circuit | Testing Accuracy Mean | Training Accuracy Mean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|0 | 0.39 | 0.33 |
|50|1e-06|0.005|0.9|0.85|1e-08|1e-10|True|None|1 | 0.38 | 0.35 |
|50|1e-06|0.001|0.99|0.9|5e-08|1e-10|True|None|0 | 0.37 | 0.31 |
|50|1e-06|0.001|0.9|0.85|1e-08|1e-10|True|None|0 | 0.37 | 0.33 |
|50|1e-06|0.005|0.99|0.9|1e-08|1e-10|True|None|0 | 0.37 | 0.34 |
|50|1e-06|0.005|0.99|0.85|1e-08|1e-10|True|None|1 | 0.36 | 0.34 |
|50|1e-06|0.005|0.9|0.9|1e-08|1e-10|True|None|1 | 0.36 | 0.35 |
|50|1e-06|0.005|0.9|0.85|1e-08|1e-10|True|None|0 | 0.36 | 0.34 |
|50|1e-06|0.001|0.9|0.85|5e-08|1e-10|True|None|0 | 0.36 | 0.35 |
|50|1e-06|0.005|0.9|0.9|1e-08|1e-10|True|None|0 | 0.35 | 0.34 |
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|1 | 0.35 | 0.34 |
|50|1e-06|0.005|0.99|0.85|1e-08|1e-10|True|None|0 | 0.35 | 0.33 |
|50|1e-06|0.001|0.99|0.85|5e-08|1e-10|True|None|0 | 0.35 | 0.34 |
|50|1e-06|0.005|0.99|0.9|5e-08|1e-10|True|None|1 | 0.35 | 0.33 |
|50|1e-06|0.005|0.99|0.85|5e-08|1e-10|True|None|1 | 0.35 | 0.35 |
|50|1e-06|0.005|0.99|0.9|1e-08|1e-10|True|None|1 | 0.34 | 0.35 |
|50|1e-06|0.005|0.99|0.85|5e-08|1e-10|True|None|0 | 0.34 | 0.34 |
|50|1e-06|0.001|0.9|0.9|5e-08|1e-10|True|None|1 | 0.34 | 0.32 |
|50|1e-06|0.001|0.9|0.9|1e-08|1e-10|True|None|1 | 0.34 | 0.33 |
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|0 | 0.34 | 0.36 |
|50|1e-06|0.001|0.99|0.85|5e-08|1e-10|True|None|1 | 0.33 | 0.32 |
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|1 | 0.32 | 0.34 |
|50|1e-06|0.001|0.99|0.85|1e-08|1e-10|True|None|0 | 0.32 | 0.35 |
|50|1e-06|0.001|0.99|0.9|5e-08|1e-10|True|None|1 | 0.32 | 0.34 |
|50|1e-06|0.001|0.9|0.85|1e-08|1e-10|True|None|1 | 0.32 | 0.33 |
|50|1e-06|0.005|0.99|0.9|5e-08|1e-10|True|None|0 | 0.32 | 0.33 |
|50|1e-06|0.001|0.9|0.85|5e-08|1e-10|True|None|1 | 0.32 | 0.35 |
|50|1e-06|0.001|0.9|0.9|5e-08|1e-10|True|None|0 | 0.32 | 0.32 |
|50|1e-06|0.001|0.99|0.9|1e-08|1e-10|True|None|0 | 0.31 | 0.35 |
|50|1e-06|0.001|0.9|0.9|1e-08|1e-10|True|None|0 | 0.3 | 0.32 |
|50|1e-06|0.001|0.99|0.9|1e-08|1e-10|True|None|1 | 0.3 | 0.35 |
|50|1e-06|0.001|0.99|0.85|1e-08|1e-10|True|None|1 | 0.3 | 0.34 |
### Sorted for best Training Accuracy:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|circuit | Testing Accuracy Mean | Training Accuracy Mean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|0 | 0.34 | 0.36 |
|50|1e-06|0.005|0.9|0.9|1e-08|1e-10|True|None|1 | 0.36 | 0.35 |
|50|1e-06|0.001|0.99|0.9|1e-08|1e-10|True|None|0 | 0.31 | 0.35 |
|50|1e-06|0.001|0.9|0.85|5e-08|1e-10|True|None|1 | 0.32 | 0.35 |
|50|1e-06|0.005|0.99|0.85|5e-08|1e-10|True|None|1 | 0.35 | 0.35 |
|50|1e-06|0.001|0.99|0.9|1e-08|1e-10|True|None|1 | 0.3 | 0.35 |
|50|1e-06|0.001|0.9|0.85|5e-08|1e-10|True|None|0 | 0.36 | 0.35 |
|50|1e-06|0.005|0.99|0.9|1e-08|1e-10|True|None|1 | 0.34 | 0.35 |
|50|1e-06|0.001|0.99|0.85|1e-08|1e-10|True|None|0 | 0.32 | 0.35 |
|50|1e-06|0.005|0.9|0.85|1e-08|1e-10|True|None|1 | 0.38 | 0.35 |
|50|1e-06|0.005|0.99|0.9|1e-08|1e-10|True|None|0 | 0.37 | 0.34 |
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|1 | 0.32 | 0.34 |
|50|1e-06|0.005|0.99|0.85|1e-08|1e-10|True|None|1 | 0.36 | 0.34 |
|50|1e-06|0.001|0.99|0.85|5e-08|1e-10|True|None|0 | 0.35 | 0.34 |
|50|1e-06|0.005|0.9|0.85|1e-08|1e-10|True|None|0 | 0.36 | 0.34 |
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|1 | 0.35 | 0.34 |
|50|1e-06|0.005|0.99|0.85|5e-08|1e-10|True|None|0 | 0.34 | 0.34 |
|50|1e-06|0.005|0.9|0.9|1e-08|1e-10|True|None|0 | 0.35 | 0.34 |
|50|1e-06|0.001|0.99|0.9|5e-08|1e-10|True|None|1 | 0.32 | 0.34 |
|50|1e-06|0.001|0.99|0.85|1e-08|1e-10|True|None|1 | 0.3 | 0.34 |
|50|1e-06|0.001|0.9|0.85|1e-08|1e-10|True|None|0 | 0.37 | 0.33 |
|50|1e-06|0.005|0.99|0.9|5e-08|1e-10|True|None|1 | 0.35 | 0.33 |
|50|1e-06|0.001|0.9|0.85|1e-08|1e-10|True|None|1 | 0.32 | 0.33 |
|50|1e-06|0.005|0.99|0.9|5e-08|1e-10|True|None|0 | 0.32 | 0.33 |
|50|1e-06|0.005|0.99|0.85|1e-08|1e-10|True|None|0 | 0.35 | 0.33 |
|50|1e-06|0.001|0.9|0.9|1e-08|1e-10|True|None|1 | 0.34 | 0.33 |
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|0 | 0.39 | 0.33 |
|50|1e-06|0.001|0.9|0.9|5e-08|1e-10|True|None|1 | 0.34 | 0.32 |
|50|1e-06|0.001|0.99|0.85|5e-08|1e-10|True|None|1 | 0.33 | 0.32 |
|50|1e-06|0.001|0.9|0.9|5e-08|1e-10|True|None|0 | 0.32 | 0.32 |
|50|1e-06|0.001|0.9|0.9|1e-08|1e-10|True|None|0 | 0.3 | 0.32 |
|50|1e-06|0.001|0.99|0.9|5e-08|1e-10|True|None|0 | 0.37 | 0.31 |
### Comparison best Training and Testing:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|circuit | Testing Accuracy Mean | Training Accuracy Mean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|0 | 0.34 | 0.36 |
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|0 | 0.39 | 0.33 |
<hr>

## Run 1:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.99|0.9|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_0_0_training_boxplot.svg)

Testing Accuracy Average: 0.31%

![Boxplot0](AMSGRAD_experiment_0_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_0_1_training_boxplot.svg)

Testing Accuracy Average: 0.3%

![Boxplot1](AMSGRAD_experiment_0_1_testing_boxplot.svg)

<hr>

## Run 2:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.99|0.9|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.31%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_1_0_training_boxplot.svg)

Testing Accuracy Average: 0.37%

![Boxplot0](AMSGRAD_experiment_1_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_1_1_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot1](AMSGRAD_experiment_1_1_testing_boxplot.svg)

<hr>

## Run 3:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.99|0.85|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_2_0_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot0](AMSGRAD_experiment_2_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_2_1_training_boxplot.svg)

Testing Accuracy Average: 0.3%

![Boxplot1](AMSGRAD_experiment_2_1_testing_boxplot.svg)

<hr>

## Run 4:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.99|0.85|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_3_0_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot0](AMSGRAD_experiment_3_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.32%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_3_1_training_boxplot.svg)

Testing Accuracy Average: 0.33%

![Boxplot1](AMSGRAD_experiment_3_1_testing_boxplot.svg)

<hr>

## Run 5:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.9|0.9|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.32%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_4_0_training_boxplot.svg)

Testing Accuracy Average: 0.3%

![Boxplot0](AMSGRAD_experiment_4_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_4_1_training_boxplot.svg)

Testing Accuracy Average: 0.34%

![Boxplot1](AMSGRAD_experiment_4_1_testing_boxplot.svg)

<hr>

## Run 6:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.9|0.9|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.32%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_5_0_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot0](AMSGRAD_experiment_5_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.32%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_5_1_training_boxplot.svg)

Testing Accuracy Average: 0.34%

![Boxplot1](AMSGRAD_experiment_5_1_testing_boxplot.svg)

<hr>

## Run 7:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.9|0.85|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_6_0_training_boxplot.svg)

Testing Accuracy Average: 0.37%

![Boxplot0](AMSGRAD_experiment_6_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_6_1_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot1](AMSGRAD_experiment_6_1_testing_boxplot.svg)

<hr>

## Run 8:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.001|0.9|0.85|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_7_0_training_boxplot.svg)

Testing Accuracy Average: 0.36%

![Boxplot0](AMSGRAD_experiment_7_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_7_1_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot1](AMSGRAD_experiment_7_1_testing_boxplot.svg)

<hr>

## Run 9:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.99|0.9|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_8_0_training_boxplot.svg)

Testing Accuracy Average: 0.37%

![Boxplot0](AMSGRAD_experiment_8_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_8_1_training_boxplot.svg)

Testing Accuracy Average: 0.34%

![Boxplot1](AMSGRAD_experiment_8_1_testing_boxplot.svg)

<hr>

## Run 10:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.99|0.9|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_9_0_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot0](AMSGRAD_experiment_9_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_9_1_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot1](AMSGRAD_experiment_9_1_testing_boxplot.svg)

<hr>

## Run 11:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.99|0.85|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_10_0_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot0](AMSGRAD_experiment_10_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_10_1_training_boxplot.svg)

Testing Accuracy Average: 0.36%

![Boxplot1](AMSGRAD_experiment_10_1_testing_boxplot.svg)

<hr>

## Run 12:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.99|0.85|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_11_0_training_boxplot.svg)

Testing Accuracy Average: 0.34%

![Boxplot0](AMSGRAD_experiment_11_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_11_1_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot1](AMSGRAD_experiment_11_1_testing_boxplot.svg)

<hr>

## Run 13:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.9|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_12_0_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot0](AMSGRAD_experiment_12_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_12_1_training_boxplot.svg)

Testing Accuracy Average: 0.36%

![Boxplot1](AMSGRAD_experiment_12_1_testing_boxplot.svg)

<hr>

## Run 14:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.9|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.36%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_13_0_training_boxplot.svg)

Testing Accuracy Average: 0.34%

![Boxplot0](AMSGRAD_experiment_13_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_13_1_training_boxplot.svg)

Testing Accuracy Average: 0.32%

![Boxplot1](AMSGRAD_experiment_13_1_testing_boxplot.svg)

<hr>

## Run 15:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.85|1e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_14_0_training_boxplot.svg)

Testing Accuracy Average: 0.36%

![Boxplot0](AMSGRAD_experiment_14_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.35%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_14_1_training_boxplot.svg)

Testing Accuracy Average: 0.38%

![Boxplot1](AMSGRAD_experiment_14_1_testing_boxplot.svg)

<hr>

## Run 16:

#### Optimizer settings:

|maxiter|tol|lr|beta_1|beta_2|noise_factor|eps|amsgrad|snapshot_dir|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|1e-06|0.005|0.9|0.85|5e-08|1e-10|True|None|

### Average accuracy per circuit:

### Circuit 0:

Training Accuracy Average: 0.33%

#### Boxplot of results:

![Boxplot0](AMSGRAD_experiment_15_0_training_boxplot.svg)

Testing Accuracy Average: 0.39%

![Boxplot0](AMSGRAD_experiment_15_0_testing_boxplot.svg)

### Circuit 1:

Training Accuracy Average: 0.34%

#### Boxplot of results:

![Boxplot1](AMSGRAD_experiment_15_1_training_boxplot.svg)

Testing Accuracy Average: 0.35%

![Boxplot1](AMSGRAD_experiment_15_1_testing_boxplot.svg)

<hr>


import argparse
from random import random
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles, make_circles, make_moons, make_multilabel_classification
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import csv
import numpy as np


def save_to_file(filename, features, labels):
    data = [np.append(features[i], labels[i]) for i in range(len(labels))]
    header = np.append(['F' + str(i) for i in range(len(features[0]))], "Class")
    #data.insert(0, header)
    with open(filename + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)
        writer.writerow(header)
        for f,l in zip(features, labels):
            writer.writerow(np.append([i for i in f], l))
    print('Finished writing csv')
    create_dataset_plot(filename)

        

def create_dataset_plot(filename):
    print('Creating parallel plot')
    print('Reading csv')
    df = pd.read_csv(filename+'.csv')
    ax = parallel_coordinates(df, 'Class', colormap=plt.get_cmap("Set2"))
    fig = ax.get_figure()
    fig.savefig(filename+'.svg', format="svg", transparent=True)
    print('Parallel plot was saved')

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Create a random dataset using sklearn, saves as .csv and generates a parallel plot for it. Check scikit doc for more info: https://scikit-learn.org/stable/datasets/sample_generators.html")
    parser.add_argument('-f', '--features', help="Set the dimension of the features", type=int, default=2)
    parser.add_argument('-c', '--classes', help="Sets how many classes you want to have", type=int, default=2)
    parser.add_argument('-s', '--size', help="How many entries you want to have", type=int, default=50)
    parser.add_argument('-r', '--random', help="Random state", type=int)
    parser.add_argument('-m', '--mode', help="Sets the generator mode. Options are 'moons', 'blobs', 'circles','classification','multilabel','gaussian'.\
        'classification' uses additional parameters -ninf, -nred, -nrep, -ncpc", default='blobs')
    parser.add_argument('-ninf', '--ninformative', help="How many features shall be informative", type=int)
    parser.add_argument('-nred', '--nredundant', help="How many features shall be redundant", type=int)
    parser.add_argument('-nrep', '--nrepeated', help="How many features shall be duplicated", type=int)
    parser.add_argument('-ncpc', '--nclusterperclass', help="How many clusters each class shall have", type=int)
    parser.add_argument('-n', '--name', help="Name the file", default="dataset")
    parser.add_argument('-p', '--plot', help="If you have data already, use this function along with -n to read a <file>.csv. \
        It has to be formatted so that the first row is the header, and the column with the classes is annoted with 'Class'", default=False)
    return parser.parse_args(argv)

def blobs(n_features, n_classes, size, random_state):
    if random_state:
        return make_blobs(n_samples=size, centers=n_classes, n_features=n_features, random_state=random_state)
    else:
        return make_blobs(n_samples=size, centers=n_classes, n_features=n_features)

def gaussian(n_features, n_classes, size, random_state):
    if random_state:
        return make_gaussian_quantiles(n_samples=size, n_features=n_features, n_classes=n_classes, random_state=random_state)
    else:
        return make_gaussian_quantiles(n_samples=size, n_features=n_features, n_classes=n_classes)

def multilabel(n_features, n_classes, size, random_state):
    if random_state:
        return make_multilabel_classification(n_samples=size, n_features=n_features, n_classes=n_classes, allow_unlabeled=False, random_state=random_state)
    else:
        return make_multilabel_classification(n_samples=size, n_features=n_features, allow_unlabeled=False, n_classes=n_classes)


def circles(size, random_state):
    if random_state:
        return make_circles(n_samples=size, random_state=random_state)
    else:
        return make_circles(n_samples=size)

def moons(size, random_state):
    if random_state:
        return make_moons(n_samples=size, random_state=random_state)
    else:
        return make_moons(n_samples=size)

def classification(n_features, n_classes, n_inf, n_red, n_rep, n_cpc, size, random_state):
    if random_state:
        return make_classification(n_samples=size, n_features=n_features, n_classes=n_classes, n_informative=n_inf, n_redundant=n_red, 
            n_repeated=n_rep, n_clusters_per_class=n_cpc, random_state=random_state)
    else:
        return make_classification(n_samples=size, n_features=n_features, n_classes=n_classes, n_informative=n_inf, n_redundant=n_red, 
            n_repeated=n_rep, n_clusters_per_class=n_cpc)


def main(argv):

    print('Artificial dataset generator')
    print('---------------------------------------------------')
    global args 
    args = parse_args(argv)
    print('Generating dataset')
    if args.mode == "blobs":
        x, y = blobs(args.features, args.classes, args.size, args.random)
    elif args.mode == "gaussian":
        x, y = gaussian(args.features, args.classes, args.size, args.random)
    elif args.mode == "multilabel":
        x, y = multilabel(args.features, args.classes, args.size, args.random)
    elif args.mode == "circles":
        x, y = circles(args.size, args.random)
    elif args.mode == "moons":
        x, y = moons(args.size, args.random)
    elif args.mode == "classification":
        if not args.ninformative and not args.nredundant and not args.nrepeated and not args.nclusterperclass:
            print("You are using mode 'classification' without the necessary arguments!")
            return
        x, y = classification(args.features, args.classes, args.ninformative, args.nredundant, 
            args.nrepeated, args.nclusterperclass, args.size, args.random)
    else:
        print('Unsupport mode selected. Aborting')
        return

    print('Saving dataset')
    save_to_file(args.name, x, y)
    print('Finished execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))
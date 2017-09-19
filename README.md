# Multigraph

This repository contains materials for artifact evaluation of the paper "MultiGraph: Efficient Graph Processing on GPUs".  It contains:

1) Source code for MultiGraph, 
2) Scripts for installation of MultiGraph and for download/installation of four other compared graph processing systems: CuSha, Groute, Gunrock, and WS.
3) The entire data set for evaluating different frameworks. The full set of test graphs takes several hours to run, hence an abbreviated version is provided in the repository: https://github.com/hochawa/multigraph2 


Installation
==============

Running  './install.sh' without any arguments puts out the following help message, specifying the different installation options.

        MultiGraph: 
			./install.sh multigraph
        Cusha: 
			./install.sh cusha
        WS: 
			./install.sh ws
        Groute: 
			./install.sh groute
        Gunrock: 
			./install.sh gunrock

Software requirements:
----------------------

	1. cmake >= 3.2
	2. boost >= 1.56
	3. gcc >= 4.9  && < 5.0
	4. nvcc 7.5

Hardware requirements:
----------------------

	1. Nvidia Kepler GPU
	2. CC 3.5
	3. Global memory >= 12 GB

Datasets
========

Running  './download_datasets.sh' will download the required datasets.

Benchmarking
============

Running './benchmark.sh <framework>' will run all datasets on the specified framework


How to select datasets 
----------------------

The file 'datasets.list' contains the name of the datasets to be
benchmarked. The lines beginning with "#" are treated as comments and can be
used to disable the dataset. Each line in "datasets.list" contains three
entries: "dataset name", "source vertex", "#iteration"

How to test a new dataset
=========================

1. Add the name of the dataset, source vertex and iteration in the
file "datasets.lists"

2. Each framework/benchmark pair has a specific configuration file. For example, gunrock.bc contains configuration information for each dataset, for executing the gunrock framework with benchmark application BC, and groute.bfs  contains configuration information for executing groute with the BFS application. Add the configuration of the new dataset to each framework/benchmark. 

3. If the new dataset's name is ABCD, the files ABCD.mtx, ABCD.gr, ABCD.cusha, etc.  should be placed in directory multigraph/ABCD. 

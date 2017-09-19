# Multigraph Version 0.11

This repository contains materials for the paper "MultiGraph: Efficient Graph Processing on GPUs".  


Installation
==============

Running  './install.sh' without any arguments.


Software requirements:
----------------------

	1. cmake >= 3.2
	3. gcc >= 4.9  && < 5.0
	4. nvcc 7.5

Hardware requirements:
----------------------

	1. Nvidia Kepler GPU
	2. CC 3.5

Datasets
========

MTX files (Matrix Market Format) are compartible with the program.

Benchmarking
============

BFS : ./BFS file-name.mtx source-vertex num-of-iteration (e.g., ./BFS soc-LiveJournal1.mtx 0 1)

SSSP, BC, CC, DATA-DRIVEN-PR, TOPOLOGY-DRIVEN-PR are the same. 


To do
=====

1. improve performance for small dataset 

2. tune for Pascal machine

3. improve overall performance

4. processing non-symmetric graphs (currently, all datasets are symmetrized)

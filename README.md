# Constant-depth magic state cultivation with Clifford measurements by gauging

Repo accompanying [https://arxiv.org/abs/2510.05225](https://arxiv.org/abs/2603.05429)

Code:
- gauging_clifford_measurement.ipynb: main notebook that presents the framework and generates the figures
- color_code.py: class that builds the circuit for the gauging clifford measurement
- qiskit_utils.py: conversion of the stim circuit to qiskit and the statevector simulator setup
- escape_stage.py: class that builds the deformation circuit to surface code
- utils.py: functions for plotting and adding noise to the stim circuits

Data:
- cultivation_stim_circuits: stim circuit files use for benhmarking magic state cultivation (Gidney et al.)
- yield_comparison: data for the comparison between cultivation and the gauging measurement

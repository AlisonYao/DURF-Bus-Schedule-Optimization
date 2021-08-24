# DURF-Bus-Schedule-Optimization

Building on our [final project](https://github.com/AlisonYao/Optimization-Final-Project) for the course **Intro to Optimization and Mathematical Programming**, we extended our problem settings to be closer to real-life scenarios. We used Genetic Algorithm to further optimized the NYU Shanghai shuttle bus schedules.

Special thanks to Professor Zhibin Chen, the supervisor of this research project, for offering us guidance along the way and Dean's Undergraduate Research Fund (DURF) for generously providing us research funding.

## DURF Report

The research outcome is presented in the form of a report. Please see `DURF_report.docx`.

## Codes for Genetic Algorithm

Baseline problem solution: please see `baseline_solution.py` code

Extended problem solution: please see `extension_solution.py` code

The file Genetic Algorithm has all of my codes in progress.

- `toy_GA_example.py` contains the YouTube tutorials I followed to learn GA.
- `baseline.py` is the first attempt at solving the **baseline problem**. I enforced the demand constraint as a requirement, rather than a penalty. This version is significantly slower to run because its one iteration could take the time of 50 iterations, only because the first 49 times failed to meet the demand constraint. Although the solutions are guaranteed to met the demand, it performs poorly.
- `baseline_v2.py` is the successful second attempt at the **baseline problem** where I converted the demand constraint into penalty. Although meeting the demand is not guaranteed at first, the solution will eventually satify demand over time. Same with the rush hour constraint and the max working hour constraint. `baseline_solution.py` is the duplicate of this file.
- `extension.py` is a failed first attempt at the **extended problem**.
- `extension_v2.py` is a successful solution to the **extended problem**. `extended_solution.py` is the duplicate of this file.
- `hpc_shuttle_v2.py` just outputs results into txt files.
- `temp.py` is just a file for testing that doesn't have anything important.

<!-- ## Blogs -->

# DURF-Bus-Schedule-Optimization

Building on our [final project](https://github.com/AlisonYao/Optimization-Final-Project) for the course Intro to Optimization and Mathematical Programming, we further extended our problem settings to be closer to real-life scenarios. We used Genetic Algorithm to further optimized the NYU Shanghai shuttle bus schedules.

Special thanks to Professor Zhibin Chen, the supervisor of this research project, for offering us guidance along the way and Dean's Undergraduate Research Fund (DURF) for generously providing us research funding.

## Codes for Genetic Algorithm

Base problem solution: please see `base_problem.py` code

Extended problem solution: please see `extended_problem.py` code

The file Genetic Algorithm has all of my codes in progress.

- `toy_genome.py` contains the YouTube tutorials I followed to learn GA.
- `shuttle.py` is the first attempt at solving the **base problem**. I enforced the demand constraint as a requirement, rather than a penalty. This version is significantly slower to run because its one iteration could take the time of 50 iterations, only because the first 49 times failed to meet the demand constraint. Although the solutions are guaranteed to met the demand, it performs poorly.
- `shuttle_v2.py` is the successful second attempt at the **base problem** where I converted the demand constraint into penalty. Although meeting the demand is not guaranteed at first, the solution will eventually satify demand over time. Same with the rush hour constraint and the max working hour constraint. `base_problem.py` is the duplicate of this file.
- `shuttle_multi_node.py` is a failed first attempt at the **extended problem**.
- `shuttle_multi_node_v2.py` is a successful solution to the **extended problem**. `extended_problem.py` is the duplicate of this file.
- `temp.py` is just my test file that doesn't have anything important.

## Paper

The research outcome is presented in the form of an academic paper. Please see `report.docx`.

<!-- ## Blogs -->

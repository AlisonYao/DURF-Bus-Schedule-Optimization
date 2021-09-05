######################################################################
# zhuang@x86_64-apple-darwin13 webapp % FLASK_APP=init.py flask run
######################################################################

from flask import Flask, render_template, request, url_for, redirect
import random
import numpy as np
import time
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/baseline", methods=["GET", "POST"])
def baseline():
    # parameters
    initial_prob = float(request.form["initial_prob"])
    population_size = int(request.form["population_size"])
    elitism_cutoff = 2
    mutationType = "New"  # Conv
    mutation_prob = 0.95
    mutation_num = 1 if mutationType == "Conv" else 1
    evolution_depth = int(request.form["evolution_depth"])
    N = int(request.form["N"])
    D = int(request.form["D"])
    tolerance = int(request.form["tolerance"])
    intervalDuration = 0.5

    # Jinqiao to AB
    JQ2AB_demand1 = request.form["JQ2AB_demand1"]
    JQ2AB_demand2 = request.form["JQ2AB_demand2"]
    JQ2AB_demand3 = request.form["JQ2AB_demand3"]
    JQ2AB_demand4 = request.form["JQ2AB_demand4"]
    JQ2AB_demand5 = request.form["JQ2AB_demand5"]
    JQ2AB_demand6 = request.form["JQ2AB_demand6"]
    JQ2AB_demand7 = request.form["JQ2AB_demand7"]
    JQ2AB_demand8 = request.form["JQ2AB_demand8"]
    JQ2AB_demand9 = request.form["JQ2AB_demand9"]
    JQ2AB_demand10 = request.form["JQ2AB_demand10"]
    JQ2AB_demand11 = request.form["JQ2AB_demand11"]
    JQ2AB_demand12 = request.form["JQ2AB_demand12"]
    JQ2AB_demand13 = request.form["JQ2AB_demand13"]
    JQ2AB_demand14 = request.form["JQ2AB_demand14"]
    JQ2AB_demand15 = request.form["JQ2AB_demand15"]
    JQ2AB_demand16 = request.form["JQ2AB_demand16"]
    JQ2AB_demand17 = request.form["JQ2AB_demand17"]
    JQ2AB_demand18 = request.form["JQ2AB_demand18"]
    JQ2AB_demand19 = request.form["JQ2AB_demand19"]
    JQ2AB_demand20 = request.form["JQ2AB_demand20"]
    JQ2AB_demand21 = request.form["JQ2AB_demand21"]
    JQ2AB_demand22 = request.form["JQ2AB_demand22"]
    JQ2AB_demand23 = request.form["JQ2AB_demand23"]
    JQ2AB_demand24 = request.form["JQ2AB_demand24"]
    JQ2AB_demand25 = request.form["JQ2AB_demand25"]
    JQ2AB_demand26 = request.form["JQ2AB_demand26"]
    JQ2AB_demand27 = request.form["JQ2AB_demand27"]
    JQ2AB_demand28 = request.form["JQ2AB_demand28"]
    JQ2AB_demand29 = request.form["JQ2AB_demand29"]
    JQ2AB_demand30 = request.form["JQ2AB_demand30"]
    JQ2AB_demand31 = request.form["JQ2AB_demand31"]
    JQ2AB_demand32 = request.form["JQ2AB_demand32"]
    JQ2AB_demand33 = request.form["JQ2AB_demand33"]
    JQ2AB_demand34 = request.form["JQ2AB_demand34"]
    # AB to Jinqiao
    AB2JQ_demand1 = request.form["AB2JQ_demand1"]
    AB2JQ_demand2 = request.form["AB2JQ_demand2"]
    AB2JQ_demand3 = request.form["AB2JQ_demand3"]
    AB2JQ_demand4 = request.form["AB2JQ_demand4"]
    AB2JQ_demand5 = request.form["AB2JQ_demand5"]
    AB2JQ_demand6 = request.form["AB2JQ_demand6"]
    AB2JQ_demand7 = request.form["AB2JQ_demand7"]
    AB2JQ_demand8 = request.form["AB2JQ_demand8"]
    AB2JQ_demand9 = request.form["AB2JQ_demand9"]
    AB2JQ_demand10 = request.form["AB2JQ_demand10"]
    AB2JQ_demand11 = request.form["AB2JQ_demand11"]
    AB2JQ_demand12 = request.form["AB2JQ_demand12"]
    AB2JQ_demand13 = request.form["AB2JQ_demand13"]
    AB2JQ_demand14 = request.form["AB2JQ_demand14"]
    AB2JQ_demand15 = request.form["AB2JQ_demand15"]
    AB2JQ_demand16 = request.form["AB2JQ_demand16"]
    AB2JQ_demand17 = request.form["AB2JQ_demand17"]
    AB2JQ_demand18 = request.form["AB2JQ_demand18"]
    AB2JQ_demand19 = request.form["AB2JQ_demand19"]
    AB2JQ_demand20 = request.form["AB2JQ_demand20"]
    AB2JQ_demand21 = request.form["AB2JQ_demand21"]
    AB2JQ_demand22 = request.form["AB2JQ_demand22"]
    AB2JQ_demand23 = request.form["AB2JQ_demand23"]
    AB2JQ_demand24 = request.form["AB2JQ_demand24"]
    AB2JQ_demand25 = request.form["AB2JQ_demand25"]
    AB2JQ_demand26 = request.form["AB2JQ_demand26"]
    AB2JQ_demand27 = request.form["AB2JQ_demand27"]
    AB2JQ_demand28 = request.form["AB2JQ_demand28"]
    AB2JQ_demand29 = request.form["AB2JQ_demand29"]
    AB2JQ_demand30 = request.form["AB2JQ_demand30"]
    AB2JQ_demand31 = request.form["AB2JQ_demand31"]
    AB2JQ_demand32 = request.form["AB2JQ_demand32"]
    AB2JQ_demand33 = request.form["AB2JQ_demand33"]
    AB2JQ_demand34 = request.form["AB2JQ_demand34"]
    demand = np.array(
        [
            [
                int(JQ2AB_demand1),
                int(JQ2AB_demand2),
                int(JQ2AB_demand3),
                int(JQ2AB_demand4),
                int(JQ2AB_demand5),
                int(JQ2AB_demand6),
                int(JQ2AB_demand7),
                int(JQ2AB_demand8),
                int(JQ2AB_demand9),
                int(JQ2AB_demand10),
                int(JQ2AB_demand11),
                int(JQ2AB_demand12),
                int(JQ2AB_demand13),
                int(JQ2AB_demand14),
                int(JQ2AB_demand15),
                int(JQ2AB_demand16),
                int(JQ2AB_demand17),
                int(JQ2AB_demand18),
                int(JQ2AB_demand19),
                int(JQ2AB_demand20),
                int(JQ2AB_demand21),
                int(JQ2AB_demand22),
                int(JQ2AB_demand23),
                int(JQ2AB_demand24),
                int(JQ2AB_demand25),
                int(JQ2AB_demand26),
                int(JQ2AB_demand27),
                int(JQ2AB_demand28),
                int(JQ2AB_demand29),
                int(JQ2AB_demand30),
                int(JQ2AB_demand31),
                int(JQ2AB_demand32),
                int(JQ2AB_demand33),
                int(JQ2AB_demand34),
            ],
            [
                int(AB2JQ_demand1),
                int(AB2JQ_demand2),
                int(AB2JQ_demand3),
                int(AB2JQ_demand4),
                int(AB2JQ_demand5),
                int(AB2JQ_demand6),
                int(AB2JQ_demand7),
                int(AB2JQ_demand8),
                int(AB2JQ_demand9),
                int(AB2JQ_demand10),
                int(AB2JQ_demand11),
                int(AB2JQ_demand12),
                int(AB2JQ_demand13),
                int(AB2JQ_demand14),
                int(AB2JQ_demand15),
                int(AB2JQ_demand16),
                int(AB2JQ_demand17),
                int(AB2JQ_demand18),
                int(AB2JQ_demand19),
                int(AB2JQ_demand20),
                int(AB2JQ_demand21),
                int(AB2JQ_demand22),
                int(AB2JQ_demand23),
                int(AB2JQ_demand24),
                int(AB2JQ_demand25),
                int(AB2JQ_demand26),
                int(AB2JQ_demand27),
                int(AB2JQ_demand28),
                int(AB2JQ_demand29),
                int(AB2JQ_demand30),
                int(AB2JQ_demand31),
                int(AB2JQ_demand32),
                int(AB2JQ_demand33),
                int(AB2JQ_demand34),
            ],
        ]
    )

    intervalNum = demand.shape[-1]
    maxWorkingHour = 4

    checkDemandFlag = True
    checkRushHourFlag = request.form["checkRushHourFlag"] == "True"
    checkMaxWorkingHourFlag = True
    alpha = float(request.form["alpha"])
    demandViolationPenalty = int(request.form["demandViolationPenalty"])
    rushHourViolationPenalty = int(request.form["rushHourViolationPenalty"])
    maxWorkingHourViolationPenalty = int(request.form["maxWorkingHourViolationPenalty"])

    def generate_random_N_paths(N, path_length):
        """
        Randomize N paths (1 path is like 010101010101) to generate one solution
        """
        one_solution = []
        for _ in range(N):
            # set the weights to initialize feasible solution faster
            one_path = random.choices(
                population=[0, 1],
                weights=[1 - initial_prob, initial_prob],
                k=path_length,
            )
            one_solution.append(one_path)
        return np.array(one_solution)

    def decode_one_path(one_path):
        decoded = []
        i, previous_node = None, None
        for j, current_node in enumerate(one_path):
            # first node
            if i == previous_node == None:
                if current_node == 0:
                    decoded.append([1, 0, 0, 0])
                else:
                    decoded.append([0, 1, 0, 0])
            # all nodes after first node
            else:
                previous_path = decoded[i]
                assert sum(previous_path) == 1
                if previous_path[0] == 1:  # A
                    if current_node == 0:  # A
                        decoded.append([1, 0, 0, 0])
                    else:  # B
                        decoded.append([0, 1, 0, 0])
                elif previous_path[1] == 1:  # B
                    if current_node == 0:  # D
                        decoded.append([0, 0, 0, 1])
                    else:  # C
                        decoded.append([0, 0, 1, 0])
                elif previous_path[2] == 1:  # C
                    if current_node == 0:  # A
                        decoded.append([1, 0, 0, 0])
                    else:  # B
                        decoded.append([0, 1, 0, 0])
                else:
                    if current_node == 0:  # D
                        decoded.append([0, 0, 0, 1])
                    else:  # C
                        decoded.append([0, 0, 1, 0])
            i, previous_node = j, current_node
        return np.array(decoded).T

    def demand_constraint(solution_chromosome, tolerance):
        """
        make sure the demand is met
        """
        # get the link representation first
        directional_N_paths = [
            decode_one_path(one_path) for one_path in solution_chromosome
        ]
        link = sum(directional_N_paths)
        supplyDemandDifference = np.greater_equal(demand - tolerance, link[1:3, :] * D)
        mask = (demand - tolerance) - (link[1:3, :] * D)
        missedDemandNum = np.sum(supplyDemandDifference * mask)
        return int(missedDemandNum) == 0, int(missedDemandNum)

    def rush_hour_constraint(solution_chromosome):
        """
        during rush hours, one interval is not enough time to commute
        """
        violationCount = 0
        for one_path in solution_chromosome:
            # morning rush hour
            if one_path[1] + one_path[2] == 2:
                violationCount += 1
            # evening rush hour
            if one_path[21] + one_path[22] == 2:
                violationCount += 1
        return int(violationCount) == 0, int(violationCount)

    def max_working_hour_constraint(solution_chromosome):
        """
        make sure that no driver works more than a few hours continuously
        """
        violationCount = 0
        for one_path in solution_chromosome:
            num, num_list = 0, []
            one_path_copy = one_path.copy()
            # first check if rush hour 10 or 01 actually is 11
            if checkRushHourFlag:
                if one_path_copy[1] == 1 and one_path_copy[2] == 0:
                    one_path_copy[2] = 1
                if one_path_copy[21] == 1 and one_path_copy[22] == 0:
                    one_path_copy[22] = 1
            for i, node in enumerate(one_path_copy):
                num += node
                if i + 1 == len(one_path_copy):
                    num_list.append(num)
                    continue
                if node == 1 and one_path_copy[i + 1] == 0:
                    num_list.append(num)
                    num = 0
            violationCount += sum(
                np.array(num_list) > maxWorkingHour / intervalDuration
            )
        return int(violationCount) == 0, int(violationCount)

    def check_feasibility(
        solution_chromosome,
        checkDemand=True,
        checkRushHour=False,
        checkMaxWorkingHour=False,
    ):
        """
        s.t. constraints (make sure initial paths & crossover paths & mutated paths are feasible)
        constraint1: meet demand
        constraint2: during rush hours, one interval is not enough time to commute (optional)
        constraint3: make sure that no driver works more than a few hours continuously
        """
        demandFlag, rushHour, maxWorkingHour = True, True, True
        if checkDemand:
            demandFlag, demandViolationNum = demand_constraint(
                solution_chromosome, tolerance
            )
        if checkRushHour:
            rushHour, rushHourViolationNum = rush_hour_constraint(solution_chromosome)
        if checkMaxWorkingHour:
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                solution_chromosome
            )
        if not demandFlag:
            print("d" + str(demandViolationNum), end="")
        if not rushHour:
            print("r" + str(rushHourViolationNum), end="")
        if not maxWorkingHour:
            print("w" + str(maxWorkingHourViolationNum), end="")
        return demandFlag and rushHour and maxWorkingHour

    def violation_result(
        solution_chromosome,
        checkDemand=True,
        checkRushHour=False,
        checkMaxWorkingHour=False,
    ):
        """
        return violation results
        """
        demandFlag, rushHour, maxWorkingHour = True, True, True
        demandViolationNum, rushHourViolationNum, maxWorkingHourViolationNum = (
            -1,
            -1,
            -1,
        )
        if checkDemand:
            demandFlag, demandViolationNum = demand_constraint(
                solution_chromosome, tolerance
            )
        if checkRushHour:
            rushHour, rushHourViolationNum = rush_hour_constraint(solution_chromosome)
        if checkMaxWorkingHour:
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                solution_chromosome
            )

        if not demandFlag:
            print("d" + str(demandViolationNum), end="")
        if not rushHour:
            print("r" + str(rushHourViolationNum), end="")
        if not maxWorkingHour:
            print("w" + str(maxWorkingHourViolationNum), end="")
        return (demandViolationNum, rushHourViolationNum, maxWorkingHourViolationNum)

    def fitness(solution_chromosome, addPenalty=False):
        """
        objective function ish -> natural selection to pick the good ones
        the lower the better!!
        """
        total_cost = 0
        # basic cost
        for one_path in solution_chromosome:
            target_indices = np.where(one_path == 1)[0]
            if len(target_indices) == 0:
                duration_interval_num = 0
            else:
                duration_interval_num = int(target_indices[-1] - target_indices[0] + 1)
            if duration_interval_num == 0:
                total_cost += 0
            elif duration_interval_num * intervalDuration <= 5:
                total_cost += 90
            elif duration_interval_num * intervalDuration <= 7.5:
                total_cost += 180
            else:
                total_cost += (20 * intervalDuration) * duration_interval_num
        # add penalty
        if addPenalty:
            demandFlag, demandViolationNum = demand_constraint(
                solution_chromosome, tolerance
            )
            rushHour, rushHourViolatonNum = rush_hour_constraint(solution_chromosome)
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                solution_chromosome
            )
            if checkDemandFlag:
                total_cost += alpha * demandViolationNum * demandViolationPenalty
            if checkRushHourFlag:
                total_cost += rushHourViolatonNum * rushHourViolationPenalty
            if maxWorkingHourViolationPenalty:
                total_cost += (
                    maxWorkingHourViolationNum * maxWorkingHourViolationPenalty
                )
        return total_cost

    def generate_population(population_size):
        population, fitness_scores_add_penalty = [], []
        for _ in range(population_size):
            solution_chromosome = generate_random_N_paths(N, intervalNum)
            population.append(solution_chromosome)
            fitness_score_add_penalty = fitness(solution_chromosome, addPenalty=True)
            fitness_scores_add_penalty.append(fitness_score_add_penalty)
        return np.array(population), np.array(fitness_scores_add_penalty)

    def elitism(population, fitness_scores, elitism_cutoff=2):
        elite_indices = np.argpartition(np.array(fitness_scores), elitism_cutoff)[
            :elitism_cutoff
        ]
        return population[elite_indices, :]

    def create_next_generation(
        population, fitness_scores, population_size, elitism_cutoff
    ):
        """
        Randomly pick the good ones and cross them over
        """
        children = []
        while True:
            parents = random.choices(
                population=population,
                weights=[
                    (max(fitness_scores) - score + 1)
                    / (
                        max(fitness_scores) * len(fitness_scores)
                        - sum(fitness_scores)
                        + len(fitness_scores)
                    )
                    for score in fitness_scores
                ],
                k=2,
            )
            kid1, kid2 = single_point_crossover(parents[0], parents[1])
            for _ in range(mutation_num):
                kid1 = mutation(kid1)
            children.append(kid1)
            if len(children) == population_size - elitism_cutoff:
                return np.array(children)
            for _ in range(mutation_num):
                kid2 = mutation(kid2)
            children.append(kid2)
            if len(children) == population_size - elitism_cutoff:
                return np.array(children)

    def single_point_crossover(parent1, parent2):
        """
        Randomly pick the good ones and cross them over
        The crossover point is ideally NOT going to disrupt a path.
        """
        assert parent1.size == parent2.size
        length = len(parent1)
        if length < 2:
            return parent1, parent2
        cut = random.randint(1, length - 1)
        kid1 = np.append(parent1[0:cut, :], parent2[cut:, :]).reshape((N, intervalNum))
        kid2 = np.append(parent2[0:cut, :], parent1[cut:, :]).reshape((N, intervalNum))
        return kid1, kid2

    def mutation(solution_chromosome):
        """
        Mutate only one node in one path for now
        """
        # case 1: concentional mutation implementation
        if mutationType == "Conv":
            path_num, node_num = solution_chromosome.shape
            for k in range(path_num):
                for i in range(node_num):
                    solution_chromosome[k, i] = (
                        solution_chromosome[k, i]
                        if random.random() > mutation_prob
                        else abs(solution_chromosome[k, i] - 1)
                    )
        # case 2: self-designed mutation implementation
        else:
            mutate_path = np.random.randint(0, N)
            mutate_node = np.random.randint(0, intervalNum)
            solution_chromosome[mutate_path][mutate_node] = abs(
                1 - solution_chromosome[mutate_path][mutate_node]
            )
        return solution_chromosome

    def result_stats(progress_with_penalty, progress):
        """
        print important stats & visulize progress_with_penalty
        """
        print("**************************************************************")
        print(
            f"Progress_with_penalty of improvement: {progress_with_penalty[0]} to {progress_with_penalty[-1]}"
        )
        print(f"Progress of improvement: {progress[0]} to {progress[-1]}")
        print(
            "Improvement Rate of progress:",
            abs(progress[-1] - progress[0]) / progress[0],
        )
        print("**************************************************************")
        plt.plot(
            progress_with_penalty, data=progress_with_penalty, label="with penalty"
        )
        plt.plot(progress, data=progress, label="no penalty")
        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()

    def run_evolution(population_size, evolution_depth, elitism_cutoff):
        """
        Main function of Genetic Algorithm
        """
        tic = time.time()

        # first initialize a population
        population, population_fitnesses_add_penalty = generate_population(
            population_size
        )
        initialization_end = time.time()
        print("\nInitialization Done!", initialization_end - tic)
        population_fitnesses = [
            fitness(solution_chromosome) for solution_chromosome in population
        ]
        print(
            f"Initial Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}"
        )
        # keep track of improvement
        progress_with_penalty, progress = [], []

        # start evolving :)
        for i in range(evolution_depth):
            progress_with_penalty.append(min(population_fitnesses_add_penalty))
            progress.append(min(population_fitnesses))
            print(
                f"----------------------------- generation {i + 1} Start! -----------------------------"
            )
            elitism_begin = time.time()
            elites = elitism(
                population, population_fitnesses_add_penalty, elitism_cutoff
            )
            print("Elites selected!")
            children = create_next_generation(
                population,
                population_fitnesses_add_penalty,
                population_size,
                elitism_cutoff,
            )
            print("Children created!")
            population = np.concatenate([elites, children])
            population_fitnesses_add_penalty = [
                fitness(solution_chromosome, addPenalty=True)
                for solution_chromosome in population
            ]
            population_fitnesses = [
                fitness(solution_chromosome) for solution_chromosome in population
            ]

            evol_end = time.time()
            print(
                f"Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}"
            )
            # check best solution feasibility
            minIndex = population_fitnesses_add_penalty.index(
                min(population_fitnesses_add_penalty)
            )
            best_solution = population[minIndex]
            allFeasibilityFlag = check_feasibility(
                best_solution,
                checkDemand=checkDemandFlag,
                checkRushHour=checkRushHourFlag,
                checkMaxWorkingHour=checkMaxWorkingHourFlag,
            )
            print("\nAll constraints met?", allFeasibilityFlag)

            # print best solution
            print("best solution (path):\n", best_solution)
            directional_N_paths = [
                decode_one_path(one_path) for one_path in population[minIndex]
            ]
            link = sum(directional_N_paths)
            print("best solution (link): \n", link)

            print(
                f"---------------------- generation {i + 1} evolved! Time: {evol_end - elitism_begin:.4f}s ----------------------\n"
            )

        # plot results
        # result_stats(progress_with_penalty, progress)

        # print best solution
        minIndex = population_fitnesses_add_penalty.index(
            min(population_fitnesses_add_penalty)
        )
        best_solution = population[minIndex]
        print("best solution (path):\n", best_solution)

        # check if all constraints are met (ideally True)
        print(
            "\nAll constraints met?",
            check_feasibility(
                best_solution,
                checkDemand=checkDemandFlag,
                checkRushHour=checkRushHourFlag,
                checkMaxWorkingHour=checkMaxWorkingHourFlag,
            ),
        )
        directional_N_paths = [
            decode_one_path(one_path) for one_path in population[minIndex]
        ]
        link = sum(directional_N_paths)
        print("best solution (link): \n", link)

        return (
            str(link[1, :]),
            str(link[2, :]),
            str(allFeasibilityFlag),
            progress[-1],
            best_solution,
        )

    (
        baselineOutput1,
        baselineOutput2,
        allFeasibilityFlag,
        minCost,
        best_solution,
    ) = run_evolution(population_size, evolution_depth, elitism_cutoff)

    (
        demandViolationNum,
        rushHourViolationNum,
        maxWorkingHourViolationNum,
    ) = violation_result(
        best_solution,
        checkDemand=checkDemandFlag,
        checkRushHour=checkRushHourFlag,
        checkMaxWorkingHour=checkMaxWorkingHourFlag,
    )
    return render_template(
        "baseline.html",
        baselineOutput1=baselineOutput1,
        baselineOutput2=baselineOutput2,
        allFeasibilityFlag=allFeasibilityFlag,
        minCost=minCost,
        demandViolationNum=demandViolationNum,
        rushHourViolationNum=rushHourViolationNum,
        maxWorkingHourViolationNum=maxWorkingHourViolationNum,
    )


@app.route("/extension", methods=["GET", "POST"])
def extension():
    # parameters
    initial_prob = float(request.form["initial_prob"])
    pusan_prob = float(request.form["pusan_prob"])
    population_size = int(request.form["population_size"])
    elitism_cutoff = 2
    mutationType = "New"  # Conv
    mutation_prob = 0.95
    mutation_num = 1
    loop_limit = int(request.form["loop_limit"])
    evolution_depth = int(request.form["evolution_depth"])
    N = int(request.form["N"])
    D = int(request.form["D"])
    tolerance = int(request.form["tolerance"])
    intervalDuration = 0.5

    # Jinqiao to AB
    JQ2AB_demand1 = request.form["JQ2AB_demand1"]
    JQ2AB_demand2 = request.form["JQ2AB_demand2"]
    JQ2AB_demand3 = request.form["JQ2AB_demand3"]
    JQ2AB_demand4 = request.form["JQ2AB_demand4"]
    JQ2AB_demand5 = request.form["JQ2AB_demand5"]
    JQ2AB_demand6 = request.form["JQ2AB_demand6"]
    JQ2AB_demand7 = request.form["JQ2AB_demand7"]
    JQ2AB_demand8 = request.form["JQ2AB_demand8"]
    JQ2AB_demand9 = request.form["JQ2AB_demand9"]
    JQ2AB_demand10 = request.form["JQ2AB_demand10"]
    JQ2AB_demand11 = request.form["JQ2AB_demand11"]
    JQ2AB_demand12 = request.form["JQ2AB_demand12"]
    JQ2AB_demand13 = request.form["JQ2AB_demand13"]
    JQ2AB_demand14 = request.form["JQ2AB_demand14"]
    JQ2AB_demand15 = request.form["JQ2AB_demand15"]
    JQ2AB_demand16 = request.form["JQ2AB_demand16"]
    JQ2AB_demand17 = request.form["JQ2AB_demand17"]
    JQ2AB_demand18 = request.form["JQ2AB_demand18"]
    JQ2AB_demand19 = request.form["JQ2AB_demand19"]
    JQ2AB_demand20 = request.form["JQ2AB_demand20"]
    JQ2AB_demand21 = request.form["JQ2AB_demand21"]
    JQ2AB_demand22 = request.form["JQ2AB_demand22"]
    JQ2AB_demand23 = request.form["JQ2AB_demand23"]
    JQ2AB_demand24 = request.form["JQ2AB_demand24"]
    JQ2AB_demand25 = request.form["JQ2AB_demand25"]
    JQ2AB_demand26 = request.form["JQ2AB_demand26"]
    JQ2AB_demand27 = request.form["JQ2AB_demand27"]
    JQ2AB_demand28 = request.form["JQ2AB_demand28"]
    JQ2AB_demand29 = request.form["JQ2AB_demand29"]
    JQ2AB_demand30 = request.form["JQ2AB_demand30"]
    JQ2AB_demand31 = request.form["JQ2AB_demand31"]
    JQ2AB_demand32 = request.form["JQ2AB_demand32"]
    JQ2AB_demand33 = request.form["JQ2AB_demand33"]
    JQ2AB_demand34 = request.form["JQ2AB_demand34"]
    # AB to Jinqiao
    AB2JQ_demand1 = request.form["AB2JQ_demand1"]
    AB2JQ_demand2 = request.form["AB2JQ_demand2"]
    AB2JQ_demand3 = request.form["AB2JQ_demand3"]
    AB2JQ_demand4 = request.form["AB2JQ_demand4"]
    AB2JQ_demand5 = request.form["AB2JQ_demand5"]
    AB2JQ_demand6 = request.form["AB2JQ_demand6"]
    AB2JQ_demand7 = request.form["AB2JQ_demand7"]
    AB2JQ_demand8 = request.form["AB2JQ_demand8"]
    AB2JQ_demand9 = request.form["AB2JQ_demand9"]
    AB2JQ_demand10 = request.form["AB2JQ_demand10"]
    AB2JQ_demand11 = request.form["AB2JQ_demand11"]
    AB2JQ_demand12 = request.form["AB2JQ_demand12"]
    AB2JQ_demand13 = request.form["AB2JQ_demand13"]
    AB2JQ_demand14 = request.form["AB2JQ_demand14"]
    AB2JQ_demand15 = request.form["AB2JQ_demand15"]
    AB2JQ_demand16 = request.form["AB2JQ_demand16"]
    AB2JQ_demand17 = request.form["AB2JQ_demand17"]
    AB2JQ_demand18 = request.form["AB2JQ_demand18"]
    AB2JQ_demand19 = request.form["AB2JQ_demand19"]
    AB2JQ_demand20 = request.form["AB2JQ_demand20"]
    AB2JQ_demand21 = request.form["AB2JQ_demand21"]
    AB2JQ_demand22 = request.form["AB2JQ_demand22"]
    AB2JQ_demand23 = request.form["AB2JQ_demand23"]
    AB2JQ_demand24 = request.form["AB2JQ_demand24"]
    AB2JQ_demand25 = request.form["AB2JQ_demand25"]
    AB2JQ_demand26 = request.form["AB2JQ_demand26"]
    AB2JQ_demand27 = request.form["AB2JQ_demand27"]
    AB2JQ_demand28 = request.form["AB2JQ_demand28"]
    AB2JQ_demand29 = request.form["AB2JQ_demand29"]
    AB2JQ_demand30 = request.form["AB2JQ_demand30"]
    AB2JQ_demand31 = request.form["AB2JQ_demand31"]
    AB2JQ_demand32 = request.form["AB2JQ_demand32"]
    AB2JQ_demand33 = request.form["AB2JQ_demand33"]
    AB2JQ_demand34 = request.form["AB2JQ_demand34"]
    demand_JQJY = np.array(
        [
            [
                int(JQ2AB_demand1),
                int(JQ2AB_demand2),
                int(JQ2AB_demand3),
                int(JQ2AB_demand4),
                int(JQ2AB_demand5),
                int(JQ2AB_demand6),
                int(JQ2AB_demand7),
                int(JQ2AB_demand8),
                int(JQ2AB_demand9),
                int(JQ2AB_demand10),
                int(JQ2AB_demand11),
                int(JQ2AB_demand12),
                int(JQ2AB_demand13),
                int(JQ2AB_demand14),
                int(JQ2AB_demand15),
                int(JQ2AB_demand16),
                int(JQ2AB_demand17),
                int(JQ2AB_demand18),
                int(JQ2AB_demand19),
                int(JQ2AB_demand20),
                int(JQ2AB_demand21),
                int(JQ2AB_demand22),
                int(JQ2AB_demand23),
                int(JQ2AB_demand24),
                int(JQ2AB_demand25),
                int(JQ2AB_demand26),
                int(JQ2AB_demand27),
                int(JQ2AB_demand28),
                int(JQ2AB_demand29),
                int(JQ2AB_demand30),
                int(JQ2AB_demand31),
                int(JQ2AB_demand32),
                int(JQ2AB_demand33),
                int(JQ2AB_demand34),
            ],
            [
                int(AB2JQ_demand1),
                int(AB2JQ_demand2),
                int(AB2JQ_demand3),
                int(AB2JQ_demand4),
                int(AB2JQ_demand5),
                int(AB2JQ_demand6),
                int(AB2JQ_demand7),
                int(AB2JQ_demand8),
                int(AB2JQ_demand9),
                int(AB2JQ_demand10),
                int(AB2JQ_demand11),
                int(AB2JQ_demand12),
                int(AB2JQ_demand13),
                int(AB2JQ_demand14),
                int(AB2JQ_demand15),
                int(AB2JQ_demand16),
                int(AB2JQ_demand17),
                int(AB2JQ_demand18),
                int(AB2JQ_demand19),
                int(AB2JQ_demand20),
                int(AB2JQ_demand21),
                int(AB2JQ_demand22),
                int(AB2JQ_demand23),
                int(AB2JQ_demand24),
                int(AB2JQ_demand25),
                int(AB2JQ_demand26),
                int(AB2JQ_demand27),
                int(AB2JQ_demand28),
                int(AB2JQ_demand29),
                int(AB2JQ_demand30),
                int(AB2JQ_demand31),
                int(AB2JQ_demand32),
                int(AB2JQ_demand33),
                int(AB2JQ_demand34),
            ],
        ]
    )

    # Pusan to AB
    PS2AB_demand1 = request.form["PS2AB_demand1"]
    PS2AB_demand2 = request.form["PS2AB_demand2"]
    PS2AB_demand3 = request.form["PS2AB_demand3"]
    PS2AB_demand4 = request.form["PS2AB_demand4"]
    PS2AB_demand5 = request.form["PS2AB_demand5"]
    PS2AB_demand6 = request.form["PS2AB_demand6"]
    PS2AB_demand7 = request.form["PS2AB_demand7"]
    PS2AB_demand8 = request.form["PS2AB_demand8"]
    PS2AB_demand9 = request.form["PS2AB_demand9"]
    PS2AB_demand10 = request.form["PS2AB_demand10"]
    PS2AB_demand11 = request.form["PS2AB_demand11"]
    PS2AB_demand12 = request.form["PS2AB_demand12"]
    PS2AB_demand13 = request.form["PS2AB_demand13"]
    PS2AB_demand14 = request.form["PS2AB_demand14"]
    PS2AB_demand15 = request.form["PS2AB_demand15"]
    PS2AB_demand16 = request.form["PS2AB_demand16"]
    PS2AB_demand17 = request.form["PS2AB_demand17"]
    PS2AB_demand18 = request.form["PS2AB_demand18"]
    PS2AB_demand19 = request.form["PS2AB_demand19"]
    PS2AB_demand20 = request.form["PS2AB_demand20"]
    PS2AB_demand21 = request.form["PS2AB_demand21"]
    PS2AB_demand22 = request.form["PS2AB_demand22"]
    PS2AB_demand23 = request.form["PS2AB_demand23"]
    PS2AB_demand24 = request.form["PS2AB_demand24"]
    PS2AB_demand25 = request.form["PS2AB_demand25"]
    PS2AB_demand26 = request.form["PS2AB_demand26"]
    PS2AB_demand27 = request.form["PS2AB_demand27"]
    PS2AB_demand28 = request.form["PS2AB_demand28"]
    PS2AB_demand29 = request.form["PS2AB_demand29"]
    PS2AB_demand30 = request.form["PS2AB_demand30"]
    PS2AB_demand31 = request.form["PS2AB_demand31"]
    PS2AB_demand32 = request.form["PS2AB_demand32"]
    PS2AB_demand33 = request.form["PS2AB_demand33"]
    PS2AB_demand34 = request.form["PS2AB_demand34"]
    # AB to Pusan
    AB2PS_demand1 = request.form["AB2PS_demand1"]
    AB2PS_demand2 = request.form["AB2PS_demand2"]
    AB2PS_demand3 = request.form["AB2PS_demand3"]
    AB2PS_demand4 = request.form["AB2PS_demand4"]
    AB2PS_demand5 = request.form["AB2PS_demand5"]
    AB2PS_demand6 = request.form["AB2PS_demand6"]
    AB2PS_demand7 = request.form["AB2PS_demand7"]
    AB2PS_demand8 = request.form["AB2PS_demand8"]
    AB2PS_demand9 = request.form["AB2PS_demand9"]
    AB2PS_demand10 = request.form["AB2PS_demand10"]
    AB2PS_demand11 = request.form["AB2PS_demand11"]
    AB2PS_demand12 = request.form["AB2PS_demand12"]
    AB2PS_demand13 = request.form["AB2PS_demand13"]
    AB2PS_demand14 = request.form["AB2PS_demand14"]
    AB2PS_demand15 = request.form["AB2PS_demand15"]
    AB2PS_demand16 = request.form["AB2PS_demand16"]
    AB2PS_demand17 = request.form["AB2PS_demand17"]
    AB2PS_demand18 = request.form["AB2PS_demand18"]
    AB2PS_demand19 = request.form["AB2PS_demand19"]
    AB2PS_demand20 = request.form["AB2PS_demand20"]
    AB2PS_demand21 = request.form["AB2PS_demand21"]
    AB2PS_demand22 = request.form["AB2PS_demand22"]
    AB2PS_demand23 = request.form["AB2PS_demand23"]
    AB2PS_demand24 = request.form["AB2PS_demand24"]
    AB2PS_demand25 = request.form["AB2PS_demand25"]
    AB2PS_demand26 = request.form["AB2PS_demand26"]
    AB2PS_demand27 = request.form["AB2PS_demand27"]
    AB2PS_demand28 = request.form["AB2PS_demand28"]
    AB2PS_demand29 = request.form["AB2PS_demand29"]
    AB2PS_demand30 = request.form["AB2PS_demand30"]
    AB2PS_demand31 = request.form["AB2PS_demand31"]
    AB2PS_demand32 = request.form["AB2PS_demand32"]
    AB2PS_demand33 = request.form["AB2PS_demand33"]
    AB2PS_demand34 = request.form["AB2PS_demand34"]
    demand_PS = np.array(
        [
            [
                int(PS2AB_demand1),
                int(PS2AB_demand2),
                int(PS2AB_demand3),
                int(PS2AB_demand4),
                int(PS2AB_demand5),
                int(PS2AB_demand6),
                int(PS2AB_demand7),
                int(PS2AB_demand8),
                int(PS2AB_demand9),
                int(PS2AB_demand10),
                int(PS2AB_demand11),
                int(PS2AB_demand12),
                int(PS2AB_demand13),
                int(PS2AB_demand14),
                int(PS2AB_demand15),
                int(PS2AB_demand16),
                int(PS2AB_demand17),
                int(PS2AB_demand18),
                int(PS2AB_demand19),
                int(PS2AB_demand20),
                int(PS2AB_demand21),
                int(PS2AB_demand22),
                int(PS2AB_demand23),
                int(PS2AB_demand24),
                int(PS2AB_demand25),
                int(PS2AB_demand26),
                int(PS2AB_demand27),
                int(PS2AB_demand28),
                int(PS2AB_demand29),
                int(PS2AB_demand30),
                int(PS2AB_demand31),
                int(PS2AB_demand32),
                int(PS2AB_demand33),
                int(PS2AB_demand34),
            ],
            [
                int(AB2PS_demand1),
                int(AB2PS_demand2),
                int(AB2PS_demand3),
                int(AB2PS_demand4),
                int(AB2PS_demand5),
                int(AB2PS_demand6),
                int(AB2PS_demand7),
                int(AB2PS_demand8),
                int(AB2PS_demand9),
                int(AB2PS_demand10),
                int(AB2PS_demand11),
                int(AB2PS_demand12),
                int(AB2PS_demand13),
                int(AB2PS_demand14),
                int(AB2PS_demand15),
                int(AB2PS_demand16),
                int(AB2PS_demand17),
                int(AB2PS_demand18),
                int(AB2PS_demand19),
                int(AB2PS_demand20),
                int(AB2PS_demand21),
                int(AB2PS_demand22),
                int(AB2PS_demand23),
                int(AB2PS_demand24),
                int(AB2PS_demand25),
                int(AB2PS_demand26),
                int(AB2PS_demand27),
                int(AB2PS_demand28),
                int(AB2PS_demand29),
                int(AB2PS_demand30),
                int(AB2PS_demand31),
                int(AB2PS_demand32),
                int(AB2PS_demand33),
                int(AB2PS_demand34),
            ],
        ]
    )

    intervalNum = demand_JQJY.shape[-1]
    maxWorkingHour = 4

    checkDemandFlag = True
    checkRushHourFlag = request.form["checkRushHourFlag"] == "True"
    checkMaxWorkingHourFlag = True
    alpha = float(request.form["alpha"])
    demandViolationPenalty = int(request.form["demandViolationPenalty"])
    rushHourViolationPenalty = int(request.form["rushHourViolationPenalty"])
    maxWorkingHourViolationPenalty = int(request.form["maxWorkingHourViolationPenalty"])

    def generate_random_N_paths(N, path_length):
        """
        Randomize N paths where 1 path is like 00 01 00 01 01 01
        """
        one_solution = []
        while len(one_solution) < N:
            one_path_single_digit = random.choices(
                population=[0, 1],
                weights=[1 - initial_prob, initial_prob],
                k=path_length,
            )
            one_path_double_digit = ""
            for i in one_path_single_digit:
                if i == 0:
                    one_path_double_digit += "00"
                elif i == 1:
                    one_path_double_digit += random.choices(
                        population=["10", "01"], weights=[1 - pusan_prob, pusan_prob]
                    )[0]
            if check_path_integrity(one_path_double_digit):
                one_solution.append(one_path_double_digit)
        return one_solution

    def check_solution_integrity(solution):
        for one_path_double_digit in solution:
            if not check_path_integrity(one_path_double_digit):
                return False
        return True

    def check_path_integrity(one_path_double_digit):
        last_visited = None
        for i in range(len(one_path_double_digit)):
            if i % 2 == 0:
                two_digits = one_path_double_digit[i : i + 2]
                if two_digits != "00":
                    # first time going to AB
                    if last_visited is None:
                        last_visited = "AB"
                    # following times
                    elif last_visited == "JQJY":
                        if two_digits == "01":
                            return False
                        else:  # '10'
                            last_visited = "AB"
                    elif last_visited == "PS":
                        if two_digits == "10":
                            return False
                        else:  # '01'
                            last_visited = "AB"
                    else:
                        if two_digits == "10":
                            last_visited = "JQJY"
                        else:  # '01'
                            last_visited = "PS"
        return True

    def decode_one_path(one_path_double_digit):
        decoded, initial_node, last_visited = [], None, None
        for i in range(len(one_path_double_digit)):
            if i % 2 == 0:
                two_digits = one_path_double_digit[i : i + 2]
                if two_digits == "00":
                    if last_visited is None:
                        decoded.append([0, 0, 0, 0, 0, 0, 0])
                    elif last_visited == "JQJY":
                        decoded.append([1, 0, 0, 0, 0, 0, 0])
                    elif last_visited == "AB":
                        decoded.append([0, 0, 0, 1, 0, 0, 0])
                    else:  # PS
                        decoded.append([0, 0, 0, 0, 0, 0, 1])
                elif two_digits == "10":
                    if last_visited is None:
                        initial_node = 0
                        last_visited = "AB"
                        decoded.append([0, 1, 0, 0, 0, 0, 0])
                    elif last_visited == "AB":
                        last_visited = "JQJY"
                        decoded.append([0, 0, 1, 0, 0, 0, 0])
                    elif last_visited == "JQJY":
                        last_visited = "AB"
                        decoded.append([0, 1, 0, 0, 0, 0, 0])
                    else:
                        print("SOMETHING IS WRONG1!!!")
                elif two_digits == "01":
                    if last_visited is None:
                        initial_node = -1
                        last_visited = "AB"
                        decoded.append([0, 0, 0, 0, 0, 1, 0])
                    elif last_visited == "AB":
                        last_visited = "PS"
                        decoded.append([0, 0, 0, 0, 1, 0, 0])
                    elif last_visited == "PS":
                        last_visited = "AB"
                        decoded.append([0, 0, 0, 0, 0, 1, 0])
                    else:
                        print("SOMETHING IS WRONG2!!!")
        decoded = np.array(decoded).T
        decoded_sum = decoded.sum(axis=0)
        if sum(decoded_sum) == 0:
            if random.random() <= pusan_prob:
                decoded[0, :] = 0
            else:
                decoded[0, :] = 1
            return decoded
        k = 0
        while decoded_sum[k] == 0:
            decoded[initial_node, k] = 1
            k += 1
        return decoded

    def demand_constraint(binary_N_paths, tolerance):
        """
        make sure the demand is met
        """
        directional_N_paths = [decode_one_path(one_path) for one_path in binary_N_paths]
        link = sum(directional_N_paths)
        link_JQJY = link[:4, :]
        link_PS = link[-1:2:-1, :]
        JQJY_supply_demand_difference = np.greater_equal(
            demand_JQJY - tolerance, link_JQJY[1:3, :] * D
        )
        JQJY_mask = (demand_JQJY - tolerance) - (link_JQJY[1:3, :] * D)
        PS_supply_demand_difference = np.greater_equal(
            demand_PS - tolerance, link_PS[1:3, :] * D
        )
        PS_mask = (demand_PS - tolerance) - (link_PS[1:3, :] * D)
        missedDemandNumJQJY = np.sum(JQJY_supply_demand_difference * JQJY_mask)
        missedDemandNumPS = np.sum(PS_supply_demand_difference * PS_mask)
        return int(missedDemandNumJQJY + missedDemandNumPS) == 0, int(
            missedDemandNumJQJY + missedDemandNumPS
        )

    def rush_hour_constraint(binary_N_paths):
        """
        during rush hours, one interval is not enough time to commute
        """
        violationCount = 0
        for one_path_double_digit in binary_N_paths:
            one_path_single_digit_list = []
            one_path_double_digit_list = list(one_path_double_digit)
            for i in range(len(one_path_double_digit_list)):
                if i % 2 == 0:
                    one_path_single_digit_list.append(
                        int(one_path_double_digit_list[i])
                        + int(one_path_double_digit_list[i + 1])
                    )
            # morning rush hour
            if one_path_single_digit_list[1] + one_path_single_digit_list[2] == 2:
                violationCount += 1
            # evening rush hour
            if one_path_single_digit_list[21] + one_path_single_digit_list[22] == 2:
                violationCount += 1
        return int(violationCount) == 0, int(violationCount)

    def max_working_hour_constraint(binary_N_paths):
        """
        make sure that no driver works more than a few hours continuously
        """
        violationCount = 0
        for one_path_double_digit in binary_N_paths:
            one_path_single_digit_list = []
            one_path_double_digit_list = list(one_path_double_digit)
            for i in range(len(one_path_double_digit_list)):
                if i % 2 == 0:
                    one_path_single_digit_list.append(
                        int(one_path_double_digit_list[i])
                        + int(one_path_double_digit_list[i + 1])
                    )
            num, num_list = 0, []
            one_path_copy = one_path_single_digit_list.copy()
            # first check if rush hour 10 actually is 11.
            if checkRushHourFlag:
                if one_path_copy[1] == 1 and one_path_copy[2] == 0:
                    one_path_copy[2] = 1
                if one_path_copy[21] == 1 and one_path_copy[22] == 0:
                    one_path_copy[22] = 1
            for i, node in enumerate(one_path_copy):
                num += node
                if i + 1 == len(one_path_copy):
                    num_list.append(num)
                    continue
                if node == 1 and one_path_copy[i + 1] == 0:
                    num_list.append(num)
                    num = 0
            violationCount += sum(
                np.array(num_list) > maxWorkingHour / intervalDuration
            )
        return int(violationCount) == 0, int(violationCount)

    def check_feasibility(
        binary_N_paths, checkDemand=True, checkRushHour=False, checkMaxWorkingHour=False
    ):
        """
        s.t. constraints (make sure initial paths & crossover paths & mutated paths are feasible)
        constraint1: meet demand
        constraint2: during rush hours, one interval is not enough time to commute (optional)
        constraint3: make sure that no driver works more than a few hours continuously
        """
        demandFlag, rushHour, maxWorkingHour = True, True, True
        if checkDemand:
            demandFlag, demandViolationNum = demand_constraint(
                binary_N_paths, tolerance
            )
        if checkRushHour:
            rushHour, rushHourViolationNum = rush_hour_constraint(binary_N_paths)
        if checkMaxWorkingHour:
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                binary_N_paths
            )
        if not demandFlag:
            print("d" + str(demandViolationNum), end="")
        if not rushHour:
            print("r" + str(rushHourViolationNum), end="")
        if not maxWorkingHour:
            print("w" + str(maxWorkingHourViolationNum), end="")
        return demandFlag and rushHour and maxWorkingHour

    def violation_result(
        solution_chromosome,
        checkDemand=True,
        checkRushHour=False,
        checkMaxWorkingHour=False,
    ):
        """
        return violation results
        """
        demandFlag, rushHour, maxWorkingHour = True, True, True
        demandViolationNum, rushHourViolationNum, maxWorkingHourViolationNum = (
            -1,
            -1,
            -1,
        )
        if checkDemand:
            demandFlag, demandViolationNum = demand_constraint(
                solution_chromosome, tolerance
            )
        if checkRushHour:
            rushHour, rushHourViolationNum = rush_hour_constraint(solution_chromosome)
        if checkMaxWorkingHour:
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                solution_chromosome
            )

        if not demandFlag:
            print("d" + str(demandViolationNum), end="")
        if not rushHour:
            print("r" + str(rushHourViolationNum), end="")
        if not maxWorkingHour:
            print("w" + str(maxWorkingHourViolationNum), end="")
        return (demandViolationNum, rushHourViolationNum, maxWorkingHourViolationNum)

    def fitness(binary_N_paths, addPenalty=False):
        """
        objective function ish -> natural selection to pick the good ones
        the lower the better!!
        """
        total_cost = 0
        # basic cost
        for one_path_double_digit in binary_N_paths:
            one_path_single_digit_list = []
            one_path_double_digit_list = list(one_path_double_digit)
            for i in range(len(one_path_double_digit_list)):
                if i % 2 == 0:
                    one_path_single_digit_list.append(
                        int(one_path_double_digit_list[i])
                        + int(one_path_double_digit_list[i + 1])
                    )
            one_path_single_digit_np = np.array(one_path_single_digit_list)
            target_indices = np.where(one_path_single_digit_np == 1)[0]
            if len(target_indices) == 0:
                duration_interval_num = 0
            else:
                duration_interval_num = int(target_indices[-1] - target_indices[0] + 1)
            if duration_interval_num == 0:
                total_cost += 0
            elif duration_interval_num * intervalDuration <= 5:
                total_cost += 90
            elif duration_interval_num * intervalDuration <= 7.5:
                total_cost += 180
            else:
                total_cost += (20 * intervalDuration) * duration_interval_num
        # add penalty
        if addPenalty:
            demandFlag, demandViolationNum = demand_constraint(
                binary_N_paths, tolerance
            )
            rushHour, rushHourViolatonNum = rush_hour_constraint(binary_N_paths)
            maxWorkingHour, maxWorkingHourViolationNum = max_working_hour_constraint(
                binary_N_paths
            )
            if checkDemandFlag:
                total_cost += alpha * demandViolationNum * demandViolationPenalty
            if checkRushHourFlag:
                total_cost += rushHourViolatonNum * rushHourViolationPenalty
            if maxWorkingHourViolationPenalty:
                total_cost += (
                    maxWorkingHourViolationNum * maxWorkingHourViolationPenalty
                )
        return total_cost

    def generate_population(population_size):
        population, fitness_scores_add_penalty = [], []
        for _ in range(population_size):
            binary_N_paths = generate_random_N_paths(N, intervalNum)
            population.append(binary_N_paths)
            fitness_score_add_penalty = fitness(binary_N_paths, addPenalty=True)
            fitness_scores_add_penalty.append(fitness_score_add_penalty)
        return np.array(population), np.array(fitness_scores_add_penalty)

    def elitism(population, fitness_scores, elitism_cutoff=2):
        elite_indices = np.argpartition(np.array(fitness_scores), elitism_cutoff)[
            :elitism_cutoff
        ]
        return population[elite_indices, :]

    def create_next_generation(
        population, population_fitnesses_add_penalty, population_size, elitism_cutoff
    ):
        """
        Randomly pick the good ones and cross them over
        """
        children = []
        while True:
            parents = random.choices(
                population=population,
                weights=[
                    (max(population_fitnesses_add_penalty) - score + 1)
                    / (
                        max(population_fitnesses_add_penalty)
                        * len(population_fitnesses_add_penalty)
                        - sum(population_fitnesses_add_penalty)
                        + len(population_fitnesses_add_penalty)
                    )
                    for score in population_fitnesses_add_penalty
                ],
                k=2,
            )
            kid1, kid2 = single_point_crossover(parents[0], parents[1])
            for _ in range(mutation_num):
                kid1 = single_mutation(kid1)
            children.append(kid1)
            if len(children) == population_size - elitism_cutoff:
                return np.array(children)
            for _ in range(mutation_num):
                kid2 = single_mutation(kid2)
            children.append(kid2)
            if len(children) == population_size - elitism_cutoff:
                return np.array(children)

    def single_point_crossover(parent1, parent2):
        """
        Randomly pick the good ones and cross them over
        """
        assert parent1.size == parent2.size
        length = len(parent1)
        if length < 2:
            return parent1, parent2
        count = 0
        while count <= loop_limit:
            cut = random.randint(1, length - 1) * 2
            kid1 = np.array(list(parent1)[:cut] + list(parent2)[cut:])
            kid2 = np.array(list(parent2)[:cut] + list(parent1)[cut:])
            if check_solution_integrity(kid1) and check_solution_integrity(kid2):
                return kid1, kid2
            elif check_solution_integrity(kid1) and not check_solution_integrity(kid2):
                return kid1, None
            elif not check_solution_integrity(kid1) and check_solution_integrity(kid2):
                return None, kid2
            count += 1
        return parent1, parent2

    def single_mutation(binary_N_paths):
        """
        Mutate only one node in one path for now
        """
        count = 0
        binary_N_paths_copy = binary_N_paths.copy()
        while count <= loop_limit:
            mutate_path = np.random.randint(0, N)
            mutate_index = np.random.randint(0, intervalNum) * 2
            double_digits_to_mutate = binary_N_paths_copy[mutate_path][
                mutate_index : mutate_index + 2
            ]
            pool = ["00", "01", "10"]
            pool.remove(double_digits_to_mutate)
            mutated_double_digits = random.choices(population=pool)[0]
            original_string = binary_N_paths_copy[mutate_path]
            mutated_string = (
                original_string[:mutate_index]
                + mutated_double_digits
                + original_string[mutate_index + 2 :]
            )
            if check_path_integrity(mutated_string):
                binary_N_paths_copy[mutate_path] = mutated_string
                return binary_N_paths_copy
            count += 1
        return binary_N_paths

    def result_stats(progress_with_penalty, progress):
        """
        print important stats & visulize progress_with_penalty
        """
        print("**************************************************************")
        print(
            f"Progress_with_penalty of improvement: {progress_with_penalty[0]} to {progress_with_penalty[-1]}"
        )
        print(f"Progress of improvement: {progress[0]} to {progress[-1]}")
        print(
            "Improvement Rate of progress:",
            abs(progress[-1] - progress[0]) / progress[0],
        )
        print("**************************************************************")
        plt.plot(
            progress_with_penalty, data=progress_with_penalty, label="with penalty"
        )
        plt.plot(progress, data=progress, label="no penalty")
        plt.xlabel("Generation")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()

    def run_evolution(population_size, evolution_depth, elitism_cutoff):
        """
        Main function of Genetic Algorithm
        """
        tic = time.time()

        # first initialize a population
        population, population_fitnesses_add_penalty = generate_population(
            population_size
        )
        initialization_end = time.time()
        print(f"\nInitialization Done! Time: {initialization_end - tic:.6f}s")
        population_fitnesses = [
            fitness(binary_N_paths) for binary_N_paths in population
        ]
        print(
            f"Initial Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}"
        )
        # keep track of improvement
        progress_with_penalty, progress = [], []

        # start evolving :)
        for i in range(evolution_depth):
            progress_with_penalty.append(min(population_fitnesses_add_penalty))
            progress.append(min(population_fitnesses))
            print(
                f"----------------------------- generation {i + 1} Start! -----------------------------"
            )
            elitism_begin = time.time()
            elites = elitism(
                population, population_fitnesses_add_penalty, elitism_cutoff
            )
            print("Elites selected!")
            children = create_next_generation(
                population,
                population_fitnesses_add_penalty,
                population_size,
                elitism_cutoff,
            )
            print("Children created!")
            population = np.concatenate([elites, children])
            population_fitnesses_add_penalty = [
                fitness(binary_N_paths, addPenalty=True)
                for binary_N_paths in population
            ]
            population_fitnesses = [
                fitness(binary_N_paths) for binary_N_paths in population
            ]

            evol_end = time.time()
            print(
                f"Min Cost: {min(population_fitnesses_add_penalty)} -> {min(population_fitnesses)}"
            )
            # check best solution feasibility
            minIndex = population_fitnesses_add_penalty.index(
                min(population_fitnesses_add_penalty)
            )
            best_solution = population[minIndex]
            allFeasibilityFlag = check_feasibility(
                best_solution,
                checkRushHour=checkRushHourFlag,
                checkMaxWorkingHour=checkMaxWorkingHourFlag,
            )
            print("\nAll constraints met?", allFeasibilityFlag)

            # print best solution
            print("best solution (path):\n", best_solution)
            directional_N_paths = [
                decode_one_path(one_path) for one_path in population[minIndex]
            ]
            link = sum(directional_N_paths)
            print("best solution (link): \n", link)

            print(
                f"---------------------- generation {i + 1} evolved! Time: {evol_end - elitism_begin:.4f}s ----------------------\n"
            )

        # plot results
        # result_stats(progress_with_penalty, progress)

        # print best solution
        minIndex = population_fitnesses_add_penalty.index(
            min(population_fitnesses_add_penalty)
        )
        best_solution = population[minIndex]
        print("best solution (path):\n", best_solution)

        # check if all constraints are met (ideally True)
        print(
            "\nAll constraints met?",
            check_feasibility(
                best_solution,
                checkDemand=checkDemandFlag,
                checkRushHour=checkRushHourFlag,
                checkMaxWorkingHour=checkMaxWorkingHourFlag,
            ),
        )
        directional_N_paths = [
            decode_one_path(one_path) for one_path in population[minIndex]
        ]
        link = sum(directional_N_paths)
        print("best solution (link): \n", link)
        return (
            str(link[1]),
            str(link[2]),
            str(link[5]),
            str(link[4]),
            str(allFeasibilityFlag),
            progress[-1],
            best_solution,
        )

    (
        extensionOutput1,
        extensionOutput2,
        extensionOutput3,
        extensionOutput4,
        allFeasibilityFlag,
        minCost,
        best_solution,
    ) = run_evolution(population_size, evolution_depth, elitism_cutoff)

    (
        demandViolationNum,
        rushHourViolationNum,
        maxWorkingHourViolationNum,
    ) = violation_result(
        best_solution,
        checkDemand=checkDemandFlag,
        checkRushHour=checkRushHourFlag,
        checkMaxWorkingHour=checkMaxWorkingHourFlag,
    )

    return render_template(
        "extension.html",
        extensionOutput1=extensionOutput1,
        extensionOutput2=extensionOutput2,
        extensionOutput3=extensionOutput3,
        extensionOutput4=extensionOutput4,
        allFeasibilityFlag=allFeasibilityFlag,
        minCost=minCost,
        demandViolationNum=demandViolationNum,
        rushHourViolationNum=rushHourViolationNum,
        maxWorkingHourViolationNum=maxWorkingHourViolationNum,
    )


if __name__ == "__main__":
    app.run("127.0.0.1", 5000) ##, debug=True)

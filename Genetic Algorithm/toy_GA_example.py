'''
Toy example of the Knapsack problem from https://www.youtube.com/watch?v=nhT56blfRpE
Also check part 1 of the video for theory explanation: https://www.youtube.com/watch?v=uQj5UNhCPuo&ab_channel=KieCodes

Credit: Kie Codes
'''

from typing import Callable, List, Tuple
from collections import namedtuple
from random import choices, randint, random, randrange
from functools import partial

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulationFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
Thing = namedtuple('Thing', ['name', 'value', 'weight'])

things = [
    Thing('Laptop', 500, 2200),
    Thing('Heaphones', 150, 160),
    Thing('Coffee Mug', 60, 350),
    Thing('Notepad', 40, 333),
    Thing('Water Bottle', 30, 192)
]

more_things = [
    Thing('Mints', 5, 25),
    Thing('Socks', 10, 30),
    Thing('Tissues', 15, 80),
    Thing('Phone', 500, 200),
    Thing('Baseball Cap', 100, 70)
] + things

def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of the same length!")  
    weight, value = 0, 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                return 0
    return value

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genome a and b must be of same length")
    length = len(a)
    if length < 2:
        return a, b
    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
        return genome

def run_evolution(
    populate_func: FitnessFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        if fitness_func(population[0]) >= fitness_limit:
            break
        next_generation = population[0:2]
        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        population = next_generation
        
    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population, i

population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(things)
    ),
    fitness_func=partial(
        fitness, things=things, weight_limit=3000
    ),
    fitness_limit=740,
    generation_limit=100
)

def genome_to_things(genome: Genome, things: List[Thing]) -> List[Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    return result

print(f"number of generation: {generations}")
print(f"best solution: {genome_to_things(population[0], things)}")

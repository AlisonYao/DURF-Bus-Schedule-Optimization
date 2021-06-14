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

print(generate_population(11, 4))

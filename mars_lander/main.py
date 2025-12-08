import numpy as np
from parameter import *
import random

def fitness(gene):
    return random.uniform(0, 1)

population = [np.random.randn(gene_num) for _ in range(population_num)]

for generation in range(max_generation):
    for gene in population:
        score = fitness(gene)
        


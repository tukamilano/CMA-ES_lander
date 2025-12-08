import math

input_size = 13
hidden_size = 8
output_size = 6
gene_num = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
population_num = 4 + (3 * math.floor(math.log(gene_num)))

max_generation = 100
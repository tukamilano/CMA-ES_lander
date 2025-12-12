input_size = 15
hidden_sizes = [24]
hidden_size = hidden_sizes[0]
output_size = 2

layer_sizes = [input_size, *hidden_sizes, output_size]
gene_num = sum(i * o for i, o in zip(layer_sizes[:-1], layer_sizes[1:])) + sum(layer_sizes[1:])

moe_hidden_size = 16
moe_output_size = 5
if moe_hidden_size > 0:
	moe_gene_num = (
		(input_size * moe_hidden_size)
		+ moe_hidden_size
		+ (moe_hidden_size * moe_output_size)
		+ moe_output_size
	)
else:
	moe_gene_num = (input_size * moe_output_size) + moe_output_size

max_generation = 200
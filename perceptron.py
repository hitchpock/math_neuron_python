from Neuron import Neuron

and_n = Neuron([1., 1.], -1.)

or_n = Neuron([2., 2.], -1.)

not_n = Neuron([-1.], 1.)

data1 = [[0], [1]]
data2 = [[0, 0], [0, 1], [1, 0], [1, 1]]

print("Compute with bias\nTable\tAND\tOR")
for line in data2:
    print(line, "\t", and_n.compute_with_bias(line),
          "\t", or_n.compute_with_bias(line))

print("\nTable\tNOT")
for line in data1:
    print(line, "\t", not_n.compute_with_bias(line))


print("\n\nCompute with threshold\nTable\tAND\tOR")
for line in data2:
    print(line, "\t", and_n.compute_with_threshold(line),
          "\t", or_n.compute_with_bias(line))

print("\nTable\tNOT")
for line in data1:
    print(line, "\t", not_n.compute_with_threshold(line))

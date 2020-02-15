import network
# number of input ,hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 1
 
# learning rate is 0.3
learning_rate = 0.3
 
# create insrance of neural network
n = network.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.loaddata()
print(n.query([0,0,0]))
print(n.query([1,0,0]))
print(n.query([1,1,0]))
print(n.query([1,1,1]))
import network
import time
# number of input ,hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 1
 
# learning rate is 0.3
learning_rate = 0.3
 
# create insrance of neural network
n = network.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#print(n.query([1.0, 0.5, -1.5]))
# while x<100000:
#     n.train([1.0, 0.5, 1.5],[1.0, 0.5, 1.5])
#     n.train([2.0, 0.6, 1.6],[2.0, 0.6, 1.6])
#     n.train([1.9, 10.5, 8],[1.9, 10.5, 8])
#     n.train([20.0, 0.66, 1.98],[20.0, 0.66, 1.98])
#     x=x+1



# while x<100000:
#     inputs=[]
#     element=random.randint(0,100)
#     outputs=[]
#     inputs.append(element)
#     inputs.append(element*2)
#     inputs.append(element*5)
#     #print(outputs)
#     n.train(inputs,inputs)
#     x=x+1
while True:
    n.loaddata()
    print("Train Start")
    x=0
    while x<1000:
        n.train([0,0,0],[0])
        n.train([0,0,1],[0])
        n.train([0,1,1],[1])
        n.train([1,1,1],[1])
        n.train([1,0,1],[1])
        n.train([0,1,0],[0])
        n.train([1,0,0],[0])
        n.train([1,1,0],[1]) 
        x=x+1   
    n.savedata()
    print("Train Finished")
    print("System Sleeping")
    time.sleep(3)

# n.train([0,0,0],[0])
# n.train([0,0,1],[0])
# n.train([0,1,1],[1])
# n.train([1,1,1],[1])
# n.train([1,0,1],[1])
# n.train([0,1,0],[0])
# n.train([1,0,0],[0])
# n.train([1,1,0],[1])      
 
# print(n.query([0,0,0]))
# print(n.query([1,0,0]))
# print(n.query([1,1,0]))
# print(n.query([1,1,1]))
# numpy.random.rand(3, 3) - 0.5
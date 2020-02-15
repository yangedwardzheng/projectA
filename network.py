# neural network definition
import numpy
import scipy.special
import random 
 
 
class neuralNetwork:
 
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j,where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # self.wih = (numpy.random.random(self.hnodes,self.inodes)-0.5)
        # self.who = (numpy.random.random(self.onodes,self.hnodes)-0.5)
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        self.activation_funcation = lambda x: scipy.special.expit(x)
        pass
 
    # train the neural network
    def train(self, inputs_list, targets_list):
        # inputs = numpy.array(inputs_list, ndmin=2).T
        # targes = numpy.array(targets_list, ndmin=2).T
        
        # hidden_inputs = numpy.dot(self.wih, inputs)
        # hidden_outputs = self.activation_funcation(hidden_inputs)
        # final_inputs = numpy.dot(self.who, hidden_outputs)
        # final_ouputs = self.activation_funcation(final_inputs)
        # output_errors = targes - final_ouputs
        # hidden_errors = numpy.dot(self.who.T, output_errors)
        # self.who += self.lr * numpy.dot((output_errors * final_ouputs * (1.0 - final_ouputs)),
        #                                 numpy.transpose(hidden_outputs))
        # self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        #                                 numpy.transpose(inputs))


        inputs = numpy.array(inputs_list, ndmin=2).T
        targes = numpy.array(targets_list, ndmin=2).T
        x=0
        while x < 100:
            hidden_inputs = numpy.dot(self.wih, inputs)
            hidden_outputs = self.activation_funcation(hidden_inputs)
            final_inputs = numpy.dot(self.who, hidden_outputs)
            final_ouputs = self.activation_funcation(final_inputs)
            output_errors = targes - final_ouputs
            hidden_errors = numpy.dot(self.who.T, output_errors)
            self.who += self.lr * numpy.dot((output_errors * final_ouputs * (1.0 - final_ouputs)),
                                            numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                            numpy.transpose(inputs))
            x=x+1
            #print(self.who)

        pass
    # query the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_funcation(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_funcation(final_inputs)
        return final_outputs

        
    def savedata(self):
        file1 = open("weight.txt","w")
        for i in self.who[0]:
            file1.write(str(i))
            file1.write("\n")
        #file1.write(str("a"))
        for i in self.wih:
            for j in i:
                file1.write(str(j))
                file1.write("\n")  
    def loaddata(self):
        file2 = open("weight.txt","r")
        data=""
        #load weight h o
        for i in range(3):
            data=""
            for datas in file2.readline():
                data=data+datas    
            self.who[0][i]=data 
        #load weight i h  
        for j in range(3):
            for i in range(3):
                data=""
                for datas in file2.readline():
                    data=data+datas
                self.wih[j][i]=data 
           
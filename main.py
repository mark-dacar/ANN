import random
import math


# Generates the data used for the neural network. This data adapts for different sizes
# of input and output vectors. The formula for the data is to find the numbers up to the
# size of the output vector that divide with a random number with a remainder of 0. The random number
# is in the range of [0, 2^inSize). This allows the binary version of the number to fit
# into the entirety of the input vector.
def generateInsAndOuts(inSize, outSize):
    file = open("ANN_Data.txt", 'w')
    testSize, testVars = 150, []
    for i in range(0, testSize):
        testIn = int(random.randrange(0, 2 ** inSize))
        testOut = []
        for j in range(1, outSize + 1):
            if testIn % (j + 1) == 0:
                testOut.insert(-1, '1')
            else:
                testOut.insert(-1, '0')

        testIn = bin(testIn)
        testIn = testIn[2:]
        if len(testIn) < inSize:
            diff = inSize - len(testIn)
            while diff > 0:
                testIn = '0' + testIn
                diff -= 1

        testOut = ''.join(testOut)
        file.write(str(testIn) + " " + testOut)
        if not i == testSize - 1:
            file.write('\n')
    file.close()


# Takes user input and formats it to be usable by the program
def inputInsAndOuts(inSize, outSize):
    file = open("ANN_Data.txt", "w")

    # DataList stores the values passed by the user to be formatted to be saved.
    # inputs will store the formatted inputs
    # outputs will store the formatted outputs
    dataList, inputs, outputs = [], [], []

    # Input loop. Will continuously take in user input until user inputs "FINISHED"
    print("Please enter data in this form: <1,0,0,1> <0,1>" +
          "\n\tWhere the vectors are the input and output, respectively.\n" +
          "\tEnter [FINISHED] when the program should proceed.")

    dataSelection = ''
    while dataSelection != "FINISHED":
        dataSelection = input("C: ")
        if dataSelection != "FINISHED":
            dataList.append(dataSelection)

    for i in range(0, len(dataList)):
        first, second = dataList[i].split(' ')
        first, second = first[1:-1], second[1:-1]
        first, second = first.split(','), second.split(',')
        file.write(''.join(first) + ' ' + ''.join(second))
        if i != len(dataList) - 1:
            file.write("\n")

    file.close()


# Converts string data into usable formats: lists of integers,
# with each integer being either a 0 or a 1.
def readData():
    file = open("ANN_Data.txt", "r")
    data = file.read()
    data = data.split('\n')

    ins, outs = [], []
    for d in data:
        i, j = d.split(' ')
        i, j = list(i), list(j)
        for pos in range(0, len(i)):
            i[pos] = int(i[pos])
        for pos in range(0, len(j)):
            j[pos] = int(j[pos])
        ins.append(i), outs.append(j)
    file.close()
    return ins, outs


# Algorithm for creating the random matrices
def createRandomMatrix(rows, cols):
    lb, ub, matrix = -1.0, 1.0, []

    for i in range(0, rows):
        vals = []
        for j in range(0, cols):
            vals.append(random.uniform(lb, ub))
        matrix.append(vals)

    return matrix


# Algorithm for generating the bias vectors
def createBiasVector(bias, num):
    bv = []
    for i in range(0, num):
        bv.append(bias)
    return bv


# Algorithm that determines if perceptrons fire in a layer
# Returns a list of outputs
def thresholdFire(inputs, thresh):
    outputs = []
    for i in inputs:
        if i > thresh:
            outputs.append(1)
        else:
            outputs.append(0)
    return outputs


# This method performs the back propagation and adjusts the weights of each matrix per cycle.
def adjustWeight(matrices, actual, targVector, learnRate):
    eIndex = identifyErrorIndices(targVector, actual)
    for i in eIndex:
        layer, edges = 0, []
        deriveEdges(matrices, i, layer, edges)

        for e in edges:
            print('\n\tModifying: matrix ' + str(len(matrices) + e[0]) + ", row " + str(e[1]) + ", column " + str(e[2]))
            print('\t\tFrom: ' + str(e[3]))
            e[3] += learnRate * (targVector[i] - actual[i])
            print('\t\tTo: ' + str(e[3]))

        modifyWeights(edges, matrices)


# Determines which nodes on the output vector need to be updated
def identifyErrorIndices(targVector, actual):
    indices = []
    for i in range(0, len(targVector)):
        if targVector[i] != actual[i]:
            indices.append(i)

    return indices


# Finds all weights to be modified for a given node, recursively.
def deriveEdges(matrices, indice, layer, edges):
    node = indice

    if layer == len(matrices):
        return

    # Finds all weights for this current node
    for i in range(0, len(matrices[-1 - layer])):
        if (matrices[-1 - layer][i][node] != 0) and (
                matrices.count((-1 - layer, i, node, matrices[-1 - layer][i][node])) == 0):
            edges.append([-1 - layer, i, node, matrices[-1 - layer][i][node]])

            # Recursively finds all weights that impact this weight in previous layer
            deriveEdges(matrices, i, layer + 1, edges)
    return


def modifyWeights(edges, matrices):
    for e in edges:
        matrices[e[0]][e[1]][e[2]] = e[3]


def main():
    numIn = int(input("Please set the size of the input vector: "))
    numOut = int(input("Please set the size of the output vector: "))

    # Used to determine if the data should be manually inputted or generated randomly
    dataSelection = ''
    while dataSelection not in ["MANUAL", "AUTO"]:
        dataSelection = input("Please decide how the data will be generated." +
                              "\n\tEnter [MANUAL] to enter the input and output values manually." +
                              "\n\tEnter [AUTO] to create the values automatically." +
                              "\n\nPlease make a menu selection: ")

    if dataSelection == "AUTO":
        generateInsAndOuts(numIn, numOut)
        print("Data has been successfully generated.\n")
    else:
        inputInsAndOuts(numIn, numOut)
        print("Data has been successfully inserted.\n")

    # Creates two lists:
    #   -ins stores input vectors
    #   -outs stores output vectors
    ins, outs = readData()

    numPerc = int(input("Please set the number of perceptrons in each hidden layer: "))
    numLayers = int(input("Please set the number of hidden layers: "))
    bias = float(input("Please set the perceptron bias: "))
    thresh = float(input("Please set the perceptron firing threshold: "))
    cycles = int(input("Please set the number of cycles: "))
    learnRate = 0.65
    errorThresh = 0.9

    # _____MATRIX GENERATION________
    # Generates the first and last matrices
    inMatrix = createRandomMatrix(numIn, numPerc)
    outMatrix = createRandomMatrix(numPerc, numOut)
    matrices = [inMatrix]

    # Generates the rest of the matrices
    for i in range(1, numLayers):
        matrices.append(createRandomMatrix(numPerc, numPerc))

    matrices.append(outMatrix)

    # _____BIAS VECTOR GENERATION ______
    inBias = createBiasVector(bias, numIn)
    outBias = createBiasVector(bias, numOut)

    # All hidden layers use the same bias vector
    hiddenBias = createBiasVector(bias, numPerc)

    # _____RUNNING THE CYCLES _____
    ith, actual = 0, []
    while cycles > 0:
        # Input and target of this cycle
        inV, targVector = ins[ith], outs[ith]

        print("Input:\n" + str(inV) + "\n")
        print("Target Output:\n" + str(targVector) + "\n")

        # inVectors will store the input vectors of each layer
        # outVectors will store the output vectors of each layer
        inVectors, outVectors, inputs, outputs = [], [], [], []
        for i in range(0, len(inV)):
            inputs.append(inV[i] + inBias[i])

        # The inputs + the bias will count as the input vector for each layer
        inVectors.append(inputs)

        inVectors.append(thresholdFire(inputs, thresh))
        outVectors.append(inVectors[-1])

        # Calculating the inputs and outputs of the hidden layers
        k = 1
        while k <= numLayers:
            lastInput = []

            for i in range(0, len(matrices[k - 1][0])):
                tally = 0
                for j in range(0, len(matrices[k - 1])):
                    tally += inVectors[-1][j] * matrices[k - 1][j][i]
                lastInput.append(tally)

            outVectors.append(lastInput)

            for i in range(0, len(lastInput)):
                lastInput[i] += hiddenBias[i]

            inVectors.append(thresholdFire(lastInput, thresh))
            k += 1

        # Calculating the input and output of the output layer
        lastInput = []
        for i in range(0, len(matrices[-1][0])):
            tally = 0
            for j in range(0, len(matrices[-1])):
                tally += inVectors[-1][j] * matrices[-1][j][i]
            lastInput.append(tally)
        outVectors.append(lastInput)

        for i in range(0, len(lastInput)):
            lastInput[i] += outBias[i]
        inVectors.append(lastInput)

        actual = thresholdFire(lastInput, thresh)
        print("Actual Output:\n" + str(actual) + "\n")

        # Calculating the error
        e = 0
        for i in range(0, len(targVector)):
            node = targVector[i] - actual[i]
            node = node ** 2
            e += node

        error = math.sqrt(e)
        print("ERROR: " + str(error))

        # Back propagation, if triggered
        if error > errorThresh:
            adjustWeight(matrices, actual, targVector, learnRate)

        cycles -= 1
        ith += 1


if __name__ == "__main__":
    main()

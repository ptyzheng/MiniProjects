import numpy as np
from collections import Counter
import random
import math

#parse documents for vectors and labels, append to list
def ParseVL(file):
    vectors = []
    labels = []

    for line in file:
        getInt = [int(x) for x in line.split()]
        vectors.append(getInt[:4003]) 
        labels.append(getInt[4003])
    return vectors, labels


def Boosting(rounds, data, labels, learners):
    boost = []
    whichLearners = []
    alphaList = []
    #if first, populate as 1/number of training samples
    for i in range(rounds):
        if i == 0:
            firstBoost = []
            for j in range(len(data)):
                firstBoost.append(1/len(data))
            boost.append(firstBoost)
        #find most accurate learner, and save to a list.
        learner, learnerNo = mostAcc(learners, data, labels, boost[i])
        whichLearners.append(learnerNo)
        error = weightedError(boost[i], learner, labels)
        #find and store alpha
        alpha = 1/2*math.log((1-error)/error)
        alphaList.append(alpha)
        boost.append(setWeight(learner, alpha, labels, boost[i]))
    print("done training Boost algorithm!")
    return whichLearners, alphaList

#find most accurate learner based on weighted error, pick randomly if tied
def mostAcc(learners, data, labels, weights):
    listOfLearners = []
    for i in range(len(learners)):
        error = weightedError(weights, learners[i], labels)
        listOfLearners.append(tuple((i, error)))
    listOfLearners.sort(key= lambda errors: errors[1])
    lowestErr = listOfLearners[0][0]
    #if tied for lowest error, randomly generate value from 0 to number of tied values
    lowestCount = list(map(lambda k : k[1], listOfLearners)).count(lowestErr)
    if lowestCount == 0:
        pickRand = 0
    else:
        pickRand = random.randint(0,lowestCount)
    #return best learner
    mostAccLearner = listOfLearners[pickRand][0]
    return learners[mostAccLearner], mostAccLearner
    
#find weighted error            
def weightedError(weights, learner, labels):
    error = 0
    for i in range(len(learner)):
        if (learner[i] != labels[i]):
            error += weights[i]*1
    return error

#get new weights for each point in vector using formula
def setWeight(learner, alpha, labels, weights):
    newWeights = weights
    for i in range(len(learner)):
        if learner[i] == labels[i]:
            newWeights[i] = math.exp(-1*alpha)*newWeights[i]
        else:
            newWeights[i] = math.exp(alpha)*newWeights[i]
    z = sum(newWeights)
    newWeights[:] = [x / z for x in newWeights]
    return newWeights

#create prediction vectors using weak learner rules
def weakLearners(data):
    posLearner = [[0 for col in range(len(data))] for row in range(len(data[0]))]
    negLearner = [[0 for col in range(len(data))] for row in range(len(data[0]))]
    for i in range(len(data)):
        for j in range(len(data[0])):
            if (data[i][j] == 1):
                posLearner[j][i] = 1
                negLearner[j][i] = -1
            else:
                posLearner[j][i] = -1
                negLearner[j][i] = 1
    joinedPred = posLearner + negLearner
    
    
    return joinedPred

#use learners + weight by alpha, return sign of final value
def predict(value, valueNo, learners, boostLearners, alphas, maximum):
    total = 0
    for i in range(maximum):
        total += alphas[i]*learners[boostLearners[i]][valueNo]
    if total >= 0:
        return 1
    else:
        return -1

rounds = [3, 4, 7, 10, 15, 20]

trainFile = open("train.txt", "r")
trainData, trainLabels = ParseVL(trainFile)
trainFile.close()
testFile = open("test.txt", "r")
testData, testLabels = ParseVL(testFile)
testFile.close()

#get results matrices for learners of training, test data
trainLearners = weakLearners(trainData)
testLearners = weakLearners(testData)

#boost using training data, run 20 times and get back alphas and learners used
boostLearners, alphas = Boosting(20, trainData, trainLabels, trainLearners)
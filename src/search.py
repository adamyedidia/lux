import random

def pront(x):
    print x

def randomBits(n):
    return [1*(random.random()>0.5) for _ in range(n)]

def randomFlip(l):
    lCopy = l[:]
    randomIndex = random.randint(0, len(l)-1)
    lCopy[randomIndex] = 1 - lCopy[randomIndex]

    return lCopy

def randomGreedyStep(l, evalFunc, maxOrMin, maxTries):
    currentVal = evalFunc(l)

    for _ in range(maxTries):

        tweakedList = randomFlip(l)
        tweakedVal = evalFunc(tweakedList)

        if maxOrMin == "min":
            if tweakedVal < currentVal:
                return tweakedList

        elif maxOrMin == "max":
            if tweakedVal > currentVal:
                return tweakedList

        else:
            pront("Error: illegal value " + maxOrMin + " for argument maxOrMin")
            raise

    return "Exceeded max tries!"


def randomGreedySearch(initList, evalFunc, maxOrMin="min", maxTries=None):
    if maxTries == None:
        maxTries = len(initList)

    currentResult = initList

    while currentResult != "Exceeded max tries!":
        oldResult = currentResult
        currentResult = randomGreedyStep(currentResult, evalFunc, maxOrMin, \
            maxTries)

    return oldResult

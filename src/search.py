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

def flipSingleEntry(l, i):
    lCopy = l[:]
    lCopy[i] = 1 - lCopy[i]    
    return lCopy

def allListsOfSizeX(x):
    if x == 0:
        return [[]]

    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]

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

def betterRandomGreedyStep(l, evalFunc, maxOrMin, helper=False,
    evalFuncOptimized=None, currentVal=None, currentHelperVal=None):
    
    if currentVal == None:
        currentVal, currentHelperVal = evalFunc(l)

    indices = range(len(l))
    random.shuffle(indices)

    for i in indices:

#        print i

        if helper:
            tweakedList = flipSingleEntry(l, i)            
            tweakedVal, helperVal = evalFuncOptimized(currentHelperVal, i)

#            print "r", tweakedList, tweakedVal, currentHelperVal

        else:    
            tweakedList = flipSingleEntry(l, i)
            tweakedVal, _ = evalFunc(tweakedList)

        if maxOrMin == "min":
            if tweakedVal < currentVal:
                if helper:
                    return tweakedList, tweakedVal, helperVal
                else:
                    return tweakedList, tweakedVal

        elif maxOrMin == "max":
            if tweakedVal > currentVal:
                if helper:
                    return tweakedList, tweakedVal, helperVal
                else:
                    return tweakedList, tweakedVal

        else:
            pront("Error: illegal value " + maxOrMin + " for argument maxOrMin")
            raise

    if helper:
        return "Local minimum!", None, None
    else:
        return "Local minimum!", None

def randomGreedySearch(initList, evalFunc, maxOrMin="min", verbose=False, \
    helper=False, evalFuncOptimized=None):

    currentList = initList
    currentVal, currentHelperVal = evalFunc(currentList)

    iterCount = 0

    while currentList != "Local minimum!":
        oldList = currentList
        if verbose:
            print "Current value:", currentVal

        if helper:
            currentList, currentVal, currentHelperVal = \
                betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                helper=helper, evalFuncOptimized=evalFuncOptimized, \
                currentVal=currentVal, currentHelperVal=currentHelperVal)

        else:
            currentList, currentVal = betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                currentVal=currentVal)

        iterCount += 1

    if verbose:
        print "Iteration count:", iterCount

    return oldList

def randomGreedySearchReturnAllSteps(initList, evalFunc, maxOrMin="min", verbose=False, \
    helper=False, evalFuncOptimized=None):

    currentList = initList
    currentVal, currentHelperVal = evalFunc(currentList)

    iterCount = 0

    listOfSteps = [initList]

    while currentList != "Local minimum!":
        oldList = currentList
        if verbose:
            print "Current value:", currentVal

        if helper:
            currentList, currentVal, currentHelperVal = \
                betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                helper=helper, evalFuncOptimized=evalFuncOptimized, \
                currentVal=currentVal, currentHelperVal=currentHelperVal)

        else:
            currentList, currentVal = betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                currentVal=currentVal)

        if currentList != "Local minimum!":
            listOfSteps.append(currentList)

        iterCount += 1

    if verbose:
        print "Iteration count:", iterCount

    return oldList, listOfSteps

def randomGreedySearchStepCount(initList, evalFunc, maxOrMin="min", verbose=False, \
    helper=False, evalFuncOptimized=None):

    currentList = initList
    currentVal, currentHelperVal = evalFunc(currentList)

    iterCount = 0

    while currentList != "Local minimum!":
        oldList = currentList
        if verbose:
            print "Current value:", currentVal

        if helper:
            currentList, currentVal, currentHelperVal = \
                betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                helper=helper, evalFuncOptimized=evalFuncOptimized, \
                currentVal=currentVal, currentHelperVal=currentHelperVal)

        else:
            currentList, currentVal = betterRandomGreedyStep(currentList, evalFunc, maxOrMin, \
                currentVal=currentVal)

        iterCount += 1

    if verbose:
        print "Iteration count:", iterCount

    return iterCount

def exhaustiveSearch(n, evalFunc, maxOrMin="min"):
    if maxOrMin == "min":
        bestValue = float("Inf")
    else:
        bestValue = float("-Inf")

    bestList = None
    allLists = allListsOfSizeX(n)

    for l in allLists:
        currentValue = evalFunc(l)
        if maxOrMin == "min":
            if currentValue < bestValue:
                bestValue = currentValue
                bestList = l
        else:
            if currentValue > bestValue:
                bestValue = currentValue
                bestList = l

    return bestList

# WARNING: This function modifies multiDimensionalArray!
# WARNING: This function depends on the mutability of lists!
# If you ever want an example why that feature is desirable, this is one.
def segmentedTrain(multiDimensionalArray, dimension, funcToTrain, \
    trainingObject, errorFunc):

    changeHappened = True
    while changeHappened:
        changeHappened = trainOneStep(multiDimensionalArray, multiDimensionalArray, dimension, \
            funcToTrain, trainingObject, errorFunc)

def trainOneStep(fullArray, subArray, dimension, funcToTrain, \
    trainingObject, errorFunc):

    if dimension == 1:
        # Base case: find best neighbor and go there

        originalFunc = funcToTrain(fullArray)
        originalError = errorFunc(originalFunc, trainingObject)

        bestError = originalError
        bestIndex = None
        vectorChanged = False

        for i, elt in enumerate(subArray):
            subArray[i] = 1-elt
            # Try the change
            neighborFunc = funcToTrain(fullArray)
            neighborError = errorFunc(fullArray, trainingObject)

            if neighborError < bestError:
                bestError = neighborError
                bestIndex = i
                vectorChanged = True

            # change it back
            subArray[i] = elt

        return vectorChanged

    elif dimension > 1:
        # The inductive case

        changeHappened = False

        # We want to trainOneStep on each child
        for i, subSubArray in enumerate(subArray):
            arrayChanged = trainOneStep(fullArray, subSubArray, dimension-1, funcToTrain, trainingObject,
                errorFunc)

        changeHappened = changeHappened or arrayChanged

        return changeHappened

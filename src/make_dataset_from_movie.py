import numpy as np

def loadMovieData():

    CUBE_FILENAME = "~/lux/movies/10ps-blur-cube.npy"
    SPHERE_FILENAME = "~/lux/movies/10ps-blur-sphere-fine.npy"

    cubeMovieArray = np.load(CUBE_FILENAME)
    sphereMovieArray = np.load(SPHERE_FILENAME)

    dataSet = []

    for grid32x32 in cubeMovieArray:
        dataSet.append((grid32x32.flatten(), np.array([1,0])))

    for grid32x32 in sphereMovieArray:
        dataSet.append((grid32x32.flatten(), np.array([0,1])))

    dataSet.shuffle()

    n = len(dataSet)
    TRAINING_SET_FRACTION = 0.5
    VALIDATION_SET_FRACTION = 0.25
    TEST_SET_FRACTION = 0.25

    assert TRAINING_SET_FRACTION + VALIDATION_SET_FRACTION + \
        TEST_SET_FRACTION == 1.

    return dataSet[:int(n*TRAINING_SET_FRACTION)], \
        dataSet[int(n*TRAINING_SET_FRACTION):int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION))], \
        dataSet[int(n*(TRAINING_SET_FRACTION+VALIDATION_SET_FRACTION)):]

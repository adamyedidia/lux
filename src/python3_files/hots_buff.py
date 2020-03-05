import matplotlib.pyplot as p
import numpy as np
from math import sqrt

maxVikingHealth = 1000
maxDamage = 1000

maxDps = 2*(sqrt(500))
maxDistance = 2*(sqrt(500))

#damage = np.random.uniform(maxDamage)

pSurvival = 0

numSamples = 10000

pSurvivalsUniform = []
pSurvivalsDoubleUniform = []
hotsBuffs = np.linspace(0.5, 2, 100)

for hotsBuff in hotsBuffs:
	print(hotsBuff)
	pSurvival = 0
	pSurvivalUniform = 0
	pSurvivalDoubleUniform = 0

	for _ in range(numSamples):
		vikingHealth = np.random.uniform(maxVikingHealth) * hotsBuff
		uniformDamage = np.random.uniform(maxDamage)
		dps = np.random.uniform(maxDps)
		distance = np.random.uniform(maxDistance)
		doubleUniformDamage = dps*distance
#		damage = np.random.exponential(maxDistance*10)


		if vikingHealth > uniformDamage:
			pSurvivalUniform += 1

		if vikingHealth > doubleUniformDamage:
			pSurvivalDoubleUniform += 1

	pSurvival /= numSamples

	pSurvivalUniform /= numSamples
	pSurvivalDoubleUniform /= numSamples


	pSurvivalsUniform.append(pSurvivalUniform)
	pSurvivalsDoubleUniform.append(pSurvivalDoubleUniform)

p.plot(hotsBuffs, pSurvivalsUniform, label="uniform damage")
p.plot(hotsBuffs, pSurvivalsDoubleUniform, label="quadratic damage")
p.xlabel("buff size")
p.ylabel("prob survival")
p.legend()
p.show()

#print(pSurvival/numSamples)



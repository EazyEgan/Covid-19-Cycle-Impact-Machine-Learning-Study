import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("jan-oct-2020-cycle-data.csv", comment='#')

GroveRoad = np.array(df.iloc[:,4])

dataLen = len(GroveRoad)
numDays = int(dataLen/24)

GroveRoadByDay = np.array_split(GroveRoad, numDays)

peaks = []
#------------get peak of each day-----------------
for i in range(0, numDays):
    peaks.append(max(GroveRoadByDay[i]))

#print(numDays, peaks)
days = list(range(0, numDays))

for i in range (0, numDays):
    #print(i, len(x), len(GroveRoadByDay[i]))
    plt.scatter(days[i], peaks[i])
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.title("Logistic Regression 2020")
plt.show()

hour = list(range(0,24))

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

for i in range (0, len(GroveRoadByDay)):
    #print(i, len(x), len(GroveRoadByDay[i]))
    plt.plot(hour, GroveRoadByDay[i])
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.title("Logistic Regression 2020")
plt.show()


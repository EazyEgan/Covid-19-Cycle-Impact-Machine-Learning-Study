import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\college4\machineLearning\Group Project\jan-oct-2020-cycle-data.csv", comment='#')

CharlivileOut = np.array(df.iloc[:,5])
CharlivileIn = np.array(df.iloc[:,6])

dataLen = len(CharlivileOut)
numDays = int(dataLen/24)

CharlivileOutByDay = np.array_split(CharlivileOut, numDays)


peaks = []
#------------get peak of each day-----------------
for i in range(0, numDays):
    peaks.append(max(CharlivileOutByDay[i]))

print(numDays, peaks)
days = list(range(0, numDays))

for i in range (0, numDays):
    #print(i, len(x), len(CharlivileOutByDay[i]))
    plt.scatter(days[i], peaks[i])
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.legend(['In'])
plt.title("Logistic Regression")
plt.show()

hour = list(range(0,24))

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

for i in range (0, len(CharlivileOutByDay)):
    #print(i, len(x), len(CharlivileOutByDay[i]))
    plt.plot(hour, CharlivileOutByDay[i])
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.legend(['In'])
plt.title("Logistic Regression")
plt.show()


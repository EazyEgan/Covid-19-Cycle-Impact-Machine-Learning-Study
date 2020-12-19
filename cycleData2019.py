import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

JAN = 0
FEB = 31
MAR = 59
APR = 90
MAY = 120
JUN = 151
JUL = 181
AUG = 212
SEP = 243
OCT = 273
NOV = 304
DEC = 334
END = 365

#---------------data cleaning notes-----------------
#2139 missing row from excel file
#NaN rows (4082-4097) - inserted 0s for the moment 

df = pd.read_csv("jan-dec-2019-cycle-data.csv", comment='#')

########################## SET UP #####################################

#identifies null columns
#print(df[df.iloc[:,1].isnull()])
#print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:,1])

dataLen = len(GroveRoad)
numDays = int(dataLen/24)
#print(dataLen, numDays, dataLen%numDays)
GroveRoadByDay = np.array_split(GroveRoad, numDays)

#getting peak of each day
peaks = []
for i in range(0, numDays):
    peaks.append(max(GroveRoadByDay[i]))
peaks = np.array(peaks).reshape(-1,1)

#getting min of each day
mins = []
for i in range(0, numDays):
    mins.append(min(GroveRoadByDay[i]))
mins = np.array(mins).reshape(-1,1)

#getting average of each day
averages = []
for i in range(0, numDays):
    averages.append(sum(GroveRoadByDay[i])/len(GroveRoadByDay[i]))
averages = np.array(averages).reshape(-1,1)

#print(numDays, peaks)
days = np.array(list(range(0, numDays))).reshape(-1,1)
hour = np.array(list(range(0,24))).reshape(-1,1)

############################# CHARTING ##################################

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

def plotDayPeaks(daysSub, peaksSub, title):
    plt.scatter(daysSub, peaksSub)
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Day Peaks'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()
    
def plotDayMins(daysSub, minsSub, title):
    plt.scatter(daysSub, minsSub)
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Day Mins'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()

def plotDayAverages(daysSub, averagesSub, title):
    plt.scatter(daysSub, averagesSub)
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Day Averages'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()

def plotPeriodVolumeInDay(start, end, title):
    for i in range (start, end):
        #print(i, len(hour), len(GroveRoadByDay[i]))
        plt.plot(hour, GroveRoadByDay[i])
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()

################################## REGRESSION #################################
'''
C = 1
peaks = np.array(peaks).reshape(-1,1)
days = np.array(days).reshape(-1,1)

#print(peaks, days)

a = 1/(2*C)
model = linear_model.Lasso(alpha=a)
model.fit(days, peaks)
yPred = model.predict(days)

plt.scatter(days, yPred)
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.legend(['Total'])
plt.title("lasso regression 2019")
plt.show()

############################### EVALUATION #####################################


mean_error=[]; std_error=[]
f = 5
C_range = [1]

for C in C_range:
    a = 1/(2*C)
    model = linear_model.Lasso(alpha=a)
    temp = []
    
    kf = KFold(n_splits=f)
    for train, test in kf.split(days):
        model.fit(days[train], peaks[train])
        print(days[train], peaks[train], days[test])
        ypred = model.predict(days[test])
        print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(peaks[test], ypred))
        temp.append(mean_squared_error(peaks[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.errorbar(C_range, mean_error, yerr=std_error)
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.title('Lasso Regression 10-fold')
plt.show()

'''

plotDayPeaks(days[JAN:FEB], peaks[JAN:FEB], "January")
plotDayAverages(days[MAR:APR], averages[MAR:APR], "March")


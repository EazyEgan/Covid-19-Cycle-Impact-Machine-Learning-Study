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
MAR = 60
APR = 91
MAY = 121
JUN = 152
JUL = 182
AUG = 213
SEP = 244
OCT = 274
NOV = 305


#---------------data cleaning notes-----------------
#2139 missing row from excel file
#NaN rows (3026-3039) filled in with zeros

df = pd.read_csv("jan-oct-2020-cycle-data.csv", comment='#')

########################## SET UP #####################################

#identifies null columns
#print(df[df.iloc[:,1].isnull()])
#print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:,4])

dataLen = len(GroveRoad)
numDays = int(dataLen/24)

#print(numDays, peaks)
days = np.array(list(range(0, numDays))).reshape(-1,1)
hours = np.array(list(range(0,24))).reshape(-1,1)

#print(dataLen, numDays, dataLen%numDays)
hoursGroupedByDay = np.array_split(GroveRoad, numDays)

print(dataLen, numDays)

#print("grove road = ", groveRoadByDay[1], "hoursGrouped = " ,hoursGroupedByDay[1])

#getting peak of each day
peaks = []
for i in range(0, numDays):
    peaks.append(max(hoursGroupedByDay[i]))
peaks = np.array(peaks).reshape(-1,1)

#getting min of each day
mins = []
for i in range(0, numDays):
    mins.append(min(hoursGroupedByDay[i]))
mins = np.array(mins).reshape(-1,1)

#getting average of each day
averages = []
for i in range(0, numDays):
    averages.append(sum(hoursGroupedByDay[i])/len(hoursGroupedByDay[i]))
averages = np.array(averages).reshape(-1,1)

#reshaping hoursGroupedByDay after above functions to avoid nesting reshaping
for i in range (0, len(hours)):
    hoursGroupedByDay[i] = hoursGroupedByDay[i].reshape(-1,1)


############################# PLOTTING ##################################

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

def plotHourlyTraffic(hoursGroupedSub, title):
    for i in range (0, len(hours)):
        plt.plot(hours, hoursGroupedSub[i])
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()

################################## REGRESSION #################################

C = 0.0001

poly = PolynomialFeatures(6)
polyDays = poly.fit_transform(days)

print(polyDays)

a = 1/(2*C)
model = linear_model.Lasso(alpha=a)
model.fit(polyDays, mins)
yPred = model.predict(polyDays)

#print("days", days, "peaks", peaks, "yPred", yPred)

plt.scatter(days, mins, color="blue")
plt.plot(days, yPred, color="red")
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.legend(['Total'])
plt.title("lasso regression 2019")
plt.show()

############################### EVALUATION #####################################

'''
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


#plotDay*something* template(days[day_range], peaks[day_range], title)
plotDayPeaks(days[JAN:FEB], peaks[JAN:FEB], "January")
plotDayAverages(days[MAR:APR], averages[MAR:APR], "March")

#plotHourlyTraffic(hoursGroupedByDay[day_range], title)
plotHourlyTraffic(hoursGroupedByDay[NOV:DEC], "November")
'''
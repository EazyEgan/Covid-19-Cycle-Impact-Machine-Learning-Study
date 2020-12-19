import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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
#print(df.iloc[2130:2140,1])

GroveRoad = np.array(df.iloc[:,1])

dataLen = len(GroveRoad)
numDays = int(dataLen/24)
#print(dataLen, numDays, dataLen%numDays)
GroveRoadByDay = np.array_split(GroveRoad, numDays)

#getting peak of each day
peaks = []
for i in range(0, numDays):
    peaks.append(max(GroveRoadByDay[i]))

#getting min of each day
mins = []
for i in range(0, numDays):
    mins.append(min(GroveRoadByDay[i]))

#getting average of each day
averages = []
for i in range(0, numDays):
    averages.append(sum(GroveRoadByDay[i])/len(GroveRoadByDay[i]))

#print(numDays, peaks)
days = list(range(0, numDays))
hour = list(range(0,24))

############################# CHARTING ##################################

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

def plotDayPeaksInPeriod(start, end, title):
    for i in range (start, end):
        plt.scatter(days[i], peaks[i])
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()
    
def plotDayMinsInPeriod(start, end, title):
    for i in range (start, end):
        plt.scatter(days[i], mins[i])
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()

def plotDayAveragesInPeriod(start, end, title):
    for i in range (start, end):
        plt.scatter(days[i], averages[i])
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
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

C = 1
peaks = np.array(peaks).reshape(1, -1)
days = np.array(days).reshape(1, -1)

#print(peaks, days)

a = 1/(2*C)
clf = linear_model.Lasso(alpha=a)
clf.fit(days, peaks)
cyclistPred = clf.predict(days)

plt.scatter(days, cyclistPred)
plt.xlabel("Hour"); plt.ylabel("Cyclists")
plt.legend(['Total'])
plt.title("lasso regression 2019")
plt.show()

############################### EVALUATION #####################################




'''
#plot day peaks by year and months
plotDayPeaksInPeriod(0, numDays, "Full Year")
plotDayPeaksInPeriod(JAN, FEB, "January")
plotDayPeaksInPeriod(FEB, MAR, "Feburary")
plotDayPeaksInPeriod(MAR, APR, "March")
plotDayPeaksInPeriod(APR, MAY, "April")
plotDayPeaksInPeriod(MAY, JUN, "May")
plotDayPeaksInPeriod(JUN, JUL, "June")
plotDayPeaksInPeriod(JUL, AUG, "July")
plotDayPeaksInPeriod(AUG, SEP, "August")
plotDayPeaksInPeriod(SEP, OCT, "September")
plotDayPeaksInPeriod(OCT, NOV, "October")
plotDayPeaksInPeriod(NOV, DEC, "November")
plotDayPeaksInPeriod(DEC, END, "December")

#plot day peaks by year and months
plotDayMinsInPeriod(0, numDays, "Full Year")
plotDayMinsInPeriod(JAN, FEB, "January")
plotDayMinsInPeriod(FEB, MAR, "Feburary")
plotDayMinsInPeriod(MAR, APR, "March")
plotDayMinsInPeriod(APR, MAY, "April")
plotDayMinsInPeriod(MAY, JUN, "May")
plotDayMinsInPeriod(JUN, JUL, "June")
plotDayMinsInPeriod(JUL, AUG, "July")
plotDayMinsInPeriod(AUG, SEP, "August")
plotDayMinsInPeriod(SEP, OCT, "September")
plotDayMinsInPeriod(OCT, NOV, "October")
plotDayMinsInPeriod(NOV, DEC, "November")
plotDayMinsInPeriod(DEC, END, "December")

#plot day averages by year and months
plotDayAveragesInPeriod(0, numDays, "Full Year")
plotDayAveragesInPeriod(JAN, FEB, "January")
plotDayAveragesInPeriod(FEB, MAR, "Feburary")
plotDayAveragesInPeriod(MAR, APR, "March")
plotDayAveragesInPeriod(APR, MAY, "April")
plotDayAveragesInPeriod(MAY, JUN, "May")
plotDayAveragesInPeriod(JUN, JUL, "June")
plotDayAveragesInPeriod(JUL, AUG, "July")
plotDayAveragesInPeriod(AUG, SEP, "August")
plotDayAveragesInPeriod(SEP, OCT, "September")
plotDayAveragesInPeriod(OCT, NOV, "October")
plotDayAveragesInPeriod(NOV, DEC, "November")
plotDayAveragesInPeriod(DEC, END, "December")

#plot period volume per day
plotPeriodVolumeInDay(0, numDays, "Full Year")
plotPeriodVolumeInDay(JAN, FEB, "January")
plotPeriodVolumeInDay(FEB, MAR, "Feburary")
plotPeriodVolumeInDay(MAR, APR, "March")
plotPeriodVolumeInDay(APR, MAY, "April")
plotPeriodVolumeInDay(MAY, JUN, "May")
plotPeriodVolumeInDay(JUN, JUL, "June")
plotPeriodVolumeInDay(JUL, AUG, "July")
plotPeriodVolumeInDay(AUG, SEP, "August")
plotPeriodVolumeInDay(SEP, OCT, "September")
plotPeriodVolumeInDay(OCT, NOV, "October")
plotPeriodVolumeInDay(NOV, DEC, "November")
plotPeriodVolumeInDay(DEC, END, "December")
'''

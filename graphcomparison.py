import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsClassifier

# start of each month
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
END = 305

#
DAY_OFFSET = 3
MON = 1
TUE = 2
WED = 3
THUR = 4
FRI = 5
SAT = 6
SUN = 7



################################ TIMELINE FUNCTIONS #####################################

def splitWeekDays2(daysList):
    weekDays = []
    weekEnds = []

    for i in range(0, numDays):
        day = i + DAY_OFFSET
        if (day % 7 > 0 and day % 7 <= 5):
            weekDays.append(daysList[i])  # monday - friday
        else:
            weekEnds.append(daysList[i])  # saturday, sunday

    weekDays = np.array(weekDays).reshape(-1, 1)
    weekEnds = np.array(weekEnds).reshape(-1, 1)
    return weekDays, weekEnds

def getWeekDaysAndWeekEndsFromList2(daysList):
    weekDays = []
    weekEnds = []
    for i in range(0, numDays):
        day = i + DAY_OFFSET
        if (day % 7 > 0 and day % 7 <= 5):
            weekDays.append(daysList[i])  # monday - friday
            weekEnds.append(np.nan)
        else:
            weekEnds.append(daysList[i])  # saturday, sunday
            weekDays.append(np.nan)

    weekDays = np.array(weekDays).reshape(-1, 1)
    weekEnds = np.array(weekEnds).reshape(-1, 1)
    return weekDays, weekEnds


def getWeekDayCountBetweenMonths2(startMonth, endMonth):
    if (startMonth == 1):
        startIndex = 0
        endIndex = np.busday_count('2020-01', f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}')
    else:
        startIndex = np.busday_count(f'2020-01', f'2020-0{startMonth}' if (startMonth < 10) else f'2020-0{startMonth}')
        endIndex = startIndex + np.busday_count(f'2020-0{startMonth}' if (startMonth < 10) else f'2020-{startMonth}'
                                                , f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}')
    return startIndex, endIndex



# start of each month
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

#
DAY_OFFSET = 2
MON = 1
TUE = 2
WED = 3
THUR = 4
FRI = 5
SAT = 6
SUN = 7

# 2019#
# ---------------data cleaning notes-----------------
# 2139 missing row from excel file
# NaN rows (4082-4097) filled in with zeros

df = pd.read_csv("jan-dec-2019-cycle-data.csv", comment='#')

# identifies null columns
# print(df[df.iloc[:,1].isnull()])
# print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:, 1])

dataLen = len(GroveRoad)
numDays = int(dataLen / 24)
days = np.array(list(range(0, numDays))).reshape(-1, 1)
hours = np.array(list(range(0, 24))).reshape(-1, 1)
hoursGroupedByDay = np.array_split(GroveRoad, numDays)

peaks = []
mins = []
averages = []

################################ TIMELINE FUNCTIONS #####################################

def splitWeekDays(daysList):
    weekDays = []
    weekEnds = []

    for i in range(0, numDays):
        day = i + DAY_OFFSET
        if (day % 7 > 0 and day % 7 <= 5):
            weekDays.append(daysList[i])  # monday - friday
        else:
            weekEnds.append(daysList[i])  # saturday, sunday

    weekDays = np.array(weekDays).reshape(-1, 1)
    weekEnds = np.array(weekEnds).reshape(-1, 1)
    return weekDays, weekEnds


def getWeekDaysAndWeekEndsFromList(daysList):
    weekDays = []
    weekEnds = []
    for i in range(0, numDays):
        day = i + DAY_OFFSET
        if (day % 7 > 0 and day % 7 <= 5):
            weekDays.append(daysList[i])  # monday - friday
            weekEnds.append(np.nan)
        else:
            weekEnds.append(daysList[i])  # saturday, sunday
            weekDays.append(np.nan)

    weekDays = np.array(weekDays).reshape(-1, 1)
    weekEnds = np.array(weekEnds).reshape(-1, 1)
    return weekDays, weekEnds


def getWeekDayCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1):
        startIndex = 0
        endIndex = np.busday_count('2019-01', f'2019-0{endMonth}' if (endMonth < 10) else f'2019-{endMonth}')
    else:
        startIndex = np.busday_count(f'2019-01', f'2019-0{startMonth}' if (startMonth < 10) else f'2019-0{startMonth}')
        endIndex = startIndex + np.busday_count(f'2019-0{startMonth}' if (startMonth < 10) else f'2019-{startMonth}'
                                                , f'2019-0{endMonth}' if (endMonth < 10) else f'2019-{endMonth}')
    return startIndex, endIndex


def gaussian_kernel(distances):
    weights = np.exp(-gamma * (distances ** 2))
    return weights / np.sum(weights)


def kNN(Xtrain, ytrain, k_list):
    Xtest = Xtrain  # xtest
    legendlist = []

    ###### WEEKEDAYS ########
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    #### 2019 ####
    plt.title("2019 kNN Regression")
    plt.scatter(Xtrain, ytrain, marker=".")

    for k in k_list:
        model = KNeighborsRegressor(n_neighbors=k, weights='uniform').fit(Xtrain,
                                                                          ytrain)  # ANything on or above is weekday
        ypred = model.predict(Xtest)
        plt.plot(Xtest, ypred)
        legendlist.append("k= " + str(k))

    plt.xlabel("Days")
    plt.ylabel("Averages")
    legendlist.append("train")
    plt.legend(loc='upper right')
    plt.legend(legendlist, prop={"size": 10})
    plt.show()

    #### 2020 ####
    """
    X2, y2 = days2, averages2

    plt.title("2020 kNN Regression")
    plt.scatter(X2, y2, marker=".")

    for k in k_list:
        model = KNeighborsRegressor(n_neighbors=k, weights='uniform').fit(Xtrain, ytrain)  # ANything on or above is weekday
        ypred = model.predict(Xtest)
        plt.plot(Xtest, ypred, color='darkorange')
        legendlist.append("k= ", str(k))

    plt.xlabel("Days")
    plt.ylabel("Averages")
    legendlist.append("train")
    plt.legend(loc='upper right')
    plt.legend(legendlist, prop={"size": 10})
    plt.show()
    """


gamArray = [0, 0.0001, 0.001, 0.01, 0.1]  # ,5,10,25]


def kernelizedKNN(Xtrain, ytrain):
    global gamma
    Xtest = Xtrain
    legendlist = []
    plt.title("2019 kNN Kernelized Regression")
    plt.scatter(Xtrain, ytrain, marker=".")
    for gam in gamArray:
        gamma = gam
        model = KNeighborsRegressor(n_neighbors=len(Xtrain), weights=gaussian_kernel).fit(Xtrain,
                                                                                          ytrain)  # ANything on or above is weekday
        ypred = model.predict(Xtest)
        plt.plot(Xtest, ypred)
        legendlist.append("gamma = " + str(gam))

    plt.xlabel("Days")
    plt.ylabel("Averages")
    legendlist.append("train")
    plt.legend(loc='upper right')
    plt.legend(legendlist, prop={"size": 10})
    plt.show()





# 2019#
# ---------------data cleaning notes-----------------
# 2139 missing row from excel file
# NaN rows (4082-4097) filled in with zeros

df = pd.read_csv("jan-dec-2019-cycle-data.csv", comment='#')

# identifies null columns
# print(df[df.iloc[:,1].isnull()])
# print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:, 1])

dataLen = len(GroveRoad)
numDays = int(dataLen / 24)
days = np.array(list(range(0, numDays))).reshape(-1, 1)
hours = np.array(list(range(0, 24))).reshape(-1, 1)
hoursGroupedByDay = np.array_split(GroveRoad, numDays)

peaks = []
mins = []
averages = []

for i in range(0, numDays):
    peaks.append(max(hoursGroupedByDay[i]))
    mins.append(min(hoursGroupedByDay[i]))
    averages.append(sum(hoursGroupedByDay[i]) / len(hoursGroupedByDay[i]))

peaks = np.array(peaks).reshape(-1, 1)
mins = np.array(mins).reshape(-1, 1)
averages = np.array(averages).reshape(-1, 1)

# reshaping hoursGroupedByDay after above functions to avoid nesting reshaping
for i in range(0, len(hours)):
    hoursGroupedByDay[i] = hoursGroupedByDay[i].reshape(-1, 1)


#------------- COMPARISON SET UP ---------------#
# 2020 #
df2 = pd.read_csv("jan-oct-2020-cycle-data.csv", comment='#')

GroveRoad2 = np.array(df2.iloc[:, 4])

dataLen2 = len(GroveRoad2)
numDays2 = int(dataLen2 / 24)
days2 = np.array(list(range(0, numDays2))).reshape(-1, 1)
hours2 = np.array(list(range(0, 24))).reshape(-1, 1)
hoursGroupedByDay2 = np.array_split(GroveRoad2, numDays2)

# getting peak of each day
peaks2 = []
for i in range(0, numDays2):
    peaks2.append(max(hoursGroupedByDay2[i]))
peaks2 = np.array(peaks2).reshape(-1, 1)

averages2 = []

for i in range(0, numDays2):
    averages2.append(sum(hoursGroupedByDay2[i]) / len(hoursGroupedByDay2[i]))


averages2 = np.array(averages2).reshape(-1, 1)
################################ main sequence ###################################
'''
#plotting dataset - all days v weekdays v weekends
weekDayPeaks, weekEndPeaks = getWeekDaysAndWeekEndsFromList(peaks)
plotDayData(days[JAN:END], peaks[JAN:END], "", "Peaks", "All Days")
plotDayData(days[JAN:END], weekDayPeaks[JAN:END], "", "Peaks", "Week Days" )
plotDayData(days[JAN:END], weekEndPeaks[JAN:END], "", "Peaks", "Weekends")
'''

# overwriting above, beginning regression
weekDayPeaks2, weekEndPeaks2 = splitWeekDays2(peaks2)
weekDayAverages2, weekEndAverages2 = splitWeekDays2(averages2)
weekDayMins, weekEndMins = splitWeekDays2(mins2)

# week days/ends as long as any of the arbitrarily picked averages/peaks/min
numWeekDays2 = len(weekDayAverages2)
numWeekEnds2 = len(weekEndAverages2)

weekDays = np.array(list(range(0, numWeekDays))).reshape(-1, 1)
weekEnds = np.array(list(range(0, numWeekEnds))).reshape(-1, 1)



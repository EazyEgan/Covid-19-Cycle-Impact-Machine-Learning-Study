import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#start of each month
JAN = 0; FEB = 31; MAR = 59; APR = 90; MAY = 120; JUN = 151; JUL = 181
AUG = 212; SEP = 243; OCT = 273; NOV = 304; DEC = 334; END = 365

#
DAY_OFFSET = 2; MON = 1; TUE = 2; WED = 3; THUR = 4; FRI = 5; SAT = 6; SUN = 7

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

#print(numDays, peaks)
days = np.array(list(range(0, numDays))).reshape(-1,1)
hours = np.array(list(range(0,24))).reshape(-1,1)

#print(dataLen, numDays, dataLen%numDays)
hoursGroupedByDay = np.array_split(GroveRoad, numDays)

def splitWeekDays(daysList):
    weekDays = []
    weekEnds = []

    for i in range (0, numDays):
        day = i + DAY_OFFSET
        if (day % 7 > 0 and day % 7 <=5):
            weekDays.append(daysList[i]) #monday - friday
        else:
            weekEnds.append(daysList[i]) #saturday, sunday

    weekDays = np.array(weekDays).reshape(-1,1)
    weekEnds = np.array(weekEnds).reshape(-1,1)
    return weekDays, weekEnds

def getWeekDayCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1): 
        startIndex = 0
        endIndex = np.busday_count('2019-01', f'2019-0{endMonth}' if (endMonth < 10)  else f'2019-{endMonth}') 
    else:
        startIndex =  np.busday_count(f'2019-01', f'2019-0{startMonth}' if(startMonth < 10) else f'2019-0{startMonth}')
        endIndex = startIndex + np.busday_count(f'2019-0{startMonth}' if(startMonth < 10) else f'2019-{startMonth}'
                                                , f'2019-0{endMonth}' if(endMonth < 10) else f'2019-{endMonth}') 
    return startIndex, endIndex

def getWeekEndCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1): 
        startIndex = 0
        endIndex = np.busday_count('2019-01', f'2019-0{endMonth}' if (endMonth < 10)  else f'2019-{endMonth}', weekmask="Sat Sun") 
    else:
        startIndex =  np.busday_count(f'2019-01', f'2019-0{startMonth}' if(startMonth < 10) else f'2019-0{startMonth}', weekmask="Sat Sun")
        endIndex = startIndex + np.busday_count(f'2019-0{startMonth}' if(startMonth < 10) else f'2019-{startMonth}'
                                                , f'2019-0{endMonth}' if(endMonth < 10) else f'2019-{endMonth}', weekmask="Sat Sun") 
    return startIndex, endIndex

#getting peak of each day
peaks = []
for i in range(0, numDays):
    peaks.append(max(hoursGroupedByDay[i]))
peaks = np.array(peaks).reshape(-1,1)
weekDayPeaks, weekEndPeaks = splitWeekDays(peaks)

#getting min of each day
mins = []
for i in range(0, numDays):
    mins.append(min(hoursGroupedByDay[i]))
mins = np.array(mins).reshape(-1,1)
weekDayMins, weekEndMins = splitWeekDays(mins)

#getting average of each day
averages = []
for i in range(0, numDays):
    averages.append(sum(hoursGroupedByDay[i])/len(hoursGroupedByDay[i]))
averages = np.array(averages).reshape(-1,1)
weekDayAverages, weekEndAverages = splitWeekDays(averages)

#reshaping hoursGroupedByDay after above functions to avoid nesting reshaping
for i in range (0, len(hours)):
    hoursGroupedByDay[i] = hoursGroupedByDay[i].reshape(-1,1)

#week days/ends as long as any of the arbitrarily picked averages/peaks/min
numWeekDays = len(weekDayAverages)
numWeekEnds = len(weekEndAverages)

weekDays = np.array(list(range(0, numWeekDays))).reshape(-1,1)
weekEnds = np.array(list(range(0, numWeekEnds))).reshape(-1,1)

############################# PLOTTING ##################################

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

def plotDayData(dataType, daysSubArray, dataSubArray, title):
    plt.scatter(daysSubArray, dataSubArray)
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend([f'Day {dataType}'])
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


def lassoRegression(y):
    C = 0.001

    poly = PolynomialFeatures(4)
    polyDays = poly.fit_transform(days)

    #print(polyDays)

    a = 1/(2*C)
    model = linear_model.Lasso(alpha=a)
    model.fit(polyDays, y)
    yPred = model.predict(polyDays)

    #print("days", days, "peaks", peaks, "yPred", yPred)

    plt.scatter(days, y, color="blue")
    plt.plot(days, yPred, color="red")
    plt.xlabel("Hour"); plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title("lasso regression 2019")
    plt.show()

#lassoRegression(averages)
#lassoRegression(mins)

############################### EVALUATION #####################################

'''
mean_error=[]; std_error=[]
f = 5
C_range = [0.001, 0.01, 1, 100, 1000]


for C in C_range:
    a = 1/(2*C)
    model = linear_model.Lasso(alpha=a)
    temp = []
    
    kf = KFold(n_splits=f)
    for train, test in kf.split(weekDays):
        model.fit(weekDays[train], weekDayPeaks[train])
        #print(days[train], peaks[train], days[test])
        ypred = model.predict(weekDays[test])
        #print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(peaks[test], ypred))
        temp.append(mean_squared_error(peaks[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plt.errorbar(C_range, mean_error, yerr=std_error)
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.title('Lasso Regression 5-fold')
plt.show()
'''

#plotDayData((string) dataType, days[day range], peaks[day range], (string) time period)
plotDayData("Peaks", days[JAN:FEB], peaks[JAN:FEB], "January")
plotDayData("Averages", days[MAR:APR], averages[MAR:APR], "March")

start, end = getWeekDayCountBetweenMonths(1, 12)
plotDayData("Peaks", days[start:end], weekDayPeaks[start:end], "Full year")
print(start, end, weekDayPeaks[start:end])

start, end = getWeekEndCountBetweenMonths(1, 12)
plotDayData("Peaks", days[start:end], weekEndPeaks[start:end], "full year")
print(start, end, weekEndPeaks[start:end])

#plotHourlyTraffic(hoursGroupedByDay[day_range], title)
plotHourlyTraffic(hoursGroupedByDay[NOV:DEC], "November")



############################### KNN CLASSIFIER #################################

#######CURRENTLY OVERFIT - HAVE TO CHANGE NEIGHBOURS


start, end = getWeekEndCountBetweenMonths(1, 12)
Xtrain = days[start:end]

ytrain = weekEndPeaks[start:end]

#WEEKENDS AND WEEKDAYS - WILL MAKE INTO FUCNTION TO PASS WEEKENDS OR WEEKDAYS
for i in [1,2]:


    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5,weights='uniform').fit(Xtrain, ytrain)
    Xtest = Xtrain

    ypred = model.predict(Xtrain)
    import matplotlib.pyplot as plt
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(Xtrain, ytrain, color='red', marker='+')
    plt.plot(Xtest, ypred, color='green')
    plt.xlabel("input x"); plt.ylabel("output y")
    plt.legend(["predict","train"])
    plt.show()

    ############################ KNN REGRESSION ####################################
    import numpy as np
    m = 25

    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=3,weights='uniform').fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    import matplotlib.pyplot as plt
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(Xtrain, ytrain, color='red', marker='+')
    plt.plot(Xtest, ypred, color='green')
    plt.xlabel("input x"); plt.ylabel("output y"); plt.legend(["predict","train"])
    plt.show()

    def gaussian_kernel100(distances):
        weights = np.exp(-100*(distances**2))
        return weights/np.sum(weights)

    def gaussian_kernel1000(distances):
        weights = np.exp(-1000*(distances**2))
        return weights/np.sum(weights)
    def gaussian_kernel10000(distances):
        weights = np.exp(-10000*(distances**2))
        return weights/np.sum(weights)

    model2 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel100).fit(Xtrain, ytrain)
    ypred2 = model2.predict(Xtest)

    model3 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel1000).fit(Xtrain, ytrain)
    ypred3 = model3.predict(Xtest)

    model4 = KNeighborsRegressor(n_neighbors=3,weights=gaussian_kernel10000).fit(Xtrain, ytrain)
    ypred4 = model4.predict(Xtest)

    plt.scatter(Xtrain, ytrain, color='red', marker='+')
    plt.plot(Xtest, ypred2, color='blue')
    plt.plot(Xtest, ypred3, color='orange')
    plt.plot(Xtest, ypred4, color='green')
    plt.xlabel("input x"); plt.ylabel("output y")
    plt.legend(["k=7,sigma=100","k=7,sigma=1000","k=7,sigma=10000","train"])
    plt.show()
    start, end = getWeekDayCountBetweenMonths(1, 12)
    Xtrain = days[start:end]
    ytrain = weekDayPeaks[start:end]
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

#start of each month
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

#
DAY_OFFSET = 2; MON = 1; TUE = 2; WED = 3; THUR = 4; FRI = 5; SAT = 6; SUN = 7

#---------------data cleaning notes-----------------
#2139 missing row from excel file
#NaN rows (3026-3039) filled in with zeros

df = pd.read_csv("jan-oct-2020-cycle-data.csv", comment='#')

################################### SET UP ####################################

#identifies null columns
#print(df[df.iloc[:,1].isnull()])
#print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:,1])

dataLen = len(GroveRoad)
numDays = int(dataLen/24)

days = np.array(list(range(0, numDays))).reshape(-1,1)
hours = np.array(list(range(0,24))).reshape(-1,1)

hoursGroupedByDay = np.array_split(GroveRoad, numDays)

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


################################ TIMELINE FUNCTIONS #####################################

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
        endIndex = np.busday_count('2020-01', f'2020-0{endMonth}' if (endMonth < 10)  else f'2020-{endMonth}') 
    else:
        startIndex =  np.busday_count(f'2020-01', f'2020-0{startMonth}' if(startMonth < 10) else f'2020-0{startMonth}')
        endIndex = startIndex + np.busday_count(f'2020-0{startMonth}' if(startMonth < 10) else f'2020-{startMonth}'
                                                , f'2020-0{endMonth}' if(endMonth < 10) else f'2020-{endMonth}') 
    return startIndex, endIndex

def getWeekEndCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1): 
        startIndex = 0
        endIndex = np.busday_count('2020-01', f'2020-0{endMonth}' if (endMonth < 10)  else f'2020-{endMonth}', weekmask="Sat Sun") 
    else:
        startIndex =  np.busday_count(f'2020-01', f'2020-0{startMonth}' if(startMonth < 10) else f'2020-0{startMonth}', weekmask="Sat Sun")
        endIndex = startIndex + np.busday_count(f'2020-0{startMonth}' if(startMonth < 10) else f'2020-{startMonth}'
                                                , f'2020-0{endMonth}' if(endMonth < 10) else f'2020-{endMonth}', weekmask="Sat Sun") 
    return startIndex, endIndex

################################ NORMALIZING ##################################

def normalize(X, y):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    scaler.fit(y)
    y = scaler.transform(y)
    return X, y

################################# PLOTTING #####################################

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

################################## LASSO REGRESSION #################################


def lassoRegression(X, y, c, poly):
    C = c

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    a = 1/(2*C)
    model = linear_model.Lasso(alpha=a)
    model.fit(polyX, y)
    yPred = model.predict(polyX)

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days"); plt.ylabel("Cyclists")
    plt.title("lasso Regression 2019")
    plt.show()

################################## RIDGE REGRESSION #################################

def ridgeRegression(X, y, c, poly):
    C = c

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    a = 1/(2*C)
    model = Ridge(alpha=a)
    model.fit(polyX, y)
    yPred = model.predict(polyX)

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days"); plt.ylabel("Cyclists")
    plt.title("Ridge Regression 2019")
    plt.show()


############################ KNN REGRESSION ####################################
#CURRENTLY OVERFIT - HAVE TO CHANGE NEIGHBOURS
def kNN(Xtrain, ytrain):

    '''
    ############################### KNN CLASSIFIER #################################
    if(dayType.lower() == "weekend"):
        start, end = getWeekEndCountBetweenMonths(startMonth, endMonth)
        Xtrain = days[start:end]

        ytrain = weekEndPeaks[start:end]

    elif (dayType.lower=="weekday"):

        start, end = getWeekDayCountBetweenMonths(1, 12)
        Xtrain = days[start:end]
        ytrain = weekDayPeaks[start:end]
        
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
    '''

    Xtest = Xtrain

    model = KNeighborsRegressor(n_neighbors=3,weights='uniform').fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

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

############################### DUMMY REGRESSION ##############################
def dummy_regressor(X, y):
    model = DummyRegressor(strategy="constant", constant=0.2)
    model.fit(X, y)
    yPred = model.predict(X)
    print(model.score(X, y))

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days"); plt.ylabel("Cyclists")
    plt.title("Dummy Regression 2019")
    plt.show()

############################### EVALUATION #####################################

def cross_validation(X, y, poly, model):
    mean_error=[]; std_error=[]
    f = 5
    C_range = [0.001, 0.01, 0.1, 1, 10]

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in C_range:
        a = 1/(2*C)
        
        if(model == "lasso"): model = linear_model.Lasso(alpha=a)
        elif(model == "ridge"): model = Ridge(alpha=a)
        elif(model == "kNN"): model = KNeighborsRegressor(n_neighbors=3,weights='uniform')
        else: model = DummyRegressor(strategy="constant", constant=0.5)
        
        temp = []
        
        kf = KFold(n_splits=f)
        for train, test in kf.split(polyX):
            model.fit(polyX[train], y[train])
            ypred = model.predict(polyX[test])
            #print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(polyX[test], ypred))
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        scores = cross_val_score(model, polyX, y, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        print("Accuracy: %0.2f (+/− %0.2f)" % (scores.mean(), scores.std()))

    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel('C'); plt.ylabel('Mean square error')
    plt.title('Lasso Regression 5-fold')
    plt.show()


############################### EXECUTION ########################################

weekDayPeaks, weekEndPeaks = splitWeekDays(peaks)
weekDayMins, weekEndMins = splitWeekDays(mins)
weekDayAverages, weekEndAverages = splitWeekDays(averages)

#week days/ends as long as any of the arbitrarily picked averages/peaks/min
numWeekDays = len(weekDayAverages)
numWeekEnds = len(weekEndAverages)

weekDays = np.array(list(range(0, numWeekDays))).reshape(-1,1)
weekEnds = np.array(list(range(0, numWeekEnds))).reshape(-1,1)


##plotDayData((string) dataType, days[day range], peaks[day range], (string) time period)
#plotDayData("Peaks", days[JAN:FEB], peaks[JAN:FEB], "January")

#start, end = getWeekDayCountBetweenMonths(1, 12)
#plotDayData("Peaks", weekDays[start:end], weekDayPeaks[start:end], "Full year")

##plotHourlyTraffic(hoursGroupedByDay[day_range], title)
#plotHourlyTraffic(hoursGroupedByDay[NOV:DEC], "November")


start, end = getWeekDayCountBetweenMonths(1,10)
X = weekDays[start:end]
y = weekDayPeaks[start:end]
X, y = normalize(X,y)

lassoRegression(X, y, 10000, 4)
ridgeRegression(X, y, 10000, 4)
kNN(X, y)
dummy_regressor(X, y)
cross_validation(X, y, 2, "lasso")






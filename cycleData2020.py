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

DAY_OFFSET = 2

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
        endIndex = np.busday_count('2020-01', f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}')
    else:
        startIndex = np.busday_count(f'2020-01', f'2020-0{startMonth}' if (startMonth < 10) else f'2020-0{startMonth}')
        endIndex = startIndex + np.busday_count(f'2020-0{startMonth}' if (startMonth < 10) else f'2020-{startMonth}'
                                                , f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}')
    return startIndex, endIndex


def getWeekEndCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1):
        startIndex = 0
        endIndex = np.busday_count('2020-01', f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}',
                                   weekmask="Sat Sun")
    else:
        startIndex = np.busday_count(f'2020-01', f'2020-0{startMonth}' if (startMonth < 10) else f'2020-0{startMonth}',
                                     weekmask="Sat Sun")
        endIndex = startIndex + np.busday_count(f'2020-0{startMonth}' if (startMonth < 10) else f'2020-{startMonth}'
                                                , f'2020-0{endMonth}' if (endMonth < 10) else f'2020-{endMonth}',
                                                weekmask="Sat Sun")
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


def plotDayData(X, y, title, dataType, xlabel):
    plt.scatter(X, y)
    plt.xlabel(xlabel)
    plt.ylabel("Cyclists")
    plt.legend([f'Daily {dataType}'], loc=1)
    plt.title(f"{title} Cycle Volume 2020")
    plt.show()


def plotHourlyTraffic(y, title):
    for i in range(0, len(hours)):
        plt.plot(hours, y[i])
    plt.xlabel("Hour")
    plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2020")
    plt.show()


################################## LASSO REGRESSION #################################


def lassoRegressionC(X, y, Xtest, c_list, P, xlabel):

    poly = PolynomialFeatures(P)
    polyX = poly.fit_transform(X)

    for C in c_list:

        a = 1 / (2 * C)
        model = linear_model.Lasso(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(poly.fit_transform(Xtest)) #train on extend x-axis

        plt.plot(Xtest, yPred, label=f"C:{C}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.legend(loc="lower left", prop={'size': 14})
    plt.ylabel("Cyclists")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Lasso Regression 2020 - P: {P}")
    plt.show()

def lassoRegressionP(X, y, Xtest, C, p_list, xlabel):

    for P in p_list:
        poly = PolynomialFeatures(P)
        polyX = poly.fit_transform(X)

        a = 1 / (2 * C)
        model = linear_model.Lasso(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(poly.fit_transform(Xtest)) #train on extend x-axis

        plt.plot(Xtest, yPred, label=f"P:{P}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel("Cyclists")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Lasso Regression 2020 - C: {C}")
    plt.show()


################################## RIDGE REGRESSION #################################

def ridgeRegressionC(X, y, Xtest, c_list, P, xlabel):
    
    poly = PolynomialFeatures(P)
    polyX = poly.fit_transform(X)

    for C in c_list:

        a = 1 / (2 * C)
        model = Ridge(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(poly.fit_transform(Xtest)) #train on extend x-axis

        plt.plot(Xtest, yPred, label=f"C:{C}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.ylabel("Cyclists")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.title(f"Ridge Regression 2020 - P: {P}")
    plt.show()

def ridgeRegressionP(X, y, Xtest, C, p_list, xlabel):

    for P in p_list:

        poly = PolynomialFeatures(P)
        polyX = poly.fit_transform(X)

        a = 1 / (2 * C)
        model = Ridge(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(poly.fit_transform(Xtest)) #train on extend x-axis

        plt.plot(Xtest, yPred, label=f"P:{P}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.ylabel("Cyclists")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.title(f"Ridge Regression 2020- C: {C}")
    plt.show()

############################ KNN REGRESSION ####################################
# CURRENTLY OVERFIT - HAVE TO CHANGE NEIGHBOURS

gamIndex=0;
gamma=0

gamArray = [0,0.0001,0.001,0.01,0.1]#,5,10,25]

def gaussian_kernel0(distances):
    weights = np.exp(0*(distances**2))
    #distArray.append(distances)
    return weights/np.sum(weights)

def gaussian_kernel0001(distances):
    weights = np.exp(-0.0001*(distances**2))
    #distArray.append(distances)
    return weights/np.sum(weights)

def gaussian_kernel001(distances):
    weights = np.exp(-0.001*(distances**2))
    #distArray.append(distances)
    return weights/np.sum(weights)

def gaussian_kernel01(distances):
    weights = np.exp(-0.01*(distances**2))
    #distArray.append(distances)
    return weights/np.sum(weights)

def gaussian_kernel1(distances):
    weights = np.exp(-0.1*(distances**2))
    #distArray.append(distances)
    return weights/np.sum(weights)


def gaussian_kernel(distances):
    weights = np.exp(-gamma*(distances**2))
    return weights/np.sum(weights)

def kNN(Xtrain, ytrain):
    Xtest = Xtrain  # xtest

    ###### WEEKEDAYS ########
    model = KNeighborsRegressor(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain)  # ANything on or above is weekday
    ypred = model.predict(Xtest)

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    #### 2019 ####
    plt.title("2020 kNN Regression")
    plt.scatter(Xtrain, ytrain, marker=".")

    plt.plot(Xtest, ypred, color='darkorange')
    plt.xlabel("Days")
    plt.ylabel("Averages")
    plt.legend(["predict", "train"])
    plt.show()



gamArray = [0,0.0001,0.001,0.01,0.1]#,5,10,25]
def kernelizedKNN(Xtrain, ytrain):
    global gamma
    Xtest = Xtrain
    legendlist = []
    plt.title("2020 kNN Kernelized Regression")
    plt.scatter(Xtrain, ytrain, marker=".")
    for gam in gamArray:
        gamma = gam
        model = KNeighborsRegressor(n_neighbors=len(Xtrain), weights=gaussian_kernel).fit(Xtrain, ytrain)  # ANything on or above is weekday
        ypred = model.predict(Xtest)
        plt.plot(Xtest, ypred)
        legendlist.append("k=7, gamma = " + str(gam))

    plt.xlabel("Days")
    plt.ylabel("Averages")
    legendlist.append("train")
    plt.legend(loc='upper right')
    plt.legend(legendlist,prop={"size":10})
    plt.show()



############################### DUMMY REGRESSION ##############################
def dummy_regressor(X, y):
    model = DummyRegressor(strategy="constant", constant=0.5)
    model.fit(X, y)
    yPred = model.predict(X)
    print(model.score(X, y))

    plt.scatter(X, y, marker=".")
    plt.plot(X, yPred, color="darkorange")
    plt.xlabel("Days")
    plt.ylabel("Cyclists")
    plt.title("Dummy Regression 2020")
    plt.show()


############################### EVALUATION #####################################


def cross_validation(X, y, c_list, poly, algorithm):
    mean_error = []
    std_error = []
    f = 5
    C_range = c_list

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in C_range:
        a = 1 / (2 * C)

        if (algorithm == "Lasso"):
            model = linear_model.Lasso(alpha=a)
        elif (algorithm == "Ridge"):
            model = Ridge(alpha=a)
        elif (algorithm == "KNN"):
            model = KNeighborsRegressor(n_neighbors=7, weights='uniform')
        elif (algorithm == "kNNkern"): model = KNeighborsRegressor(n_neighbors=C, weights=gaussian_kernel)
        else:
            model = DummyRegressor(strategy="constant", constant=0.5)

        temp = []

        kf = KFold(n_splits=f)
        for train, test in kf.split(polyX):
            model.fit(polyX[train], y[train])
            ypred = model.predict(polyX[test])
            # print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(polyX[test], ypred))
            temp.append(mean_squared_error(y[test], ypred))

        if (algorithm == "kNNkern"):
            ################################# kNN EVALUATION ################################

            from sklearn.model_selection import GridSearchCV
            gaussian_kernelCV = [gaussian_kernel0, gaussian_kernel0001, gaussian_kernel001, gaussian_kernel01,
                                 gaussian_kernel1]
            gamArrayDict = dict(weights=gaussian_kernelCV);
            clf = GridSearchCV(model, gamArrayDict, cv=5)
            best_model = clf.fit(X, y)
            print("Best gamma: ",
                  best_model.best_estimator_.get_params()['weights'])  # Gives best gamma - hyper param tuning

        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        scores = cross_val_score(model, polyX, y, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title(f'{algorithm} Regression 5-fold')
    plt.show()


def cross_validation_C(X, y, c_list, poly, algorithm):
    mean_error = []
    std_error = []
    f = 5

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in c_list:
        a = 1 / (2 * C)

        if(algorithm == "Lasso"): model = linear_model.Lasso(alpha=a)
        elif(algorithm == "Ridge"): model = Ridge(alpha=a)
        elif(algorithm == "KNN"): model =  KNeighborsRegressor(n_neighbors=7, weights='uniform')
        else: model = DummyRegressor(strategy="constant", constant=0.5)

        temp = []

        kf = KFold(n_splits=f)
        for train, test in kf.split(polyX):
            model.fit(polyX[train], y[train])
            ypred = model.predict(polyX[test])
            # print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(polyX[test], ypred))
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        scores = cross_val_score(model, polyX, y, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    plt.errorbar(c_list, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title(f'{algorithm} Regression 5-fold')
    plt.show()

def cross_validation_P(X, y, c, p_list, algorithm):
    mean_error = []
    std_error = []
    f = 5

    for P in p_list:
        a = 1 / (2 * c)
        poly = PolynomialFeatures(P)
        polyX = poly.fit_transform(X)

        if(algorithm == "Lasso"): model = linear_model.Lasso(alpha=a)
        elif(algorithm == "Ridge"): model = Ridge(alpha=a)
        elif(algorithm == "KNN"): model =  KNeighborsRegressor(n_neighbors=7, weights='uniform')
        else: model = DummyRegressor(strategy="constant", constant=0.5)

        temp = []

        kf = KFold(n_splits=f)
        for train, test in kf.split(polyX):
            model.fit(polyX[train], y[train])
            ypred = model.predict(polyX[test])
            # print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(polyX[test], ypred))
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        scores = cross_val_score(model, polyX, y, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    plt.errorbar(p_list, mean_error, yerr=std_error)
    plt.xlabel('Polynomial')
    plt.ylabel('Mean square error')
    plt.title(f'{algorithm} Regression 5-fold')
    plt.show()

################################# SET UP ########################################

#2020#
# ---------------data cleaning notes-----------------
# NaN rows (3026-3039) filled in with zeros

df = pd.read_csv("jan-oct-2020-cycle-data.csv", comment='#')

# identifies null columns
# print(df[df.iloc[:,1].isnull()])
# print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:, 4])

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


################################ main sequence ###################################

#plotting dataset - all days v weekdays v weekends
weekDayPeaks, weekEndPeaks = getWeekDaysAndWeekEndsFromList(peaks)
weekDayAverages, weekEndAverages = getWeekDaysAndWeekEndsFromList(averages)

plotDayData(days[JAN:END], peaks[JAN:END], "", "Peaks", "All Days")
plotDayData(days[JAN:END], weekDayAverages[JAN:END], "", "Averages", "Week Days" )
plotDayData(days[JAN:END], weekEndPeaks[JAN:END], "", "Peaks", "Weekends")

#overwriting above, beginning regression
weekDayPeaks, weekEndPeaks = splitWeekDays(peaks)
weekDayAverages, weekEndAverages = splitWeekDays(averages)
weekDayMins, weekEndMins = splitWeekDays(mins)

# week days/ends as long as any of the arbitrarily picked averages/peaks/min
numWeekDays = len(weekDayAverages)
numWeekEnds = len(weekEndAverages)

weekDays = np.array(list(range(0, numWeekDays))).reshape(-1, 1)
weekEnds = np.array(list(range(0, numWeekEnds))).reshape(-1, 1)

#hourly traffic
plotHourlyTraffic(hoursGroupedByDay[JAN:FEB], "November")

#set timeline for below regressions
start, end = getWeekDayCountBetweenMonths(1, 10)
X = weekDays[start:end]
y = weekDayAverages[start:end]
c_list = [1, 50, 100, 200, 400, 800]
k_list = [1,2,3,4,5,6,7,8,9,10]

Xnorm, ynorm = normalize(X, y)
cross_validation(Xnorm, ynorm, k_list, 2, "kNN")

cross_validation(Xnorm, ynorm, k_list, 2, "kNNkern")
kNN(X, y)
kernelizedKNN(X,y)

#kNN regression analysis
start, end = getWeekDayCountBetweenMonths(1, 12)
X = weekDays[start:end]
y = weekDayAverages[start:end]
Xnorm, ynorm = normalize(X, y)
cross_validation(Xnorm, ynorm, k_list, 2, "kNN")
cross_validation(Xnorm, ynorm, k_list, 2, "kNNkern")
kNN(X, y)
kernelizedKNN(X, y)

#kNN regression analysis
start, end = getWeekEndCountBetweenMonths(1, 12)
X = weekEnds[start:end]
y = weekEndAverages[start:end]
Xnorm, ynorm = normalize(X, y)
cross_validation(Xnorm, ynorm, k_list, 2, "kNN")


cross_validation(Xnorm, ynorm, k_list, 2, "kNNkern")
kNN(X, y)
kernelizedKNN(X, y)
X, y = normalize(X, y)
p_list = [1, 2, 3, 4, 5]
polynomial = 3
C = 500
limit = (1/305)*(366 + 90)         #predict first three months of 2021
Xtest=np.linspace(0,limit).reshape(-1, 1)

#lasso regression analysis
cross_validation_C(X, y, c_list, polynomial, "Lasso")
lassoRegressionC(X, y, Xtest, c_list, polynomial, "Weekdays")
cross_validation_P(X, y, C, p_list, "Lasso")
lassoRegressionP(X, y, Xtest, C, p_list, "Weekdays")

#ridge regression analysis
cross_validation_P(X, y, C, p_list, "Ridge")
ridgeRegressionP(X, y, Xtest, C, p_list, "Weekdays")
cross_validation_C(X, y, c_list, polynomial, "Ridge")
ridgeRegressionC(X, y, Xtest, c_list, polynomial, "Weekdays")


#dummy regression analysis
dummy_regressor(X, y)
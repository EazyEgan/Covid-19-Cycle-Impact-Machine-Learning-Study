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


def getWeekEndCountBetweenMonths(startMonth, endMonth):
    if (startMonth == 1):
        startIndex = 0
        endIndex = np.busday_count('2019-01', f'2019-0{endMonth}' if (endMonth < 10) else f'2019-{endMonth}',
                                   weekmask="Sat Sun")
    else:
        startIndex = np.busday_count(f'2019-01', f'2019-0{startMonth}' if (startMonth < 10) else f'2019-0{startMonth}',
                                     weekmask="Sat Sun")
        endIndex = startIndex + np.busday_count(f'2019-0{startMonth}' if (startMonth < 10) else f'2019-{startMonth}'
                                                , f'2019-0{endMonth}' if (endMonth < 10) else f'2019-{endMonth}',
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
    plt.title(f"{title} Cycle Volume 2019")
    plt.show()


def plotHourlyTraffic(y, title):
    for i in range(0, len(hours)):
        plt.plot(hours, y[i])
    plt.xlabel("Hour")
    plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()


################################## LASSO REGRESSION #################################


def lassoRegression(X, y, c_list, poly, xlabel):

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in c_list:

        a = 1 / (2 * C)
        model = linear_model.Lasso(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(polyX)

        plt.plot(X, yPred, label=f"C:{C}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel("Cyclists")
    plt.title("lasso Regression 2019")
    plt.show()


################################## RIDGE REGRESSION #################################

def ridgeRegression(X, y, c_list, poly, xlabel):
    
    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in c_list:

        a = 1 / (2 * C)
        model = Ridge(alpha=a)
        model.fit(polyX, y)
        yPred = model.predict(polyX)

        plt.plot(X, yPred, label=f"C:{C}")
    plt.scatter(X, y, marker=".")
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel("Cyclists")
    plt.title("Ridge Regression 2019")
    plt.show()


############################ KNN REGRESSION ####################################
# CURRENTLY OVERFIT - HAVE TO CHANGE NEIGHBOURS
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot all samples
   X_test, y_test = X[test_idx, :], y[test_idx]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)
   # highlight test samples
   if test_idx:
      X_test, y_test = X[test_idx, :], y[test_idx]
      plt.scatter(X_test[:, 0], X_test[:, 1], c='',
               alpha=1.0, linewidth=1, marker='o',
               s=55, label='test set')


start, end = getWeekDayCountBetweenMonths(1, 12)
X = days[start:end]
y = peaks[start:end]
X, y = normalize(X, y)

X_train = X
X_test = X
y_train= y
y_test = y

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
print(len(X_train_std), len(X_test_std))
print(len(y_train), len(y_test))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
import pandas as pd
y_resampled = pd.DataFrame(y_train)
knn.fit(X_train_std, y_resampled.values.ravel())

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
"""
def gaussian_kernel100(distances):
    weights = np.exp(-100 * (distances ** 2))
    return weights / np.sum(weights)

def gaussian_kernel1000(distances):
    weights = np.exp(-1000 * (distances ** 2))
    return weights / np.sum(weights)

def gaussian_kernel10000(distances):
    weights = np.exp(-10000 * (distances ** 2))
    return weights / np.sum(weights)

def kNN(Xtrain, ytrain):
    """
    model = KNeighborsClassifier(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain)
    ypred = model.predict(Xtrain)
    plt.scatter(Xtrain, ytrain, color='red', marker ='+')
    plt.plot(Xtrain, ypred, color='green')
    plt.xlabel("inputx")
    plt.ylabel("outputy")
    plt.legend(["predict", "train"])
    plt.show()
    
    ############################### KNN CLASSIFIER #################################
    
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
    """

    Xtest = Xtrain #xtest


###### WEEKEDAYS ########
    model = KNeighborsRegressor(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain) #ANything on or above is weekday
    ypred = model.predict(Xtest)

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
#### 2019 ####
    plt.scatter(Xtrain,ytrain, marker="." )

    plt.plot(Xtest, ypred, color='darkorange')
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend(["predict", "train"])
    plt.show()

#### 2020 ####
    X2, y2 = days2, peaks2
    plt.scatter(X2,y2, marker=".")

    plt.plot(Xtest, ypred, color='darkorange')
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend(["predict", "train"])
    plt.show()

    '''

##### WEEKENDS ##### MODEL MUST BE TRAINED ON WEEKEND DATA THAT HASNT BEEN EXTRACTED FROM THE DATASET BECAUSE IT TREATS THEM AS BEING SEQUENTIAL
    #OR MAYBE WE CAN JUST STRETCH IT OUT BECAUSE IT MIGHT BE THE SAME? NOT ENTIRELY SURE PLUS IT'S 5:11 AM AND I HAVE BEEN AWAKE FAR TOO LONG
    
    start, end = getWeekEndCountBetweenMonths(1, 12)
    X = weekEnds[start:end]
    y = weekEndPeaks[start:end]
    #X, y = normalize(days2, peaks2)
    model = KNeighborsRegressor(n_neighbors=7, weights='uniform').fit(X, y)  # ANything on or above is weekday
    ypred = model.predict(X)
    #### 2019 ####
    plt.scatter(Xtrain, ytrain, color='red', marker='+')

    plt.plot(X, ypred, color='green')
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend(["predict", "train"])
    plt.show()

    #### 2020 ####
    plt.scatter(X2, y2, color='red', marker='+')

    plt.plot(X, ypred, color='green')
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend(["predict", "train"])
    plt.show()
    '''
def kernelizedKNN(Xtrain, ytrain):
    Xtest = Xtrain

    model2 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel100).fit(Xtrain, ytrain)
    ypred2 = model2.predict(Xtest)

    model3 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel1000).fit(Xtrain, ytrain)
    ypred3 = model3.predict(Xtest)

    model4 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel10000).fit(Xtrain, ytrain)
    ypred4 = model4.predict(Xtest)

    plt.scatter(Xtrain, ytrain, marker=".")
    plt.plot(Xtest, ypred2, color='purple')
    plt.plot(Xtest, ypred3, color='darkorange')
    plt.plot(Xtest, ypred4, color='green')
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend(["k=7,sigma=100", "k=7,sigma=1000", "k=7,sigma=10000", "train"])
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
    plt.title("Dummy Regression 2019")
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

        if(algorithm == "lasso"): model = linear_model.Lasso(alpha=a)
        elif(algorithm == "ridge"): model = Ridge(alpha=a)
        elif(algorithm == "kNN"): model =  KNeighborsRegressor(n_neighbors=7, weights='uniform')
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

    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title(f'{algorithm} Regression 5-fold')
    plt.show()


################################# SET UP ########################################

#2019#
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


################################ main sequence ###################################
'''
#plotting dataset - all days v weekdays v weekends
weekDayPeaks, weekEndPeaks = getWeekDaysAndWeekEndsFromList(peaks)

plotDayData(days[JAN:END], peaks[JAN:END], "", "Peaks", "All Days")
plotDayData(days[JAN:END], weekDayPeaks[JAN:END], "", "Peaks", "Week Days" )
plotDayData(days[JAN:END], weekEndPeaks[JAN:END], "", "Peaks", "Weekends")
'''

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
plotHourlyTraffic(hoursGroupedByDay[NOV:DEC], "November")

#set timeline for below regressions
start, end = getWeekDayCountBetweenMonths(1, 12)
X = weekDays[start:end]
y = weekDayAverages[start:end]
X, y = normalize(X, y)
c_list = [0.1, 1, 50, 100, 500]

#lasso regression analysis
cross_validation(X, y, c_list, 4, "lasso")
lassoRegression(X, y, c_list, 4, "Weekdays")

#ridge regression analysis
cross_validation(X, y, c_list, 4, "ridge")
ridgeRegression(X, y, c_list, 4, "Weekdays")

#kNN regression analysis
cross_validation(X, y, c_list, 2, "kNN")
kNN(X, y)

#Kernalised kNN regression analysis
cross_validation(X, y, c_list, 2, "kNN")
kernelizedKNN(X, y)

#dummy regression analysis
dummy_regressor(X, y)


############################# COMPARISON ####################################

X = days 
y = peaks 
X2 = days2
y2 = peaks2

#kNN(X, y)

plt.scatter(X, y, color="darkorange", marker="+")
plt.scatter(X2, y2, marker=".")
plt.xlabel("days")
plt.ylabel("Cyclists")
plt.title("lasso Regression 2019")
plt.show()

'''
#Test data augmentation
""" 
Xtest = []
grid = np.linspace(-3, 3)
for i in grid:
    Xtest.append([i])
Xtest = np.array(Xtest)
"""

X = days 
y = peaks 
#X, y = normalize(X, y)

start, end = getWeekEndCountBetweenMonths(1, 12)
Xwknd = weekEnds[start:end]
ywknd = weekEndPeaks[start:end]
#X2, y2 = normalize(X2, y2)

lassoRegression(X, y, 0.0001, 2)
ridgeRegression(X, y, 0.0001, 2)
kNN(X, y)
dummy_regressor(X, y)
cross_validation(X, y, 2, "lasso")
'''
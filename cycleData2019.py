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

# start of each month
JAN = 0;
FEB = 31;
MAR = 59;
APR = 90;
MAY = 120;
JUN = 151;
JUL = 181
AUG = 212;
SEP = 243;
OCT = 273;
NOV = 304;
DEC = 334;
END = 365

#
DAY_OFFSET = 2;
MON = 1;
TUE = 2;
WED = 3;
THUR = 4;
FRI = 5;
SAT = 6;
SUN = 7

###########2020########

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



df = pd.read_csv("jan-dec-2019-cycle-data.csv", comment='#')

################################### SET UP ####################################

# identifies null columns
# print(df[df.iloc[:,1].isnull()])
# print(df.iloc[2130:2140,1].isnull())

GroveRoad = np.array(df.iloc[:, 1])

dataLen = len(GroveRoad)
numDays = int(dataLen / 24)

days = np.array(list(range(0, numDays))).reshape(-1, 1)
hours = np.array(list(range(0, 24))).reshape(-1, 1)

hoursGroupedByDay = np.array_split(GroveRoad, numDays)

# getting peak of each day
peaks = []
for i in range(0, numDays):
    peaks.append(max(hoursGroupedByDay[i]))
peaks = np.array(peaks).reshape(-1, 1)

# getting min of each day
mins = []
for i in range(0, numDays):
    mins.append(min(hoursGroupedByDay[i]))
mins = np.array(mins).reshape(-1, 1)

# getting average of each day
averages = []
for i in range(0, numDays):
    averages.append(sum(hoursGroupedByDay[i]) / len(hoursGroupedByDay[i]))
averages = np.array(averages).reshape(-1, 1)

# reshaping hoursGroupedByDay after above functions to avoid nesting reshaping
for i in range(0, len(hours)):
    hoursGroupedByDay[i] = hoursGroupedByDay[i].reshape(-1, 1)


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


def plotDayData(dataType, daysSubArray, dataSubArray, title):
    plt.scatter(daysSubArray, dataSubArray)
    plt.xlabel("Hour");
    plt.ylabel("Cyclists")
    plt.legend([f'Day {dataType}'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()


def plotHourlyTraffic(hoursGroupedSub, title):
    for i in range(0, len(hours)):
        plt.plot(hours, hoursGroupedSub[i])
    plt.xlabel("Hour");
    plt.ylabel("Cyclists")
    plt.legend(['Total'])
    plt.title(f"{title} cycle volume 2019")
    plt.show()


################################## LASSO REGRESSION #################################


def lassoRegression(X, y, c, poly):
    C = c

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    a = 1 / (2 * C)
    model = linear_model.Lasso(alpha=a)
    model.fit(polyX, y)
    yPred = model.predict(polyX)

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days");
    plt.ylabel("Cyclists")
    plt.title("lasso Regression 2019")
    plt.show()


################################## RIDGE REGRESSION #################################

def ridgeRegression(X, y, c, poly):
    C = c

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    a = 1 / (2 * C)
    model = Ridge(alpha=a)
    model.fit(polyX, y)
    yPred = model.predict(polyX)

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days")
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

def kNN(Xtrain, ytrain):#, xtest):

    from sklearn.neighbors import KNeighborsClassifier
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

    model = KNeighborsRegressor(n_neighbors=7, weights='uniform').fit(Xtrain, ytrain) #ANything on or above is weekday
    ypred = model.predict(Xtest)

    plt.rc('font', size=18);
    plt.rcParams['figure.constrained_layout.use'] = True
    #start, end = getWeekDayCountBetweenMonths(1, 12)
    #X = weekDays[start:end]
    #y = weekDayPeaks[start:end]
    X, y = normalize(days2, peaks2)

    plt.scatter(X,y, color='red', marker='+')

    plt.plot(Xtest, ypred, color='green')
    plt.xlabel("input x");
    plt.ylabel("output y");
    plt.legend(["predict", "train"])
    plt.show()



    plt.scatter(Xtrain, ytrain, color='red', marker='+')

    plt.plot(Xtest, ypred, color='green')
    plt.xlabel("input x");
    plt.ylabel("output y");
    plt.legend(["predict", "train"])
    plt.show()

    def gaussian_kernel100(distances):
        weights = np.exp(-100 * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_kernel1000(distances):
        weights = np.exp(-1000 * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_kernel10000(distances):
        weights = np.exp(-10000 * (distances ** 2))
        return weights / np.sum(weights)

    model2 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel100).fit(Xtrain, ytrain)
    ypred2 = model2.predict(Xtest)

    model3 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel1000).fit(Xtrain, ytrain)
    ypred3 = model3.predict(Xtest)

    model4 = KNeighborsRegressor(n_neighbors=7, weights=gaussian_kernel10000).fit(Xtrain, ytrain)
    ypred4 = model4.predict(Xtest)

    plt.scatter(Xtrain, ytrain, color='red', marker='+')
    plt.plot(Xtest, ypred2, color='blue')
    plt.plot(Xtest, ypred3, color='orange')
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

    plt.scatter(X, y, color="blue")
    plt.plot(X, yPred, color="red")
    plt.xlabel("Days");
    plt.ylabel("Cyclists")
    plt.title("Dummy Regression 2019")
    plt.show()


############################### EVALUATION #####################################

def cross_validation(X, y, poly):
    mean_error = [];
    std_error = []
    f = 5
    C_range = [0.001, 0.01, 1, 100, 1000]

    poly = PolynomialFeatures(poly)
    polyX = poly.fit_transform(X)

    for C in C_range:
        a = 1 / (2 * C)
        model = linear_model.Lasso(alpha=a)
        temp = []

        kf = KFold(n_splits=f)
        for train, test in kf.split(polyX):
            model.fit(polyX[train], y[train])
            ypred = model.predict(polyX[test])
            # print("intercept ", model.intercept_, "slope ", model.coef_, " square error ", mean_squared_error(polyX[test], ypred))
            temp.append(mean_squared_error(peaks[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

        scores = cross_val_score(linear_model.Lasso(), polyX, y, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel('C');
    plt.ylabel('Mean square error')
    plt.title('Lasso Regression 5-fold')
    plt.show()


############################### EXECUTION ########################################

weekDayPeaks, weekEndPeaks = splitWeekDays(peaks)
weekDayMins, weekEndMins = splitWeekDays(mins)
weekDayAverages, weekEndAverages = splitWeekDays(averages)

# week days/ends as long as any of the arbitrarily picked averages/peaks/min
numWeekDays = len(weekDayAverages)
numWeekEnds = len(weekEndAverages)

weekDays = np.array(list(range(0, numWeekDays))).reshape(-1, 1)
weekEnds = np.array(list(range(0, numWeekEnds))).reshape(-1, 1)

##plotDayData((string) dataType, days[day range], peaks[day range], (string) time period)
# plotDayData("Peaks", days[JAN:FEB], peaks[JAN:FEB], "January")

# start, end = getWeekDayCountBetweenMonths(1, 12)
# plotDayData("Peaks", weekDays[start:end], weekDayPeaks[start:end], "Full year")

##plotHourlyTraffic(hoursGroupedByDay[day_range], title)
# plotHourlyTraffic(hoursGroupedByDay[NOV:DEC], "November")

start, end = getWeekDayCountBetweenMonths(1, 12)
X = days #weekDays[start:end]
y = peaks #weekDayPeaks[start:end]
X, y = normalize(X, y)
print(X, y)
#Test data augmentation
""" 
Xtest = []
grid = np.linspace(-3, 3)
for i in grid:
    Xtest.append([i])
Xtest = np.array(Xtest)
"""
lassoRegression(X, y, 0.0001, 2)
ridgeRegression(X, y, 0.0001, 2)
kNN(X, y)#,Xtest)
dummy_regressor(X, y)
cross_validation(X, y, 2)

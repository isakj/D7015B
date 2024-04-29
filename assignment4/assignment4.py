# %% [markdown]
# # Assignment 4, D7015B, Industrial AI and eMaintenance - Part I: Theories & Concepts #
# 
# Isak Jonsson, isak.jonsson@gmail.com

# %% [markdown]
# Preamble

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import geopandas
from geopy.geocoders import Nominatim
import calendar
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from lazypredict import LazyClassifier
from lazypredict import LazyRegressor

# %%
df = pd.read_excel('train delays.xlsx')

print(df.shape)
print(dict(df).keys())

# %%
df = df.drop('Unnamed: 0', axis='columns');

loc = Nominatim(user_agent="GetLoc")

memo = {}

# %% [markdown]
# Resolving latitute and longitude
# * Memoise the lookup of locations, as it is a bit slow. This has the effect that we are only looking up each location once, even if it occurs multiple times.
# * Strategy for resolving names:
#   - Try to lookup the location without "C" (central station), with "Sverige" (Sweden) appended
#   - If that fails, lookup the location as-is, but with "Sverige" (Sweden) appended
#   - If that fails, lookup the first word of the location, with "Sverige" (Sweden) appended
#   - If that fails, remove "sberget" ('s mountain), with "Sverige" (Sweden) appended
#   - If that fails, remove "godsbangård" (freight yard), with "Sverige" (Sweden) appended
#   - If that fails, lookup the location as-is

# %%
def memgeocode(place):
    if place in memo:
        if memo[place] == 0:
            return None
        return memo[place]
    memo[place] = loc.geocode(place)
    if not memo[place]:
        memo[place] = 0
    return memo[place]

hits = {}
latitude = []
longitude = []
address = []
for value in df["Place"]:
    location = None
    location = location or memgeocode(re.sub("s [cC]$", " C", value)+", Sverige")
    location = location or memgeocode(value+", Sverige")
    location = location or memgeocode(re.sub(' .*$','',value)+", Sverige")
    location = location or memgeocode(re.sub('sberget$','',value)+", Sverige")
    location = location or memgeocode(re.sub('godsbangård$','',value)+", Sverige")
    location = location or memgeocode(value)
    if not value in hits:
        hits[value] = 1
        print(value,' = ',location)
    latitude.append(location.latitude)
    longitude.append(location.longitude)
    address.append(str(location))

df["latitude"] = latitude
df["longitude"] = longitude
df["address"] = address
    

# %% [markdown]
# ## Geographical location ##
# 
# Trying to get an understanding of where these delays happen geographically.
# 
# Let's plot the number of delay per location. Larger circle: more number of delays.
# 
# ### Conclusion ###
# 
# 4 general areas:
# * The coast of Norrland in general
# * Luleå-Kiruna
# * Sundsvall-Norway
# * Värmland

# %%
world = geopandas.read_file("ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")

minlat = df["latitude"].min()
maxlat = df["latitude"].max()
minlong = df["longitude"].min()
maxlong = df["longitude"].max()

DDF = df.groupby(["address","latitude","longitude"],as_index=False).size()

gdf = geopandas.GeoDataFrame(
    DDF, geometry=geopandas.points_from_xy(DDF.longitude, DDF.latitude), crs="EPSG:4326"
)
plt.figure(dpi=1200)

ax = world.clip([minlong - (maxlong-minlong)/10, minlat - (maxlat-minlat)/10, maxlong + (maxlong-minlong)/10, maxlat + (maxlat-minlat)/10]).plot(color="white", edgecolor="black")
bx = gdf.plot(ax=ax, color="red", markersize=DDF['size']/3.0, alpha=0.5, linewidth=0)


# %% [markdown]
# ## Time of year ##
# 
# ### Hypothesis ###
# 
# * Due to weather, delays are worse in wintertime.
# * Some extreme delays skew the average value, and should be removed.
# 
# ### Conclusion ###
# 
# * Some relation to month of year, but not clear. Low delays in December stands out. And very high in November.
# * Extreme delays do not seem to make a bit impact. The overall pattern is the same, even if delay is capped.

# %%
df['month'] = pd.to_datetime(df['Date']).dt.month
df['day_of_year'] = pd.to_datetime(df['Date']).transform(lambda d: (d.to_pydatetime() - datetime.datetime(d.to_pydatetime().year, 1, 1))  / datetime.timedelta(days=1))
df['cap delay'] = df['registered delay'].clip(upper=40)
df['cap delay2'] = df['registered delay'].clip(upper=20)
df['delay category'] = pd.cut(df['registered delay'], [0, 7.5, 25.5, np.inf], labels=['small', 'medium', 'large'])
X = df[{'month','registered delay', 'cap delay'}].groupby(['month'], as_index=False).mean()
plt.plot(X['month'].transform(lambda m: calendar.month_name[m]), X['registered delay'], label='delay')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Month of year')
plt.plot(X['month'].transform(lambda m: calendar.month_name[m]), X['cap delay'], label='min(delay,40 minutes)')
plt.legend()

# %% [markdown]
# ## Location ##
# 
# ### Hypothesis ###
# 
# * Further north, longer delays (again weather)
# 
# ### Conclusion ###
# 
# * Latitude: no clear difference could be observed. Surprising that the southernmost locations also have high average delay
# * Longitude: more further east. Could be due to consequential delays along the coast of Norrland. Rather big difference when very long delays are capped.

# %%
df['latitude_int'] = df['latitude'].round()
X = df[{'latitude_int','registered delay','cap delay'}].groupby(['latitude_int'], as_index=False).mean()
plt.figure(0)
plt.plot(X['latitude_int'], X['registered delay'], label='delay')
plt.plot(X['latitude_int'], X['cap delay'], label='min(delay,40 minutes)')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Latitude')
plt.legend()

df['longitude_int'] = df['longitude'].round()
X = df[{'longitude_int','registered delay','cap delay'}].groupby(['longitude_int'], as_index=False).mean()
plt.figure(1)
plt.plot(X['longitude_int'], X['registered delay'], label='delay')
plt.plot(X['longitude_int'], X['cap delay'], label='min(delay,40 minutes)')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$, delay')
plt.xlabel('Longitude')
plt.legend()


# %% [markdown]
# ## Reason ##
# 
# ### Hypothesis ###
# 
# * Perhaps engine related reasons cause longer delays
# * Perhaps difference in delay depending on operator
# 
# ### Conclusion ###
# 
# * Reason: no clear difference could be observed.
# * Operator: no clear difference could be observed.
# 
# ### Result ###
# 
# * It still makes sense to remove entries with "no reason"

# %%
X = df.groupby(['Reason code Level 2'], as_index=False).mean()
plt.figure(0)
plt.bar(X['Reason code Level 2'], X['registered delay'], label='delay')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Reason')
plt.legend()

X = df.groupby(['Reason code Level 2'], as_index=False).count()
plt.figure(1)
plt.bar(X['Reason code Level 2'], X['registered delay'], label='count')
plt.xticks(rotation=90)
plt.ylabel('$n_x$')
plt.xlabel('Reason')
plt.legend()

X = df.groupby(['Operator'], as_index=False).mean()
plt.figure(2)
plt.bar(X['Operator'], X['registered delay'], label='delay')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Operator')
plt.legend()

X = df.groupby(['Operator'], as_index=False).count()
plt.figure(3)
plt.pie(X['registered delay'], labels=X['Operator'])
plt.title('Count of delays')



# %% [markdown]
# ## Routes ##
# 
# ### Hypothesis ###
# 
# * Some routes are more prone to delays than others.
# 
# ### Conclusion ###
# 
# * The "routes" stand out.
#   - BLG-MRA. This route has only one datapoint. What is interesting is that this route is between Borlänge and Mora, and the delay is registered in Umeå, some 600 km away. But as it is only one datapoint, no more work is spent on this.
#   - "-" and TJT. TJT stands for Tjänstetåg (Service train), and is the route assigned to trains that are being transported to another location (without passenger or freight). At the same time, one could argue that "-" are also trains that are not really in service.
#     Further, one could speculate that these trains are more likely to have delays, as the consequence for delays of an empty train is less severe.
# 
# 
# ### Result ###
# 
# * Remove all delays for routes TJT and "-".

# %%
X = df.groupby(['Route'], as_index=False).mean()
plt.figure(0)
plt.bar(X['Route'], X['registered delay'], label='delay')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Reason')
plt.legend()
X = df.groupby(['Route'], as_index=False).count()
plt.figure(1)
plt.bar(X['Route'], X['registered delay'], label='count')
plt.xticks(rotation=90)
plt.ylabel('$n_x$')
plt.xlabel('Routes')


# %% [markdown]
# ## Adjusted data ##
# 
# With the routes above removed, let's plot the new month-delay and location-delay relationship.
# 
# ### Observations ###
# 
# * The November peak is all gone.
# * A somewhat stronger correlation between latitude and delay
# 
# We are also plotting how the Reason code affects delay. For the average delay, it does not seem to have a big impact. But it is interesting that so many delays are caused by terminal handling.
# 
# ## Delay categories ##
# 
# Introducing delay categories:
# * `small`: 0-7 minutes
# * `medium`: 8-25 minutes
# * `large`: 26 and more minutes
# 
# Plotting reason and categories.
# 
# ### Observations ###
# 
# * Quite the same distribution of small/medium/large regardless of reason

# %%
df = df[~df['Route'].isin({'TJT','-'})]
df = df[~df['Reason code Level 2'].isin({'Ingen uppgift från JF'})]

df['month'] = pd.to_datetime(df['Date']).dt.month
df['cap delay'] = df['registered delay'].clip(upper=40)
X = df[{'month','registered delay', 'cap delay'}].groupby(['month'], as_index=False).mean()
plt.plot(X['month'].transform(lambda m: calendar.month_name[m]), X['registered delay'], label='delay')
plt.plot(X['month'].transform(lambda m: calendar.month_name[m]), X['cap delay'], label='min(delay,40 minutes)')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Month of year')
plt.legend()

df['latitude_int'] = df['latitude'].round()
X = df[{'latitude_int','registered delay','cap delay'}].groupby(['latitude_int'], as_index=False).mean()
plt.figure(2)
plt.plot(X['latitude_int'], X['registered delay'], label='delay')
plt.plot(X['latitude_int'], X['cap delay'], label='min(delay,40 minutes)')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Latitude')
plt.legend()

df['longitude_int'] = df['longitude'].round()
X = df[{'longitude_int','registered delay','cap delay'}].groupby(['longitude_int'], as_index=False).mean()
plt.figure(3)
plt.plot(X['longitude_int'], X['registered delay'], label='delay')
plt.plot(X['longitude_int'], X['cap delay'], label='min(delay,40 minutes)')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Longitude')
plt.legend()

X = df.groupby(['Reason code Level 2'], as_index=False).mean()
plt.figure(4)
plt.bar(X['Reason code Level 2'], X['registered delay'], label='delay')
plt.xticks(rotation=90)
plt.ylabel('$\\bar{x}$')
plt.xlabel('Reason')
plt.legend()

X = df.groupby(['Reason code Level 2'], as_index=False).count()
plt.figure(5)
plt.bar(X['Reason code Level 2'], X['registered delay'], label='count')
plt.xticks(rotation=90)
plt.ylabel('$n_x$')
plt.xlabel('Reason')

plt.figure(6)
sns.countplot(x='Reason code Level 2', hue='delay category', data=df.sort_values(by='Reason code Level 2'))
plt.xticks(rotation=90)
plt.ylabel('$n_x$')
plt.xlabel('Reason')

DDF = df.groupby(["address","latitude","longitude"],as_index=False).size()
gdf = geopandas.GeoDataFrame(
    DDF, geometry=geopandas.points_from_xy(DDF.longitude, DDF.latitude), crs="EPSG:4326"
)
plt.figure(7)
plt.figure(dpi=1200)
ax = world.clip([minlong - (maxlong-minlong)/10, minlat - (maxlat-minlat)/10, maxlong + (maxlong-minlong)/10, maxlat + (maxlat-minlat)/10]).plot(color="white", edgecolor="black")
bx = gdf.plot(ax=ax, color="red", markersize=DDF['size']/3.0, alpha=0.5, linewidth=0)

#plt.errorbar(X['month'],X)
#df.plot(df['month'], df.groupby('month').agg(np.mean)['registered delay'])



# %% [markdown]
# # Linear regression #
# 
# ## Month of year vs delay ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `scikit-learn` is used for linear regression.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Very poor result. This is also evident when looking at the scatter plot.
# 
# ## More observations ##
# 
# Other plots:
# 
# * Day of year vs delay. Not easy to find a pattern.
# * Route vs delay. Same lack of pattern. Also introducing "canonical route", merging the route A->B with B->A.
# * Reason vs delay. See also delay category plot above.

# %%
X = np.array(df['month'], dtype=np.float64).reshape(-1, 1)
y = np.array(df['registered delay'], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

print(model)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficient of determination:", model.score(X_test, y_test))
print(model.intercept_)
print(model.coef_)

plt.scatter(X, y, color="black", label='delay')
plt.plot(X_test, y_pred, color="blue", linewidth=3, label='prediction')

plt.xticks(())
plt.yticks(())
plt.xlabel('month')
plt.ylabel('delay')

plt.legend()
plt.show()

plt.figure(1, figsize=(15,10))
sns.scatterplot(df, x='day_of_year', y='registered delay', hue='Reason code Level 2')
plt.figure(2, figsize=(15,10))
sns.scatterplot(df, x='day_of_year', y='registered delay', hue='Route')

routemap = {}
destinationsmap = {}
routes = df['Route'].unique()
routes.sort()
for route in routes:    
    destinations = re.split(r'[-/]', route)
    destinations.sort()
    destinations = '-'.join(destinations)
    if destinations in destinationsmap:
        routemap[route] = destinationsmap[destinations]
    else:
        routemap[route] = destinationsmap[destinations] = route

df['canonical route'] = df['Route'].map(routemap)
plt.figure(3, figsize=(15,10))
sns.scatterplot(df, x='day_of_year', y='registered delay', hue='canonical route')
plt.figure(4, figsize=(15,10))
sns.scatterplot(df, x='Reason code Level 2', y='registered delay', hue='canonical route')
plt.xticks(rotation=90)

# %% [markdown]
# ## Latitude vs delay ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `scikit-learn` is used for linear regression.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Very poor result. This is also evident when looking at the scatter plot.

# %%
X = np.array(df['latitude'], dtype=np.float64).reshape(-1, 1)
y = np.array(df['registered delay'], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

print(model)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficient of determination:", model.score(X_test, y_test))
print(model.intercept_)
print(model.coef_)

plt.scatter(X, y, color="black", label='delay')
plt.plot(X_test, y_pred, color="blue", linewidth=3, label='prediction')

plt.xticks(())
plt.yticks(())
plt.xlabel('latitude')
plt.ylabel('delay')
plt.legend()

plt.show()

# %% [markdown]
# ## Longitude vs delay ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `scikit-learn` is used for linear regression.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Very poor result. This is also evident when looking at the scatter plot.

# %%
X = np.array(df['longitude'], dtype=np.float64).reshape(-1, 1)
y = np.array(df['registered delay'], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

print(model)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficient of determination:", model.score(X_test, y_test))
print(model.intercept_)
print(model.coef_)

plt.scatter(X, y, color="black", label="delay")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="prediction")

plt.xticks(())
plt.yticks(())
plt.xlabel('longitude')
plt.ylabel('delay')
plt.legend()

plt.show()

# %% [markdown]
# # Polynomial regression #
# 
# ## Latitude, longitude, month vs delay ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `scikit-learn` is used for polynomial regression.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Still very poor result.

# %%
X = np.array(df[{'latitude','longitude','month'}], dtype=np.float64)
y = np.array(df['registered delay'], dtype=np.float64)

poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

print(model)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Coefficient of determination:", model.score(X_test, y_test))


# %% [markdown]
# # Classification #
# 
# ## Longitude, latitude, day of year, reasons, route, place, train route number ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `LazyClassifier` is used for a "try-all" attempt on an "all-in" set of features.
# The capped delay (delays more than 40 minutes are treated as 40 minutes) is
# output variable.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Quite poor result. $R^2 \approx 0.34$, not really good enough to be usable.

# %%
df['Reason2'] = pd.Categorical(df['Reason code Level 2']).codes
df['Reason3'] = pd.Categorical(df['Reason code Level 3']).codes
df['RouteCode'] = pd.Categorical(df['Route']).codes
df['PlaceCode'] = pd.Categorical(df['Place']).codes
X = df[['longitude','latitude','day_of_year', 'Reason2', 'Reason3','RouteCode','PlaceCode','Train mission']]
y = df[['cap delay']]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# %% [markdown]
# # Regression #
# 
# ## Longitude, latitude, day of year, reasons, route, place, train route number ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `LazyRegressor` is used for a "try-all" attempt on an "all-in" set of features.
# The capped delay (delays more than **20** minutes are treated as **20** minutes) is
# output variable.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# Quite poor result. $R^2 \approx 0.28$, not really good enough to be usable.

# %%
X = df[['longitude','latitude','day_of_year', 'Reason2', 'Reason3','RouteCode','PlaceCode','Train mission']]
y = df[['cap delay2']]

print(df.shape)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# %%

clf = HistGradientBoostingRegressor()
clf.fit(X, y)

print(models.iloc[0])
for i in range(0,20):
    test = df.iloc[i][['longitude','latitude','day_of_year', 'Reason2', 'Reason3','RouteCode','PlaceCode','Train mission']].to_frame()
    result = clf.predict(test.values.reshape(1,8))
    print(result, df.iloc[i]['registered delay'])


# %% [markdown]
# # Classification #
# 
# ## Longitude, latitude, day of year, reasons, route, place, train route number ##
# 
# Out of the washed data, 80% is used for training and 20% is for testing.
# 
# `RandomForestClassifier` is used for a "try-all" attempt on an "all-in" set of features.
# The delay category is
# output variable.
# 
# $R^2$ is used to evaluate.
# 
# ## Result ##
# 
# $R^2 \approx 0.72$, however when plotting true vs predicted value, it is clear that for non-small values the prediction is not so good.

# %%
X = df[['longitude','latitude','day_of_year', 'Reason2', 'Reason3','RouteCode','PlaceCode','Train mission']]
y = df[['delay category']]

print(df.shape)
print(X.shape)
print(y.shape)

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(
     estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
     voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
     scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
clf1.fit(X, y)
clf2.fit(X_train, y_train)
eclf.fit(X, y)

# ['longitude','latitude','day_of_year', 'Reason2', 'Reason3','RouteCode','PlaceCode','Train mission']
#print(X_test)
result = pd.DataFrame(clf2.predict(X_test), columns=['predicted'])
result['truevalue'] = y_test['delay category'].to_numpy()
counts = result.groupby(['predicted', 'truevalue']).size().reset_index(name='count')
sns.scatterplot(
    x='predicted',  # x-axis is one category
    y='truevalue',  # y-axis is another category
    size='count',   # size of circles based on count
    hue='count',    # color by count (optional)
    sizes=(0, 500), # set range of circle sizes
    data=counts,
    legend=False
)
counts



# %%
sns.pairplot(df, diag_kind = "kde")
plt.show()



"""Machine Learning-Based Speed Detection System with ADXL345
and GPS Validation"""
"""This project is made by Parsa Karami"""

"""Importing libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


#Loading data
data = pd.read_excel(r'D:\G\Arduino projects\ESP8266\ADXL_GPS_SparkFun_project\datas\rawdata.xlsx')

#checking data & finding null data
df = pd.DataFrame(data)
df.describe()
df.info

#removing 9.81(earth gravity) from Z axis acceleration
df['Z'] = df['Z'] - 9.81

#Handeling outliers with IQR method
#IQR for axes
Q1_x = df['X'].quantile(0.25)
Q3_x = df['X'].quantile(0.75)
IQR_x = Q3_x - Q1_x
lower_bound_x = Q1_x - (1.5 * IQR_x)
upper_bound_x = Q3_x + (1.5 * IQR_x)
#Y axis IQR
Q1_y = df['Y'].quantile(0.25)
Q3_y = df['Y'].quantile(0.75)
IQR_y = Q3_y - Q1_y
lower_bound_y = Q1_y - (2.5 * IQR_y)
upper_bound_y = Q3_y + (1.5 * IQR_y)
# Z axis IQR
Q1_z = df['Z'].quantile(0.25)
Q3_z = df['Z'].quantile(0.75)
IQR_z = Q3_z - Q1_z
lower_bound_z = Q1_z - (2 * IQR_z)
upper_bound_z = Q3_z + (2 * IQR_z)
#Nunmber of outliers
outliers_below_x = df[df['X'] < lower_bound_x]
outliers_above_x = df[df['X'] > upper_bound_x]

num_outliers_below_x = outliers_below_x.shape[0]
num_outliers_above_x = outliers_above_x.shape[0]
total_outliers_x = num_outliers_below_x + num_outliers_above_x
print(total_outliers_x, num_outliers_below_x, num_outliers_above_x)
#Y axis bounderis
outliers_below_y = df[df['Y'] < lower_bound_y]
outliers_above_y = df[df['Y'] > upper_bound_y]

num_outliers_below_y = outliers_below_y.shape[0]
num_outliers_above_y = outliers_above_y.shape[0]
total_outliers_y = num_outliers_below_y + num_outliers_above_y
print(total_outliers_y, num_outliers_below_y, num_outliers_above_y)
#Z axis bounderis
outliers_below_z = df[df['Z'] < lower_bound_z]
outliers_above_z = df[df['Z'] > upper_bound_z]

num_outliers_below_z = outliers_below_z.shape[0]
num_outliers_above_z = outliers_above_z.shape[0]
total_outliers_z = num_outliers_below_z + num_outliers_above_z
print(total_outliers_z, num_outliers_below_z, num_outliers_above_z)
#Illustrating data with outliers boundaries
#1_ X axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(df['X'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_x, color='r', linestyle='--', linewidth=1, label='Q1 (25th percentile)')
plt.axhline(y=Q3_x, color='g', linestyle='--', linewidth=1, label='Q3 (75th percentile)')

plt.axhline(y=lower_bound_x, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1 - 1.5 * IQR)')
plt.axhline(y=upper_bound_x, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3 + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#lower_bound = 2.85 , upper_bound = 1.71

#2_ Y axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(df['Y'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_y, color='r', linestyle='--', linewidth=1, label='Q1_y (25th percentile)')
plt.axhline(y=Q3_y, color='g', linestyle='--', linewidth=1, label='Q3_y (75th percentile)')

plt.axhline(y=lower_bound_y, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_y - 1.5 * IQR)')
plt.axhline(y=upper_bound_y, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_y + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#3_ Z axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(df['Z'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_z, color='r', linestyle='--', linewidth=1, label='Q1_Z (25th percentile)')
plt.axhline(y=Q3_z, color='g', linestyle='--', linewidth=1, label='Q3_Z (75th percentile)')

plt.axhline(y=lower_bound_z, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_z - 1.5 * IQR)')
plt.axhline(y=upper_bound_z, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_z + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
"""Removing outliers"""
filtered_data = df.copy()
print(filtered_data.info())
filtered_data = filtered_data[(filtered_data['Y'] > lower_bound_y) & (filtered_data['Y'] < upper_bound_y)]
filtered_data = filtered_data[(filtered_data['X'] > lower_bound_x) & (filtered_data['X'] < upper_bound_x)]
filtered_data = filtered_data[(filtered_data['Z'] > lower_bound_z) & (filtered_data['Z'] < upper_bound_z)]
print(filtered_data.info())
#X axis
outliers_below_x = filtered_data[filtered_data['X'] < lower_bound_x]
outliers_above_x = filtered_data[filtered_data['X'] > upper_bound_x]

num_outliers_below_x = outliers_below_x.shape[0]
num_outliers_above_x = outliers_above_x.shape[0]
total_outliers_x = num_outliers_below_x + num_outliers_above_x
print(total_outliers_x, num_outliers_below_x, num_outliers_above_x)
#Y axis
outliers_below_y = filtered_data[filtered_data['Y'] < lower_bound_y]
outliers_above_y = filtered_data[filtered_data['Y'] > upper_bound_y]

num_outliers_below_y = outliers_below_y.shape[0]
num_outliers_above_y = outliers_above_y.shape[0]
total_outliers_y = num_outliers_below_y + num_outliers_above_y
print(total_outliers_y, num_outliers_below_y, num_outliers_above_y)
#Z axis
outliers_below_z = filtered_data[filtered_data['Z'] < lower_bound_z]
outliers_above_z = filtered_data[filtered_data['Z'] > upper_bound_z]

num_outliers_below_z = outliers_below_z.shape[0]
num_outliers_above_z = outliers_above_z.shape[0]
total_outliers_z = num_outliers_below_z + num_outliers_above_z
print(total_outliers_z, num_outliers_below_z, num_outliers_above_z)

#Illustrating removed data and the boundaries
#1_ X axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['X'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_y, color='r', linestyle='--', linewidth=1, label='Q1_x (25th percentile)')
plt.axhline(y=Q3_y, color='g', linestyle='--', linewidth=1, label='Q3_x (75th percentile)')

plt.axhline(y=lower_bound_x, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_x - 1.5 * IQR)')
plt.axhline(y=upper_bound_x, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_x + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

#Y axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['Y'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_y, color='r', linestyle='--', linewidth=1, label='Q1_y (25th percentile)')
plt.axhline(y=Q3_y, color='g', linestyle='--', linewidth=1, label='Q3_y (75th percentile)')

plt.axhline(y=lower_bound_y, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_y - 1.5 * IQR)')
plt.axhline(y=upper_bound_y, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_y + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

#Z axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['Z'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_z, color='r', linestyle='--', linewidth=1, label='Q1_Z (25th percentile)')
plt.axhline(y=Q3_z, color='g', linestyle='--', linewidth=1, label='Q3_Z (75th percentile)')

plt.axhline(y=lower_bound_z, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_z - 1.5 * IQR)')
plt.axhline(y=upper_bound_z, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_z + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

"""Removing remaining outliers"""
#here Local Outlier Factor is the method
window_size = 100
k = 70
lof = LocalOutlierFactor(n_neighbors=k)
outlier_indices = []
columns = ['X', 'Y', 'Z']

for item in columns:
  for i in range(0, len(filtered_data[item]), window_size):
    window_data = filtered_data[item].iloc[i:i+window_size]
    window_data_reshaped = window_data.values.reshape(-1, 1)
    window_labels = lof.fit_predict(window_data_reshaped)
    outlier_indices.extend([i + idx for idx, label in enumerate(window_labels) if label == -1])

unique_outlier_indices = list(dict.fromkeys(outlier_indices))

print(len(outlier_indices))
print(len(unique_outlier_indices))
valid_outlier_indices = [idx for idx in unique_outlier_indices if idx in filtered_data.index]
print(len(valid_outlier_indices))
# Optionally, remove the outliers from the DataFrame
filtered_data = filtered_data.drop(valid_outlier_indices).reset_index(drop=True)
filtered_data.info()
#illustrating formated data
#X axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['X'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_y, color='r', linestyle='--', linewidth=1, label='Q1_x (25th percentile)')
plt.axhline(y=Q3_y, color='g', linestyle='--', linewidth=1, label='Q3_x (75th percentile)')

plt.axhline(y=lower_bound_x, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_x - 1.5 * IQR)')
plt.axhline(y=upper_bound_x, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_x + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#Y axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['Y'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_y, color='r', linestyle='--', linewidth=1, label='Q1_y (25th percentile)')
plt.axhline(y=Q3_y, color='g', linestyle='--', linewidth=1, label='Q3_y (75th percentile)')

plt.axhline(y=lower_bound_y, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_y - 1.5 * IQR)')
plt.axhline(y=upper_bound_y, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_y + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#Z axis
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['Z'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_z, color='r', linestyle='--', linewidth=1, label='Q1_Z (25th percentile)')
plt.axhline(y=Q3_z, color='g', linestyle='--', linewidth=1, label='Q3_Z (75th percentile)')

plt.axhline(y=lower_bound_z, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_z - 1.5 * IQR)')
plt.axhline(y=upper_bound_z, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_z + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

"""Feature Engineering"""
#Applying magnitude technique
filtered_data['Magnitude'] = np.sqrt(filtered_data['X']**2 +filtered_data['Y']**2 + filtered_data['Z']**2)
#Applyig difference of acceleration of each axis
filtered_data['jerk_X'] = filtered_data['X'].diff()
filtered_data['jerk_Y'] = filtered_data['Y'].diff()
filtered_data['jerk_Z'] = filtered_data['Z'].diff()

Q1_jerk_x = filtered_data['jerk_X'].quantile(0.25)
Q3_jerk_x = filtered_data['jerk_X'].quantile(0.75)
IQR_jerk_x = Q3_jerk_x - Q1_jerk_x
lower_bound_jerk_x = Q1_jerk_x - (1.5 * IQR_jerk_x)
upper_bound_jerk_x = Q3_jerk_x + (1.5 * IQR_jerk_x)
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
#X axis
plt.plot(filtered_data['jerk_X'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_jerk_x, color='r', linestyle='--', linewidth=1, label='Q1_x (25th percentile)')
plt.axhline(y=Q3_jerk_x, color='g', linestyle='--', linewidth=1, label='Q3_x (75th percentile)')

plt.axhline(y=lower_bound_jerk_x, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_x - 1.5 * IQR)')
plt.axhline(y=upper_bound_jerk_x, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_x + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#Y axis
Q1_jerk_y = filtered_data['jerk_Y'].quantile(0.25)
Q3_jerk_y = filtered_data['jerk_Y'].quantile(0.75)
IQR_jerk_y = Q3_jerk_x - Q1_jerk_x
lower_bound_jerk_y = Q1_jerk_y - (1.5 * IQR_jerk_y)
upper_bound_jerk_y = Q3_jerk_y + (1.5 * IQR_jerk_y)
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['jerk_Y'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_jerk_y, color='r', linestyle='--', linewidth=1, label='Q1_y (25th percentile)')
plt.axhline(y=Q3_jerk_y, color='g', linestyle='--', linewidth=1, label='Q3_y (75th percentile)')

plt.axhline(y=lower_bound_jerk_y, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_y - 1.5 * IQR)')
plt.axhline(y=upper_bound_jerk_y, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_y + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
#Z axis
Q1_jerk_z = filtered_data['jerk_Z'].quantile(0.25)
Q3_jerk_z = filtered_data['jerk_Z'].quantile(0.75)
IQR_jerk_z = Q3_jerk_z - Q1_jerk_z
lower_bound_jerk_z = Q1_jerk_z - (1.5 * IQR_jerk_z)
upper_bound_jerk_z = Q3_jerk_z + (1.5 * IQR_jerk_z)
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['jerk_Z'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_jerk_z, color='r', linestyle='--', linewidth=1, label='Q1_Z (25th percentile)')
plt.axhline(y=Q3_jerk_z, color='g', linestyle='--', linewidth=1, label='Q3_Z (75th percentile)')

plt.axhline(y=lower_bound_jerk_z, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_z - 1.5 * IQR)')
plt.axhline(y=upper_bound_jerk_z, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_z + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

#nubmer of outlier in these new features
#jerk x
outliers_below_jerk_x = filtered_data[filtered_data['jerk_X'] < lower_bound_x]
outliers_above_jerk_x = filtered_data[filtered_data['jerk_X'] > upper_bound_x]

num_outliers_below_jerk_x = outliers_below_jerk_x.shape[0]
num_outliers_above_jerk_x = outliers_above_jerk_x.shape[0]
total_outliers_jerk_x = num_outliers_below_jerk_x + num_outliers_above_jerk_x
print(total_outliers_jerk_x, num_outliers_below_jerk_x, num_outliers_above_jerk_x)
#jerk y
outliers_below_jerk_y = filtered_data[filtered_data['jerk_Y'] < lower_bound_y]
outliers_above_jerk_y = filtered_data[filtered_data['jerk_Y'] > upper_bound_y]

num_outliers_below_jerk_y = outliers_below_jerk_y.shape[0]
num_outliers_above_jerk_y = outliers_above_jerk_y.shape[0]
total_outliers_jerk_y = num_outliers_below_jerk_y + num_outliers_above_jerk_y
print(total_outliers_jerk_y, num_outliers_below_jerk_y, num_outliers_above_jerk_y)
#jerk z
outliers_below_jerk_z = filtered_data[filtered_data['jerk_Z'] < lower_bound_z]
outliers_above_jerk_z = filtered_data[filtered_data['jerk_Z'] > upper_bound_z]

num_outliers_below_jerk_z = outliers_below_jerk_z.shape[0]
num_outliers_above_jerk_z = outliers_above_jerk_z.shape[0]
total_outliers_jerk_z = num_outliers_below_jerk_z + num_outliers_above_jerk_z
print(total_outliers_jerk_z, num_outliers_below_jerk_z, num_outliers_above_jerk_z)
filtered_data = filtered_data[(filtered_data['jerk_X'] > lower_bound_jerk_x) & (filtered_data['jerk_X'] < upper_bound_jerk_y)]
filtered_data = filtered_data[(filtered_data['jerk_Y'] > lower_bound_jerk_y) & (filtered_data['jerk_Y'] < upper_bound_jerk_x)]
filtered_data = filtered_data[(filtered_data['jerk_Z'] > lower_bound_jerk_z) & (filtered_data['jerk_Z'] < upper_bound_jerk_z)]
filtered_data.info()

Q1_magnitude = filtered_data['Magnitude'].quantile(0.25)
Q3_magnitude = filtered_data['Magnitude'].quantile(0.75)
IQR_magnitude = Q3_magnitude - Q1_magnitude
lower_bound_magnitude = Q1_magnitude - (1 * IQR_magnitude)
upper_bound_magnitude = Q3_magnitude + (1.5 * IQR_magnitude)
plt.figure(figsize=(10, 6))

# Plot the data (e.g., as a scatter plot or line plot)
plt.plot(filtered_data['Magnitude'], marker='o', linestyle='-', label='Data')

plt.axhline(y=Q1_magnitude, color='r', linestyle='--', linewidth=1, label='Q1_magnitude (25th percentile)')
plt.axhline(y=Q3_magnitude, color='g', linestyle='--', linewidth=1, label='Q3_magnitude (75th percentile)')

plt.axhline(y=lower_bound_magnitude, color='b', linestyle='--', linewidth=1, label='Lower bound (Q1_magnitude - 1.5 * IQR)')
plt.axhline(y=upper_bound_magnitude, color='b', linestyle='--', linewidth=1, label='Upper bound (Q3_magnitude + 1.5 * IQR)')

plt.title('Data with IQR Boundaries')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

outliers_below_magnitude = filtered_data[filtered_data['Magnitude'] < lower_bound_magnitude]
outliers_above_magnirude = filtered_data[filtered_data['Magnitude'] > upper_bound_magnitude]

num_outliers_below_magnitude = outliers_below_magnitude.shape[0]
num_outliers_above_magnitude = outliers_above_magnirude.shape[0]
num_total_outliers = num_outliers_below_magnitude + num_outliers_above_magnitude
print(num_total_outliers, num_outliers_below_magnitude, num_outliers_above_magnitude)

filtered_data = filtered_data[(filtered_data['Magnitude'] > lower_bound_magnitude) & (filtered_data['Magnitude'] < upper_bound_magnitude)]
filtered_data.info()

#checking correlation of new features
filtered_data.corr()

#visualizing the correlation of the dataset with the seaborn library.
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(filtered_data.corr(), annot = True, linewidths = 0.5, linecolor = "black", fmt = ".4f", ax = ax)
plt.show()
"""Train Test split"""
y = filtered_data['Speed']
x = filtered_data.drop(columns=['Speed', 'Latitude', 'Longitude'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x.head()
#Making new label to see we can find the true speed more accurately or the difference of speeds
y_diff_train = y_train.diff()
y_diff_train.fillna(y_diff_train.mean(), inplace=True)
#checking the shape of splitted data
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

"""Machine learning Models"""
#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_diff_train)
speed_predictions = lin_reg.predict(x_train)
lin_rmse = mean_squared_error(y_train, speed_predictions, squared=False)
lin_rmse
lin_reg_rmses = cross_val_score(lin_reg, x_train, y_train, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_reg_rmses).describe()

#Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)
speed_tree_predictions = tree_reg.predict(x_train)
tree_rmse = mean_squared_error(y_train, speed_tree_predictions, squared=False)
tree_rmse

tree_reg_rmse = cross_val_score(tree_reg, x_train, y_train, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_reg_rmse).describe()

#Random Forest Regressor
forest_reg = RandomForestRegressor()
forest_reg.fit(x_train, y_train)
speed_forest_predictions = forest_reg.predict(x_train)
forest_rmse = mean_squared_error(y_train, speed_forest_predictions, squared=False)
forest_rmse

forest_reg_rmse = cross_val_score(forest_reg, x_train, y_train, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(forest_reg_rmse).describe()

#XGboost
xgb_model = XGBRegressor()
search_sapce = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 9],
    "gamma": [0.01, 0.1],
    "learning_rate": [0.03, 0.1]
}

gsx = GridSearchCV(estimator= xgb_model,
                   param_grid = search_sapce,
                   scoring=["r2", "neg_root_mean_squared_error"],
                   refit = "r2",
                   cv=10,
                   verbose=4)
gsx.fit(x_train, y_train)
#checking xgboost numbers
print(gsx.best_estimator_)
print(gsx.best_params_)
print(gsx.best_score_)

xg = XGBRegressor(gamma=0.1, learning_rate=0.03, max_depth=6, n_estimators=100)
xg.fit(x_train, y_train)
speed_xg_prediction = xg.predict(x_train)
xg_rmse = mean_squared_error(y_train, speed_forest_predictions, squared=False)
xg_rmse
xg_reg_rmse = cross_val_score(xg, x_train, y_train, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(xg_reg_rmse).describe()

"""Neural Network"""
def linear_model(input_shape):
  model = Sequential()
  model.add(Dense(128, input_dim=input_shape, activation='relu')) #it was relu at the best one
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='tanh')) # it was tanh at the best one
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='linear'))  # Single output for regression with linear activation

  model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
  return model

input_dimension = x_train.shape[1]
model = linear_model(input_dimension)
model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_train, y_train))

y_train_pred = model.predict(x_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f'Train RMSE: {train_rmse}')
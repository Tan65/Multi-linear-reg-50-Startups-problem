# Multi-linear-reg-50-Startups-problem
Prepared a prediction model for profit of 50_startups data. Do transformations for getting better predictions of profit and make a table containing R^2 value for each prepared model.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("50_Startups.csv")

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Univariate Analysis: Histograms
data.hist(figsize=(15, 10), bins=20)
plt.suptitle("Histograms of Numeric Variables", fontsize=16)
plt.show()

# Bivariate Analysis: Pairplot
sns.pairplot(data)
plt.suptitle("Pairplot of Numeric Variables", fontsize=16)
plt.show()

# Multivariate Analysis: Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Perform transformations on the data
# Convert the categorical feature 'State' to numerical using LabelEncoder
label_encoder = LabelEncoder()
data['State'] = label_encoder.fit_transform(data['State'])

# Split the data into features (X) and target variable (y)
X = data.drop('Profit', axis=1)
y = data['Profit']

# Create a table to store R-squared values for each prepared model
r2_table = pd.DataFrame(columns=['Transformation', 'R-squared'])

# Perform different transformations and evaluate the models
transformations = [
    ('Original', X),
    ('Log Transformation', X.apply(lambda x: np.log1p(x))),
    ('Square Root Transformation', X.apply(lambda x: np.sqrt(x))),
]

for transformation_name, transformed_X in transformations:
    # Split the transformed data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2, random_state=42)

    # Create a linear regression model and fit the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the R-squared value
    r2 = r2_score(y_test, y_pred)

    # Add the R-squared value to the table
    r2_table = r2_table.append({'Transformation': transformation_name, 'R-squared': r2}, ignore_index=True)

# Display the R-squared values for each prepared model
print(r2_table)

# Analyze and Predict Housing Prices by Using Python
The dataset file is attached in this repository.
### Data Dictionary:
| Variable        | Description                                                                                  | Content type          |
|-----------------|----------------------------------------------------------------------------------------------|-----------------------|
| id              | A notation for a house                                                                       | integer               |
| date            | Date house was sold                                                                          | date                  |
| price           | Price is prediction target                                                                   | float                 |
| bedrooms        | Number of bedrooms                                                                           | integer               |
| bathrooms       | Number of bathrooms                                                                          | float                 |
| sqft_living     | Square footage of the home                                                                   | integer               |
| sqft_lot        | Square footage of the lot                                                                    | integer               |
| floors          | Total floors (levels) in house                                                               | integer               |
| waterfront      | House which has a view to a waterfront                                                       | integer (0 or 1)      |
| view            | Has been viewed                                                                              | integer (0 to 4)      |
| condition       | How good the condition is overall                                                            | integer (1 to 5)      |
| grade           | Overall grade given to the housing unit, based on King County grading system                  | integer (1 to 13)     |
| sqft_above      | Square footage of house apart from basement                                                  | integer               |
| sqft_basement   | Square footage of the basement                                                               | integer               |
| yr_built        | Built Year                                                                                   | integer               |
| yr_renovated    | Year when house was renovated                                                                | integer               |
| zipcode         | Zip code                                                                                     | integer               |
| lat             | Latitude coordinate                                                                          | float                 |
| long            | Longitude coordinate                                                                         | float                 |
| sqft_living15   | Living room area in 2015 (implies some renovations; might or might not affect lot size area)  | integer               |
| sqft_lot15      | Lot size area in 2015 (implies some renovations)                                             | integer               |

# Objectives:
I will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on.

## 1. Data Preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

file_path = r"D:\Hien_Others\1. Learning\IBM Data Analyst\kc_house_data_NaN.csv"
df = pd.read_csv(file_path)
print(df.head())
```

![image](https://github.com/user-attachments/assets/5a70b9fd-71d5-4f22-a03c-a9c5f54337bb)

**Drop the columns "id" and "Unnamed: 0"**
```python
df.drop(["Unnamed: 0", "id"], axis=1,inplace=True)
```
**Find duplicated rows, and remove them (if any)**
```python
print(df[df.duplicated()])
```
_Result: Empty DataFrame that means no duplicated row_

**Info of dataframe after removing 2 columns "id" and "Unnamed: 0"**
```python
df.info()
```

![image](https://github.com/user-attachments/assets/2c571c21-fb05-4394-b6b9-078f36490678)

_Result: "bathrooms" and "bedrooms" missing value_

How many values of "bathrooms" and "bedrooms" are missing?
```python
print ("Number of missing values for bathrooms:", df["bathrooms"].isnull().sum())
print("Number of missing values for bedrooms:", df["bedrooms"].isnull().sum())
```

![image](https://github.com/user-attachments/assets/e53aa43c-cbb3-40bd-955d-fd642c9ac106)

**Replace the missing values of the column 'bedrooms' with the mean**
```python
mean_bedrooms = df["bedrooms"].mean()
df["bedrooms"] = df["bedrooms"].replace(np.nan, mean_bedrooms)
```
**Replace the missing values of the column 'bathrooms' with the mean**
```python
mean_bathrooms = df["bathrooms"].mean()
df["bathrooms"] = df["bathrooms"].replace(np.nan, mean_bathrooms)
```
**Crosscheck after replacing values**
```python
print("Number of missing values for bathrooms:", df["bathrooms"].isnull().sum())
print("Number of missing values for bedrooms:", df["bedrooms"].isnull().sum())
```

![image](https://github.com/user-attachments/assets/59e150c7-a17b-41f0-be66-ab22bb02dc1b)

and now it is the cleaned data:
![image](https://github.com/user-attachments/assets/788d288e-9e65-490d-80fc-806dc83ade77)

## 2. Exploratory Data Analysis (EDA)
**Find the feature other than price that is most correlated with price**
```python
# Only use numeric data
df_numeric = df._get_numeric_data()
print(df_numeric.corr()["price"].sort_values())
```

![image](https://github.com/user-attachments/assets/4d58fb05-bd66-4b7e-b2b3-85e78f4aa651)

_Result: sqft_living is the most correlated with price with the Pearson Correlation coefficient is 0.702035_

**Plot correlation between "sqft_living" and "price"**
```python
sns.regplot(x=df["sqft_living"], y=df["price"], data=df)
plt.tight_layout()
plt.ylim(0,)
plt.show()
plt.close()
```

![image](https://github.com/user-attachments/assets/699e6566-3509-4639-afc1-15ee44def770)

## 3. Model Development
**Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2.**
```python
from sklearn.linear_model import LinearRegression

X = df[["sqft_living"]]
Y = df["price"]
lm = LinearRegression()
lm.fit(X,Y)
r_square = lm.score(X,Y)
print("R^2 of sqft_lving and price is:", r_square)
```
_Result: R^2: 49.29%_

**Fit a linear regression model to predict the 'price' using the list of features: ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]**
```python
from sklearn.linear_model import LinearRegression

Z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
Y = df["price"]
lm1 = LinearRegression()
lm1.fit(Z,Y)
print("R^2:", lm1.score(Z,Y))
```
_Result: R^2: 65.76%_

**Create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2**
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

Z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
Y = df["price"]

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
print("R^2:", r2_score(Y, ypipe))
```
_Result: R^2: 75.13%_

## 4. Model Refinement
**Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data**
```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
yhat = RigeModel.predict(x_test)
print("R^2:", r2_score(y_test,yhat))
```
_Result: R^2: 64.79%_

**Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

poly = PolynomialFeatures(degree=2, include_bias=False) # include_bias = False is to fit a model without an intercept

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
RigeModel = Ridge(alpha=1)
RigeModel.fit(x_train_poly, y_train)
yhat = RigeModel.predict(x_test_poly)
print("R^2:", r2_score(y_test, yhat))
```
_Result: R^2: 69.97%_
## Conclusion:

With quite good R^2 = 69.97% I've created a training pipeline using StandardScaler(), PolynomialFeatures(), and LinearRegression() to predict charges, the next step is to deploy and use this trained model in real-work scenarios.

Thank you for stopping by, and I'm pleased to connect with you, my new friend!

Please do not forget to FOLLOW and star ⭐ the repository if you find it valuable.

I wish you a day filled with happiness and energy!

Warm regards,

Hien Moon

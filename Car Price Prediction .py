#!/usr/bin/env python
# coding: utf-8

# In[1]:


#for preprocesing of data
import pandas as pd 
#for algebraic operations 
import numpy as np


# In[2]:


df=pd.read_csv('C:/Users/osa/Downloads/car_resale_prices (3).csv')


# In[3]:


df=df.drop(df.columns[0],axis=1)
df


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


#replacing null values by taking years from full_name of the car
df['registered_year'] = df['full_name'].apply(lambda x: x.split(' ')[0]).astype(float)
df['registered_year'].isnull().sum()


# In[7]:


#dropping the null values
df.dropna(inplace = True)
df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df.head()


# In[10]:


num_cols = ['seats', 'registered_year']


# In[11]:


#converting datatype of resale_price
def format_price(resale_price):
    split = resale_price.split(' ')
    k = len(split)
    if(k == 2):
        return float(str(split[1]).replace(',', ''))
    elif(split[-1] == 'Lakh'):
        return float(str(split[1]))*100000
    return float(str(split[1]))*10000000

df['num_price'] = df['resale_price'].apply(format_price)
num_cols.append('num_price')


# In[12]:


df.head()


# In[13]:


def format_kms(kms_driven):
    if kms_driven.split(' ')[1] != 'Kms':
        print(kms_driven.split(' ')[1])
    return float(str(kms_driven.split(' ')[0]).replace(',', ''))

df['num_kms'] = df['kms_driven'].apply(format_kms)
num_cols.append('num_kms')
df.head(2)


# In[14]:


def format_engine(engine_capacity):
    if engine_capacity.split(' ')[1] != 'cc':
        print(engine_capacity.split(' ')[1])
    return float(str(engine_capacity.split(' ')[0]))

df['num_engine'] = df['engine_capacity'].apply(format_engine)
num_cols.append('num_engine')
df.head(2)


# In[15]:


#check if unit other than bhp exists

df['max_power'].apply(lambda x: x[-3:]).value_counts()


# In[16]:


#dropping the rows with units other than bhp
df = df.drop(df[df['max_power'].apply(lambda x: x[-3:]).str.lower() != 'bhp'].index)


# In[17]:


#type casting of max_power
df['max_power'] = df['max_power'].apply(lambda x : x[:-3]).astype(float)

num_cols.append('max_power')

df.head(2)


# In[18]:


df.info()


# In[19]:


#verifying the units of mileage 
df['mileage'].apply(lambda x: x.split(' ')[-1]).value_counts()


# In[20]:


#dropping the mileage values other than kmpl
df.drop(df[df['mileage'].apply(lambda x: x.split(' ')[-1]).str.lower() != 'kmpl'].index, inplace = True)


# In[21]:


#re-verifying the units of mileage
df['mileage'].apply(lambda x: x.split(' ')[-1]).value_counts()


# In[22]:


#type casting of mileage
df['mileage'] = df['mileage'].apply(lambda x: x.split(' ')[0]).astype(float)

num_cols.append('mileage')

df.head(2)


# In[23]:


df.info()


# In[24]:


num_cols


# In[25]:


num_encoded = df[num_cols].copy()
num_encoded.head()


# In[26]:


#by this numerical cleaning is completed



# In[ ]:


#categorical encoding


# In[33]:


df.head(2)


# In[34]:


# finding frequency of each city 
df['city'].value_counts()


# In[35]:


# finding frequency of body type
df = df.groupby('body_type').filter(lambda x: len(x) > 45)
df['body_type'].value_counts()


# In[36]:


#finding the count of each fuel type
df['fuel_type'].value_counts()


# In[37]:


#finding the count of each owner type
df['owner_type'].value_counts()


# In[38]:


#finding the occurance of each value in transmission type
df['transmission_type'].value_counts()


# In[39]:


#finding the occurance of each value in insurance
df['insurance'].value_counts()


# In[40]:


#filtering to delete elements with less frequency
df= df.groupby('insurance').filter(lambda x: len(x) > 500)
df['insurance'].value_counts()


# In[41]:


categorical_columns = ['city', 'body_type', 'fuel_type', 'owner_type', 'insurance', 'transmission_type']


# In[42]:


#introducing dummy variables
from sklearn.preprocessing import OneHotEncoder

cat_encoded = pd.get_dummies(df[categorical_columns])
cat_encoded.head()


# In[43]:


#finding number of unique models from feature full_name
df['model'] = df['full_name'].apply(lambda x: x.split(' ')[1] + " " + x.split(' ')[2])

df['model'].nunique()


# In[44]:


#introducing dummy variable for the variable car
model_encoded = pd.get_dummies(df['model'])


# In[45]:


model_encoded.head()


# In[46]:


label_encoded = df[['model']]
label_encoded


# In[47]:


#merging the data
df = pd.concat([df[num_encoded.columns], cat_encoded, model_encoded], axis = 1)
df.head()


# In[48]:


#using visualisation techniques to explore the data
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(3, 3, figsize=(24, 24))
k = 0
columns = list(num_encoded.columns)

for i in range(3):
    for j in range(3):
        if k < len(columns):
            sns.histplot(df[columns[k]], ax=ax1[i][j], color='orange', kde=True, stat="frequency", label=columns[k])
            ax1[i][j].set_title(columns[k])  # Add column name as subplot title
            ax1[i][j].legend()  # Add legend to the subplot
            
            # Add scale annotations
            min_val = df[columns[k]].min()
            max_val = df[columns[k]].max()
            ax1[i][j].annotate(f'Min: {min_val:.2f}\nMax: {max_val:.2f}', 
                               xy=(0.7, 0.8), xycoords='axes fraction',
                               fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', alpha=0.1))

            k += 1
        else:
            # If there are no more columns, remove the remaining subplots
            fig.delaxes(ax1[i][j])

# Set common xlabel and ylabel for all subplots
fig.text(0.5, 0.04, 'X-axis Label', ha='center', va='center')
fig.text(0.06, 0.5, 'Y-axis Label', ha='center', va='center', rotation='vertical')

plt.show()



# In[49]:


#checking  for the outliers
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="whitegrid")
fig, ax1 = plt.subplots(3, 3, figsize=(16, 16))
col = 0

for i in range(3):
    for j in range(3):
        # Check if there are more columns to plot
        if col < len(columns):
            sns.boxplot(y=df[columns[col]], ax=ax1[i][j], orient='v')
            ax1[i][j].set_title(columns[col])  # Add column name as subplot title

            # Add scale annotations
            min_val = df[columns[col]].min()
            max_val = df[columns[col]].max()
            ax1[i][j].annotate(f'Min: {min_val:.2f}\nMax: {max_val:.2f}', 
                               xy=(0.7, 0.8), xycoords='axes fraction',
                               fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', alpha=0.1))

            col += 1
        else:
            # If there are no more columns, remove the remaining subplots
            fig.delaxes(ax1[i][j])

# Set common xlabel and ylabel for all subplots
fig.text(0.5, 0.04, 'X-axis Label', ha='center', va='center')
fig.text(0.06, 0.5, 'Y-axis Label', ha='center', va='center', rotation='vertical')

plt.show()


# In[50]:


#removing outliers by quartiles and inner quartiles
skewed_cols = ['seats', 'num_kms', 'num_engine', 'max_power', 'mileage']

for col in skewed_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df.loc[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    df = df.drop(outliers.index)


# In[51]:


fig = plt.figure(figsize = (8,4))
plt.xticks(rotation = 45)
plt.xlabel('numeric features')
plt.ylabel('frequency')
plt.title('Boxplot for Skewed Columns')
sns.set(style="whitegrid")
sns.boxplot(data=df[skewed_cols])
plt.show()


# In[69]:


s=df['num_price']
q=df['num_kms']
r=df['num_engine']
t=df['max_power']
a=df['mileage']
#checking for normality
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Defining the variables
variables = {'num_price': s, 'num_kms': q, 'num_engine': r, 'max_power': t, 'mileage': a}

# Iterate over each variable
for var_name, data in variables.items():
    # Subsampling to reduce sample size
    sample_size = 5000  # Choose an appropriate sample size
    data_subsample = np.random.choice(data, sample_size, replace=False)

    # Visual Method
    plt.figure(figsize=(12, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data_subsample, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {var_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # QQ Plot
    plt.subplot(1, 2, 2)
    stats.probplot(data_subsample, dist="norm", plot=plt)
    plt.title(f'QQ Plot of {var_name}')

    plt.tight_layout()
    plt.show()

    # Statistical Test
    statistic, p_value = stats.shapiro(data_subsample)
    print(f"Shapiro-Wilk Test for {var_name}:")
    print("Statistic:", statistic)
    print("p-value:", p_value)
    alpha = 0.05
    if p_value > alpha:
        print(f"The data for {var_name} appears to be normally distributed (fail to reject H0)")
    else:
        print(f"The data for {var_name} does not appear to be normally distributed (reject H0)")


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
normalized_s = scaler.fit_transform(s.values.reshape(-1, 1))
normalized_q = scaler.fit_transform(q.values.reshape(-1, 1))
normalized_r = scaler.fit_transform(r.values.reshape(-1, 1))
normalized_t = scaler.fit_transform(t.values.reshape(-1, 1))

# Plot histograms of the normalized data
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(normalized_s, bins=30, color='red', kde=True)
plt.title('Normalized s')

plt.subplot(2, 2, 2)
sns.histplot(normalized_q, bins=30, color='green', kde=True)
plt.title('Normalized q')

plt.subplot(2, 2, 3)
sns.histplot(normalized_r, bins=30, color='blue', kde=True)
plt.title('Normalized r')

plt.subplot(2, 2, 4)
sns.histplot(normalized_t, bins=30, color='orange', kde=True)
plt.title('Normalized t')

plt.tight_layout()
plt.show()


# In[71]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit and transform the data
standardized_s = scaler.fit_transform(s.values.reshape(-1, 1))
standardized_q = scaler.fit_transform(q.values.reshape(-1, 1))
standardized_r = scaler.fit_transform(r.values.reshape(-1, 1))
standardized_t = scaler.fit_transform(t.values.reshape(-1, 1))

# Plot histograms of the standardized data
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(standardized_s, bins=30, color='red', kde=True)
plt.title('Standardized s')

plt.subplot(2, 2, 2)
sns.histplot(standardized_q, bins=30, color='green', kde=True)
plt.title('Standardized q')

plt.subplot(2, 2, 3)
sns.histplot(standardized_r, bins=30, color='blue', kde=True)
plt.title('Standardized r')

plt.subplot(2, 2, 4)
sns.histplot(standardized_t, bins=30, color='orange', kde=True)
plt.title('Standardized t')

plt.tight_layout()
plt.show()



# In[52]:


#finding correlation between price and numerical features
df[num_encoded.columns].corr()['num_price'].sort_values(ascending = False)


# In[53]:


df.describe()


# In[54]:


counts = [10544,5060,307,19,4]
fuelType = ("Petrol","Diesel","CNG","LPG","Electric")
index = np.arange(len(fuelType))

# index = X axis
# counts = Height of the bars

plt.bar(index, counts, color=['red', 'blue', 'cyan','green','yellow'])

# Title and labels.

plt.title("Bar plot of fuel types")
plt.xlabel("Fuel Types")
plt.ylabel("Frequency")

# index - Set the location of the xticks
# fuelType - Set the labels of the xticks

plt.xticks(index,fuelType,rotation = 90)
plt.show()


# In[55]:


#visulazing through heatmaps
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Check if all specified columns exist in the DataFrame
required_columns = ['num_engine', 'mileage', 'registered_year', 'num_price','num_kms','max_power']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"{col} is not present in the DataFrame.")

# Create a correlation matrix
correlation_matrix = df[required_columns].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, cbar_kws={'label': 'Correlation'})

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Heatmap')

plt.show()







# In[56]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and target variable in training data
X_train = train_data.drop('num_price', axis=1)
Y_train = train_data['num_price']

# Separate features and target variable in testing data
X_test = test_data.drop('num_price', axis=1)
Y_test = test_data['num_price']

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, Y_train)

#Evaluate the model on the training set
train_predictions=model.predict(X_train)
train_mse=mean_squared_error(Y_train,train_predictions)

#Evaluate the model on the test set
test_predictions=model.predict(X_test)
test_mse=mean_squared_error(Y_test,test_predictions)


#print model evaluation metrics
print(f"Training Mean Squared Error : {train_mse}")
print(f"Test Mean Squared Error : {test_mse}")




# In[ ]:


#The model seems to be well fitted


# In[57]:


from sklearn.tree import DecisionTreeRegressor

# Initialize the decision tree regressor model
decision_tree_model_c45 = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
decision_tree_model_c45.fit(X_train, Y_train)

# Evaluate the model on the training set
train_predictions_c45 = decision_tree_model_c45.predict(X_train)
train_mse_c45 = mean_squared_error(Y_train, train_predictions_c45)

# Evaluate the model on the test set
test_predictions_c45 = decision_tree_model_c45.predict(X_test)
test_mse_c45 = mean_squared_error(Y_test, test_predictions_c45)

# Print model evaluation metrics
print("Decision Tree Model (C4.5 algorithm):")
print(f"Training Mean Squared Error: {train_mse_c45}")
print(f"Test Mean Squared Error: {test_mse_c45}")
from sklearn.metrics import r2_score
import math

# Calculate R-squared for training and test sets
train_r2_c45 = r2_score(Y_train, train_predictions_c45)
test_r2_c45 = r2_score(Y_test, test_predictions_c45)

# Calculate RMSE for training and test sets
train_rmse_c45 = math.sqrt(train_mse_c45)
test_rmse_c45 = math.sqrt(test_mse_c45)

# Print model evaluation metrics
print("Decision Tree Model (C4.5 algorithm):")
print(f"Training R-squared: {train_r2_c45:.4f}")
print(f"Test R-squared: {test_r2_c45:.4f}")
print(f"Training Root Mean Squared Error (RMSE): {train_rmse_c45:.4f}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse_c45:.4f}")




# In[ ]:


#model seems to be well fitted using rsquare


# In[58]:


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting regressor model
gradient_boosting_model = GradientBoostingRegressor(random_state=42)

# Train the model on the training data
gradient_boosting_model.fit(X_train, Y_train)

# Evaluate the model on the training set
train_predictions_gb = gradient_boosting_model.predict(X_train)
train_mse_gb = mean_squared_error(Y_train, train_predictions_gb)

# Evaluate the model on the test set
test_predictions_gb = gradient_boosting_model.predict(X_test)
test_mse_gb = mean_squared_error(Y_test, test_predictions_gb)

# Print model evaluation metrics
print("Gradient Boosting Model:")
print(f"Training Mean Squared Error: {train_mse_gb}")
print(f"Test Mean Squared Error: {test_mse_gb}")
from sklearn.metrics import r2_score
import math

# Calculate R-squared for training and test sets
train_r2_gb = r2_score(Y_train, train_predictions_gb)
test_r2_gb = r2_score(Y_test, test_predictions_gb)

# Calculate RMSE for training and test sets
train_rmse_gb = math.sqrt(train_mse_gb)
test_rmse_gb = math.sqrt(test_mse_gb)

# Print model evaluation metrics
print("Gradient Boosting Model:")
print(f"Training R-squared: {train_r2_gb:.4f}")
print(f"Test R-squared: {test_r2_gb:.4f}")
print(f"Training Root Mean Squared Error (RMSE): {train_rmse_gb:.4f}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse_gb:.4f}")



# In[ ]:


#model seems to be well fitted using rsquare



# In[62]:


from sklearn.linear_model import LinearRegression

# Instantiate the Linear Regression model
lin_reg = LinearRegression()

# Train the model on the training data
lin_reg.fit(X_train, Y_train)


# In[63]:


#finding accuracy for linear regresion
from sklearn.metrics import r2_score

# Make predictions on the training set
train_predictions = lin_reg.predict(X_train)

# Calculate R-squared score for training set
train_r2 = r2_score(Y_train, train_predictions)

# Print the R-squared score for training set
print("Training R-squared score:", train_r2)

# Make predictions on the test set
test_predictions = lin_reg.predict(X_test)

# Calculate R-squared score for test set
test_r2 = r2_score(Y_test, test_predictions)

# Print the R-squared score for test set
print("Test R-squared score:", test_r2)


# In[60]:


#finding accuracy through decision tree (c4.5)
from sklearn.metrics import r2_score

# Calculate R-squared for training and test sets
train_r2_c45 = r2_score(Y_train, train_predictions_c45)
test_r2_c45 = r2_score(Y_test, test_predictions_c45)

# Print R-squared scores
print("Decision Tree Model (C4.5 algorithm) R-squared scores:")
print(f"Training R-squared: {train_r2_c45:.4f}")
print(f"Test R-squared: {test_r2_c45:.4f}")


# In[61]:


from sklearn.metrics import r2_score, mean_squared_error

# Calculate R-squared for training and test sets
train_r2_gb = r2_score(Y_train, train_predictions_gb)
test_r2_gb = r2_score(Y_test, test_predictions_gb)

# Calculate Mean Squared Error (MSE) for training and test sets
train_mse_gb = mean_squared_error(Y_train, train_predictions_gb)
test_mse_gb = mean_squared_error(Y_test, test_predictions_gb)

# Print R-squared and MSE scores
print("Gradient Boosting Model:")
print(f"Training R-squared: {train_r2_gb:.4f}")
print(f"Test R-squared: {test_r2_gb:.4f}")
print(f"Training Mean Squared Error (MSE): {train_mse_gb:.4f}")
print(f"Test Mean Squared Error (MSE): {test_mse_gb:.4f}")


# In[64]:


#comparing all the methods
import matplotlib.pyplot as plt
import numpy as np

# Calculate R-squared scores for Linear Regression
lin_train_r2 = 0.9019  
lin_test_r2 = 0.9109 

# Calculate R-squared scores for Decision Tree (C4.5)
dt_train_r2 = 0.9997   
dt_test_r2 = 0.8733    

# Calculate R-squared scores for Gradient Boosting
gb_train_r2 = 0.9234    
gb_test_r2 = 0.9238    

# Create lists of R-squared scores for training and testing
train_r2_scores = [lin_train_r2, dt_train_r2, gb_train_r2]
test_r2_scores = [lin_test_r2, dt_test_r2, gb_test_r2]

# Models' names
models = ['Linear Regression', 'Decision Tree (C4.5)', 'Gradient Boosting']

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = np.arange(len(train_r2_scores))
r2 = [x + bar_width for x in r1]

# Create bar plot
plt.bar(r1, train_r2_scores, color='b', width=bar_width, edgecolor='grey', label='Training')
plt.bar(r2, test_r2_scores, color='r', width=bar_width, edgecolor='grey', label='Testing')

# Add labels, title, and legend
plt.xlabel('Models', fontweight='bold')
plt.ylabel('R-squared Score', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(train_r2_scores))], models)
plt.title('Training and Testing R-squared Scores for Different Models')
plt.legend()

# Show plot
plt.show()


# In[65]:


#comparing actual and predicted prices thriugh gradient boosting
new_data_features=df.drop('num_price',axis=1)
predicted_prices=gradient_boosting_model.predict(new_data_features)
comparison_df=pd.DataFrame({'Actual':df['num_price'],'predicted':predicted_prices.astype(int)})
print(comparison_df)


# In[66]:


#comparing actual and predicted through frequency histogram 
import matplotlib.pyplot as plt

# Prepare the features of the new data
new_data_features = df.drop('num_price', axis=1)

# Make predictions on the new data
predicted_prices = gradient_boosting_model.predict(new_data_features)

# Plot frequency histograms for actual and predicted prices
plt.figure(figsize=(10, 6))
plt.hist(df['num_price'], bins=50, color='blue', alpha=0.5, label='Actual', density=True)
plt.hist(predicted_prices, bins=50, color='red', alpha=0.5, label='Predicted', density=True)
plt.xlabel('num_price')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Actual vs. Predicted num_price')
plt.legend()
plt.grid(True)
plt.show()


# In[67]:


import seaborn as sns

# Prepare the features of the new data
new_data_features = df.drop('num_price', axis=1)

# Make predictions on the new data
predicted_prices = gradient_boosting_model.predict(new_data_features)

# Plot frequency curves for actual and predicted prices
plt.figure(figsize=(10, 6))
sns.kdeplot(df['num_price'], color='blue', label='Actual', linewidth=2)
sns.kdeplot(predicted_prices, color='red', label='Predicted', linewidth=2)
plt.xlabel('num_price')
plt.ylabel('Density')
plt.title('Frequency Distribution of Actual vs. Predicted num_price')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





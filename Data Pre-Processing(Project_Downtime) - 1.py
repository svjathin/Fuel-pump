ṇ

##### import throught pip install #######

import pandas as pd # data manipulation
import numpy as np ## numerical calculation
from sqlalchemy import create_engine
from urllib.parse import quote 
from getpass import getpass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sweetviz # autoEDA
import matplotlib.pyplot as plt # data visualization
from sqlalchemy import create_engine # connect to SQL database
from feature_engine.outliers import Winsorizer
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
import scipy.stats as stats
import pylab
from scipy import stats
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv(r"C:\Users\priya\Downloads\Data Set (3)\Machine_Downtime_Project_Dataset.csv")
numeric_features = df.select_dtypes(exclude = ['object']).columns

numeric_features

catagorical_feature = df.select_dtypes(include = ['object']).columns

catagorical_feature

numerical_col = pd.DataFrame(df[numeric_features])

categorical_col = pd.DataFrame(df[catagorical_feature])

## converting data into float #######

numerical_col = numerical_col.astype('float32')
numerical_col.dtypes
print(numerical_col)

## missing value count ####

numerical_col.isna().sum()

categorical_col.isna().sum()

## duplicate value in row #####

numerical_col.duplicated()
numerical_col.duplicated().sum()

### duplicate column ######

numerical_col.corr()

### mean imputatation ########

mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

numerical_col.mean()

numerical_col1 = pd.DataFrame(mean_imputer.fit_transform(numerical_col))

numerical_col1.isna().sum()

new_column_names = ['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure',
                    'Coolant_Temperature', 'Hydraulic_Oil_Temperature',
                    'Spindle_Bearing_Temperature', 'Spindle_Vibration',
                    'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting']

###### Replace existing column names with new column names ########
numerical_col1.columns = new_column_names

print(numerical_col)

###### outlier treatment ###############
numerical_col1.dtypes

########## Let's find outliers in numerical_col1 ##########

numerical_col1.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  


# Detection of outliers (find limits for salary based on IQR)
IQR = numerical_col1.quantile(0.75) - numerical_col1.quantile(0.25)

lower_limit = numerical_col1.quantile(0.25) - (IQR * 1.5)
upper_limit = numerical_col1.quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5, 
                          variables = ['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting'])

numerical_col2 = winsor_iqr.fit_transform(numerical_col1[['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting']])

## checking outlier #######
numerical_col2.plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()  
# *No outliers observed*

### checking data is normally distributed or not by histogram and q - q plot################
### 1) histogram -- 
fig=plt.figure(figsize=(10,20))
axd=fig.gca()
numerical_col2.hist(ax= axd)
plt.show()

####  pairplots to get an intuition of potential correlations ######

sns.pairplot(numerical_col2[['Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 'Spindle_Speed', 'Voltage', 'Torque', 'Cutting']], diag_kind="kde")

######### Normal Quantile-Quantile Plot ######

# Assuming 'numerical_col2' is your DataFrame and q-q plot for entire numerical_col2 normarical dataset.
for column in numerical_col2.columns:
    sm.qqplot(numerical_col2[column], line='s')
    plt.title(f'Q-Q Plot for {column}')
    plt.show()

##### only Coolent_temprature and cuuting is not normal so we are using diffrant transform technique to normal data #########

#### Checking Coolent_temprature data is normally distributed or not #####
stats.probplot(numerical_col2.Coolant_Temperature, dist = "norm", plot = pylab)

## doing the transformation of Coolent_temprature by exponential transformation and then we see the data is normally distriuted.

stats.probplot(np.exp(numerical_col2.Coolant_Temperature), dist = "norm", plot = pylab)

##### Checking cutting data is normally distributed or not ########

stats.probplot(numerical_col2.Cutting, dist = "norm", plot = pylab)

######### Transformation to make cutting variable normal#############################

stats.probplot(np.sqrt(numerical_col2.Cutting), dist = "norm", plot = pylab)

stats.probplot(np.exp(numerical_col2.Cutting), dist = "norm", plot = pylab)

stats.probplot(np.log(numerical_col2.Cutting), dist = "norm", plot = pylab)

########## standarization of the data ############################

################################ Initialise the Scaler #########################################
scaler = StandardScaler()

# To scale data
numerical_col3 = scaler.fit_transform(numerical_col2)
# Convert the array back to a dataframe
numerical_col4 = pd.DataFrame(numerical_col3)
describe_standarization = numerical_col4.describe()

########### Now concat the two dataset numerical_col4 and categorical_col ####################

dataset = pd.concat([numerical_col4, categorical_col], axis=1)

### Now make the connection between python to sql to move clean datset in sql database #########

user_name = 'root'
database = 'Machine'
your_password = 'priya08021'
engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{your_password}'))

dataset.to_sql('Machine', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

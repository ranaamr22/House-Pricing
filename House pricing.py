import pandas as pd

df = pd.read_csv('train.csv')
df=pd.get_dummies(df ,columns=['SaleCondition','MSZoning','Street','Alley',
                                   'LotShape','LandContour','Utilities','Neighborhood'
                                   ,'LotConfig','LandSlope','Condition1','Condition2',
                                   'BldgType','HouseStyle','RoofMatl','RoofStyle',
                                   'Exterior1st','Exterior2nd','MasVnrType','ExterQual',
                                   'ExterCond','Foundation','BsmtQual','BsmtCond',
                                   'BsmtExposure','BsmtFinType1','BsmtFinType2',
                                   'Heating','HeatingQC','CentralAir','Electrical',
                                   'KitchenQual','Functional','FireplaceQu','GarageType',
                                   'GarageFinish','GarageQual','GarageCond','PavedDrive',
                                   'PoolQC','Fence','MiscFeature','SaleType'])

df.drop(columns=df.columns[df.isnull().sum().values>200],inplace=True)
print(type(df.columns))
for col in df.columns:
  df[col].fillna(df[col].mode, inplace = True)

  df.isnull().sum().sum()
  
import numpy

y = df['SalePrice']
x = df.drop(['SalePrice'], inplace = False, axis = 1)
print(x.shape)


y = numpy.array(y, dtype = int).reshape(-1,1)

featurelist = []
for col in x.columns:
  try:
    x_ = numpy.array(x[col], dtype = int)
    featurelist.append(x_)
  except:
    continue

features = numpy.array(featurelist, dtype = int)
print(f"the features shape {features.shape}, and the target shape {y.shape}")

from sklearn import linear_model

reg=linear_model.LinearRegression()
model=reg.fit(features.T,y)

model.score(features.T,y)
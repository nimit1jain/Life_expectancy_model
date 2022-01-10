from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


life_exp=pd.read_csv('N:\Machine learning\Algorithms\Life Expectancy Data.csv')

life_exp.columns=life_exp.columns.str.strip()

life_exp=life_exp.drop('Year',axis=1)
temp_life_exp=life_exp

status=pd.get_dummies(life_exp.Status)
life_exp=pd.concat([life_exp,status],axis=1)
life_exp=life_exp.drop(['Status'],axis=1)
life_exp.rename(columns={'Developing':0,'Developed':1})

life_exp=life_exp.groupby('Country').mean()

                                                #-----Droped duplicates if any------

# temp_life_exp=temp_life_exp.drop_duplicates()
le=LabelEncoder()
# Country_encoded=temp_life_exp['GDP']
# temp_life_exp=pd.concat([temp_life_exp,Country_encoded],axis=1)
temp_life_exp['Country_encoded']=le.fit_transform(temp_life_exp['Country']) 
temp_life_exp=temp_life_exp.drop('Country',axis=1)
temp_life_exp['Status_encoded']=le.fit_transform(temp_life_exp['Status']) 
temp_life_exp=temp_life_exp.drop('Status',axis=1)

# print(temp_life_exp.head(20))



                                                #----seperating target and features---------
target=temp_life_exp['Life expectancy']
life_features=temp_life_exp.drop(['Life expectancy'],axis=1)



                                                  #------DEaling with NULL values------
life_features.fillna(value=life_features.mean(),inplace=True)

target.fillna(value=target.mean(),inplace=True)

                                                   #------checking outliers-------


# for i in life_features.columns:
#     q75, q25 = np.percentile(life_features[i], [75 ,25])
#     iqr = q75 - q25
#     min_val = q25 - (iqr*1.5)
#     max_val = q75 + (iqr*1.5)
#     
    
#     life_features=life_features[(life_features[i]<max_val)]
#     life_features=life_features[(life_features[i]>min_val)]

    



                                               #-------Feature selection----------
                                               #-------Filter Method-------

# cmap=sns.diverging_palette(500,10, as_cmap=True)
# sns.heatmap(life_exp.corr(),annot=True)
# plt.show()


# By analysing the correlation map we can say that (GDP,percentage expenditure), (thinness 5-9 years,thinness 1-19 years),(under five deaths, infant deaths), (ICOR,Schooling) are highly correlated
# GDP, ICOR, HIV/AIDS, Adult Mortality, BMI have high correlation with life expectancy

life_features=life_features.drop(['percentage expenditure','thinness 5-9 years','infant deaths'],axis=1) 

                                               #------applying feature embedding-------

# rfr=RandomForestRegressor(n_estimators=1300)
# rfr.fit(life_features,target)
# importance=rfr.feature_importances_
# importance_df=pd.DataFrame({"Features": life_features.columns,"Importance": importance})
# importance_df=importance_df.sort_values("Importance")

# plt.bar(importance_df["Features"],importance_df["Importance"])
# plt.show()


# sns.histplot(life_exp['Life expectancy'].dropna(),kde=True,color='green')
# plt.show()


                                                         #-------scaling---------


min_max_scaler=MinMaxScaler()
life_features=min_max_scaler.fit_transform(life_features)

target=np.reshape(np.array(target),(len(target),1))
target=min_max_scaler.fit_transform(target)

                                                        #--------splitting data------

x_train,x_test,y_train,y_test=train_test_split(life_features,target,test_size=0.2,random_state=42)

                                                        #--------training the model--------

model=LinearRegression()
model.fit(x_train,y_train)

                                                        #--------evaluating model over test data----------
y_pred=model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2score = r2_score(y_test, y_pred)
print('Model mse: ',mse)
print('Model rmse: ',rmse)
print('Model r2_score: ',r2score)
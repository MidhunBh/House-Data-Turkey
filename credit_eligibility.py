import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

"""#Loading the dataset"""

data=pd.read_csv('/content/drive/MyDrive/HouseData.csv')

data.head()

data.shape

data.isnull().sum()

sb.heatmap(data.isnull(),yticklabels=False,cbar=False)

"""#removing columns with null values"""

data.drop(columns={'Unnamed: 0','EligibilityForInvestment','AdUpdateDate','BuildStatus','Type','TitleStatus','FloorLocation','AdCreationDate','ItemStatus','address','Swap','district','Balcony','StructureType','Category','NumberOfWCs','MortgageStatus','RentalIncome','NumberOfBalconies','BalconyType','HallSquareMeters','WCSquareMeters','IsItVideoNavigable?','Subscription','BathroomSquareMeters','BalconySquareMeters'},axis=1,inplace=True)

sb.heatmap(data.isnull(),yticklabels=False,cbar=False)

data.head()

"""#change columns with object datatypes into numerical values(int and float)"""

net = data['NetSquareMeters'].values
gross = data['GrossSquareMeters'].values
price = data['price'].values
bath = data['NumberOfBathrooms'].values
age=data['BuildingAge'].values
rooms=data['NumberOfRooms'].values

x1=[]
for i in net: 
  j=i.split(' m2')[0]    #seperate the numerical value in feature - 'Net square meters'
  x1.append(float(j))

x2=[]
for k in gross: 
  l = k.split(" m2")[0]    #seperate numerical value of gross square meters
  x2.append(float(l))

x3=[]
for k in price: 
  x = k.split("TL")[0]
  o=float(x.replace(',', ''))     #seperate price's numerical values,avoid all commas
  x3.append(o)

x4= []
for k in bath:
  if k=='Yok':
    k=0
  elif k=='6+':
    k=6.5             #take rough average of bathrooms equal and above the count 6 as 6.5            
  else:
    k=int(k)
  x4.append(k)

x5 = []               #organize 'building age' into manageable interval classes and take median of interval
for k in age:
  if k=='0 (Yeni)':
    k=0
  elif k=='1' or k=='2' or k=='3' or k=='4':
    k=2.5
  elif k=='5-10':
    k=7.5
  elif k=='11-15':
    k=13
  elif k=='16-20':
    k=18
  elif k=='21 Ve Üzeri':
    k=22
  
  x5.append(k)

x6=[]
for k in rooms:
  if k =='1 Oda' or k=='5 Oda':  
    x = float(k.split()[0])
  elif k =='8+ Oda':                           # average number of rooms of class 'equal or above 8' approximated as 8.5 
    x =8.5
  elif k =='Stüdyo':  
    x =2.5                                     #approximate studio room count as 2.5, as there aren't clear demarcations between rooms
  else:
    m=k.split("+")  
    x = float(m[0])+float(m[1])                #find the sum of total rooms in values written in the form 'x+y'
  x6.append(x)

"""#create new dataframe 'df' consisting of columns that now contain only numerical values"""

df = pd.DataFrame(list(zip(x1,x2,x3,x4,x5,x6)))
df.columns = ['Net SquareMeters', 'Gross SquareMeters', 'Price','Number of Bathrooms','Building Age','Number of rooms']

df.shape

data.shape

data=pd.concat([data,df],axis=1)

"""#remove old columns with object datatypes"""

data = data.drop({'NetSquareMeters','GrossSquareMeters','price','NumberOfBathrooms','BuildingAge','NumberOfRooms'},axis=1)

data.shape

data

"""#the feature 'HeatingType' has several unique values but 90% of them belong to three values.So filter the data that that have either three of these values """

data = data[(data['HeatingType']=='Kombi Doğalgaz')|(data['HeatingType']=='Merkezi (Pay Ölçer)')|(data['HeatingType']=='Merkezi Doğalgaz')]

"""#dataset converted to numerical features and some catergorical features that are suitable for encoding."""

data.shape

"""#Encoding"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['UsingStatus'] = le.fit_transform(data['UsingStatus'])
data['CreditEligibility'] = le.fit_transform(data['CreditEligibility'])
data['InsideTheSite'] = le.fit_transform(data['InsideTheSite'])
data['PriceStatus'] = le.fit_transform(data['PriceStatus'])
data['HeatingType'] = le.fit_transform(data['HeatingType'])
data['Building Age'] = le.fit_transform(data['Building Age'])

data

data.dtypes

"""#This model aims to predict the 'credit eligibility' feature, whether the house is eligible for loan or not ('Krediye Uygun' and 'Krediye Uygun Değil')		"""

x = data.drop({'CreditEligibility'},axis=1)
x

y = data['CreditEligibility']
y

"""#Scaling"""

from sklearn.preprocessing import StandardScaler

a = StandardScaler()

x = a.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=66)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

"""#Random forest classifier is used """

from sklearn.ensemble import RandomForestClassifier

"""#Hyper parameter tuning by randomized search"""

from sklearn.model_selection import RandomizedSearchCV

grid1={'n_estimators':[10,50,90,140,190,240],'criterion':['entropy','log_loss','gini'],'max_features':['sqrt', 'log2'], 'min_samples_split': [1, 3,6,10], 'min_samples_leaf': [1, 3,6,10]}

rndm_search_RFC = RandomizedSearchCV(RandomForestClassifier(),param_distributions=grid1,n_jobs=-1)

rndm_search_RFC.fit(x_train,y_train)

rndm_search_RFC.best_score_

rndm_search_RFC.best_params_

"""#use the parameters with the best scores"""

rfc = RandomForestClassifier(n_estimators=140,max_features='log2',criterion='gini',min_samples_leaf=3,min_samples_split=6)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

"""#Check accuracy"""

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

rfc.score(x_train,y_train)

rfc.score(x_test,y_test)

"""#Cross Validation by Stratified K-Fold Method"""

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

skf = StratifiedKFold(10)

score = cross_val_score(rfc,x,y,cv=skf)

len(score)

np.mean(score)

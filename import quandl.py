import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key='oaUWCMQ1xq4xHmvLtagd'
df=quandl.get("NSE/RELIANCE")
#print(df.head(15))

plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Closing price')
#plt.show()

df['Open - Close']=df['Open']-df['Close']
df['High - Low'] = df['High']-df['Low']
df=df.dropna()

X=df[['Open - Close','High - Low']]
X.head()
#print(X)

Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
#print(Y)

from sklearn.model_selection import train_test_split
X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=44)





from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

params={'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn =neighbors.KNeighborsClassifier()
model =GridSearchCV(knn,params,cv=5)

model.fit(X_train,Y_train)

accuracy_train = accuracy_score(Y_train,model.predict(X_train))
accuracy_test = accuracy_score(Y_test,model.predict(X_test))

accuracy_train=accuracy_train*100
print('Train data Accuracy is %.3f' %accuracy_train)
#print('Test data Accuracy is %.3f' %accuracy_test)

predctions_classification =model.predict(X_test)
actual_predicted_data =pd.DataFrame({'Actual Class':Y_test, 'Predicted class ':predctions_classification})
#print(actual_predicted_data.head(1))
rt=predctions_classification[-1]
if(rt==1):
    print("BUY")
else:
    print("SELL")    

Y=df['Close']
#print(Y)    



from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors
X_train_reg ,X_test_reg,Y_train_reg,Y_test_reg = train_test_split(X,Y,test_size=0.25,random_state=44)


params={'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg =neighbors.KNeighborsRegressor()
model_reg =GridSearchCV(knn_reg,params,cv=5)
model_reg.fit(X_train_reg,Y_train_reg)
predctions =model_reg.predict(X_test_reg)

rrm=predctions
print(predctions[-1])
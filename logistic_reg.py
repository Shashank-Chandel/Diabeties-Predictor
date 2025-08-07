import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# read the data file
data = pd.read_csv('diabetes.csv')
print(data.head())


print(data.describe())


print(data.isnull().sum())


# there is a misconception like there is BMI which can not be zero , BP cant be zero , glucose , insulin cant be zero so lets try to fix it 
# now replacing zero values with the mean of the columns

data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())

print(data.describe())


# creating boc plot to check outliners

fig , ax = plt.subplots(figsize = (15,10))
sns.boxplot(data=data , width=0.5 , ax = ax , fliersize=3 )
#plt.show()

# segregate the dependent and independent variable
x = data.drop(columns= ['Outcome'])
y = data['Outcome']

# train test split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25 , random_state=0)
print(x_train.shape , x_test.shape)


import pickle
# standard scaling - standardization
def scaler_standard(x_train , x_test) :
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    # saving the model
    file = open('Model/standardScaler.pkl','wb')
    pickle.dump(scaler,file)
    file.close()

    return x_train_scaled , x_test_scaled


x_train_scaled , x_test_scaled = scaler_standard(x_train , x_test)


logreg = LogisticRegression()
logreg.fit(x_train_scaled , y_train) # this line will train model on the basis of x train and y train
y_pred = logreg.predict(x_test_scaled)
print(y_pred)



from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print("Before hyperparameter tuning:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





# hyperparameter tuning
# grid search cv

from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# parameter grid

parameters = {
    'penalty' : ['l1','l2'] , 
    'C' : np.logspace(-3,3,7) , 
    'solver' : ['newton-cg' , 'lbfgs' , 'liblinear']
}


clf = GridSearchCV(logreg , 
                   param_grid=parameters , 
                   scoring='accuracy' , 
                   cv=10)
clf.fit(x_train_scaled,y_train) # scaled to validation data

print(clf.best_params_)
print(clf.best_score_)



logreg = LogisticRegression ( C =  1, penalty = 'l2',  solver = 'liblinear' )
logreg.fit(x_train_scaled,y_train)

y_pred = logreg.predict(x_test_scaled)


# accuracy check


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(y_pred , y_test))
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


 # saving the model
file = open('Model/modelforprediction.pkl','wb')
pickle.dump(logreg,file)
file.close()

print(data.columns)

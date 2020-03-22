import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_excel('D:\AAA\Hackathon\Flipr\Train_dataset.xlsx')


#values={'Mode_transport':'Public','comorbidity':'None','cardiological pressure':	'Normal',
#        'Diuresis':265,	'Platelets':82,	'HBB':115,	'd-dimer':275,	'Heart rate':75,
#        'HDL cholesterol':52,'FT/month':1,'Children':1,'Occupation':'Sales'}

##fill nan with mean
values={'Diuresis':int(dataset['Diuresis'].mean()),	'Platelets':int(dataset['Platelets'].mean()),	
        'HBB':int(dataset['HBB'].mean()),	'd-dimer':int(dataset['d-dimer'].mean()),	
        'Heart rate':int(dataset['Heart rate'].mean()),'HDL cholesterol':int(dataset['HDL cholesterol'].mean()),
        'FT/month':1,'Children':1}

dataset=dataset.fillna(value=values)

#dataset.dropna(inplace=True)
dataset['Mode_transport'].fillna(method='ffill',inplace=True)
dataset['comorbidity'].fillna(method='ffill',inplace=True)
dataset['cardiological pressure'].fillna(method='ffill',inplace=True)
dataset['Occupation'].fillna(method='ffill',inplace=True)
#
#
#X = dataset.iloc[:, [1,2,8,9,10,11,12,13,14,15,16,17,18,19,21,20,22,23,26,5,7,6]].values
#y = dataset.iloc[:, 27].values


dataset=dataset.drop(columns=['people_ID','Designation','Name','Insurance','salary'])
X=dataset.iloc[:,0:22].values
y=dataset.iloc[:,22:23].values


################
from sklearn.preprocessing import LabelEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])

labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])

labelencoder_X3 = LabelEncoder()
X[:, 2] = labelencoder_X3.fit_transform(X[:, 2])

labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])

labelencoder_X5 = LabelEncoder()
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])


labelencoder_X6 = LabelEncoder()
X[:, 8] = labelencoder_X6.fit_transform(X[:, 8])

labelencoder_X7 = LabelEncoder()
X[:, 11] = labelencoder_X7.fit_transform(X[:, 11])
#
labelencoder_X8 = LabelEncoder()
X[:, 12] = labelencoder_X8.fit_transform(X[:, 12])


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

y=y.flatten()
#OneHot Encoding
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 70, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


y_test=y_test
from sklearn import metrics

print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt((metrics.mean_squared_error(y_test,y_pred)))) 
print(metrics.r2_score(y_test,y_pred))
   
#######################################################################################

test_dataset = pd.read_excel('D:\AAA\Hackathon\Flipr\Test_dataset.xlsx')
dataset=test_dataset
values={'Diuresis':int(dataset['Diuresis'].mean()),	'Platelets':int(dataset['Platelets'].mean()),	
        'HBB':int(dataset['HBB'].mean()),	'd-dimer':int(dataset['d-dimer'].mean()),	
        'Heart rate':int(dataset['Heart rate'].mean()),'HDL cholesterol':int(dataset['HDL cholesterol'].mean()),
        'FT/month':1,'Children':1}

dataset=dataset.fillna(value=values)

#dataset.dropna(inplace=True)
dataset['Mode_transport'].fillna("Public",inplace=True)
dataset['comorbidity'].fillna("None",inplace=True)
dataset['cardiological pressure'].fillna("Normal",inplace=True)
dataset['Occupation'].fillna("",inplace=True)


dataset=dataset.drop(columns=['people_ID','Designation','Name','Insurance','salary'])
X=dataset.iloc[:,0:22].values


################
from sklearn.preprocessing import LabelEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])

labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X2.fit_transform(X[:, 1])

labelencoder_X3 = LabelEncoder()
X[:, 2] = labelencoder_X3.fit_transform(X[:, 2])

labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])

labelencoder_X5 = LabelEncoder()
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])


labelencoder_X6 = LabelEncoder()
X[:, 8] = labelencoder_X6.fit_transform(X[:, 8])

labelencoder_X7 = LabelEncoder()
X[:, 11] = labelencoder_X7.fit_transform(X[:, 11])
#
labelencoder_X8 = LabelEncoder()
X[:, 12] = labelencoder_X8.fit_transform(X[:, 12])


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

y_pred = regressor.predict(X)

###############################
import xlsxwriter
workbook=xlsxwriter.Workbook('Result1.xlsx')
sheet=workbook.add_worksheet()
row=0
column=0
content=[]

for i in y_pred:
    sheet.write(row,column,i)
    row=row+1

workbook.close()




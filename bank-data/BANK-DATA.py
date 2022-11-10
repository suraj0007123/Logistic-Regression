import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Importing Data
bank_data = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\logistic regressionss\Datasets_LR\bank_data.csv")

bank_data.columns

bank_data.head(11)

bank_data.describe()

bank_data.info()

bank_data.isna().sum() # no na values

bank_data.columns = 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown','y' #renaming so that no sapces is there otherwise error.

bank_data = bank_data[['y', 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown']] # rearranging columns

############# Model building ##################

from sklearn.linear_model import LogisticRegression

X = bank_data.iloc[:,1:]
y = bank_data[["y"]].values.ravel()

log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_model.fit(X, y)

#############################################################

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

#Model Predictions
y_pred = log_model.predict(X)
y_pred

#Testing Model Accuracy
# Confusion Matrix for the model accuracy
confusion_matrix(y, y_pred)

# The model accuracy is calculated by (a+d)/(a+b+c+d)
accuracy = (39012 + 1710)/(45211) 
accuracy #0.9007100042025171

print(classification_report(y,y_pred)) # accuracy = 0.90

# As accuracy = 0.8923049700294177, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 
log_model.predict_proba(X)[:,1]

# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(y,log_model.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc) #auc accuracy: 0.6502590431375214 - Average model, it is less than 0.8 

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
classifier = LogisticRegression(random_state = 0,solver='lbfgs', max_iter=1000)
model1 = classifier.fit(X_train, y_train)

#Testing model
from sklearn.metrics import confusion_matrix, accuracy_score
y_predtest = classifier.predict(X_test)
print(confusion_matrix(y_test,y_predtest))
print(accuracy_score(y_test,y_predtest)) #Accuracy = 0.8963432615747567 = 89.63%


#Training model
y_predtrain = classifier.predict(X_train)
print(confusion_matrix(y_train,y_predtrain))
print(accuracy_score(y_train,y_predtrain)) #Accuracy = 0.9015704490157045 = 90.15%

# train and test accuracy is close enough so it is good model.


import statsmodels.formula.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

logit_model = sm.logit('y ~ default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data).fit(max_iter=1000)

#AIC means Akaike's Information Criteria and BIC means Bayesian Information Criteria. It should be less
#summary
logit_model.summary2() # for AIC:22693.5690, BIC:22928.9845 
logit_model.summary()

pred = logit_model.predict(bank_data.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(bank_data.y, pred) #It gives us FPR, TPR for different thresholds(cutoff)
optimal_idx = np.argmax(tpr - fpr) # TP Should be maximum as compare to FP
optimal_threshold = thresholds[optimal_idx] #at that maximum value what is the threshold(cutoff)
optimal_threshold #0.11508308700688377

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red') #True Positive Rate - Sensitivity
pl.plot(roc['1-fpr'], color = 'blue') # True Negative Rate
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc) # 0.890842

# filling all the cells with zeroes
bank_data["pred"] = np.zeros(45211) # add new column "pred" with all zeros
# taking threshold value and above the prob value will be treated as correct value 
bank_data.loc[pred > optimal_threshold, "pred"] = 1 # if the value is greater than threshold value mark it as "1" otherwise "0"
# classification report
classification = classification_report(bank_data["pred"], bank_data["y"])
classification # support=accuracy=0.69 


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(bank_data, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + joblue_collar + joentrepreneur + johousemaid + jomanagement + joretired + joself_employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data).fit( max_iter=1000)

#summary
model.summary2() # for AIC  AIC:22693.5690   , BIC:22928.9845 
model.summary() 

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (1295 + 9816)/(13564) 
accuracy_test #0.8191536419935123

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test #0.8899674437017001 - average model, it should greater than 0.8 


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (3005 + 23010)/(31647)
print(accuracy_train) #0.8220368439346541

# train and test accuracy is close enough so we can accept.
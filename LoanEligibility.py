#!/usr/bin/env python
# coding: utf-8

# In[1]:

#MODULE IMPLIMENTATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:

#LOADING DATASET
dataset = pd.read_csv("loan.csv")


# In[4]:

#LOADING DATASET PREVIEW
dataset.head()


# In[5]:

# THE ROWS AND COLS THE DATASET CONSIST OF
dataset.shape


# In[6]:

#INFO FUNCTION TO SPOT MISSING VALUES
dataset.info()


# In[7]:

#BASIC INFO ABOUT THE DATASET
dataset.describe()


# In[8]:

#HOW THE CREDIT HISTORY RELATES TO LOAN STATUS
#WE OBSERVE PEOPLE WITH CREDIT HISTORY VALUE 1 ARE MORE LIKELY TO GET A LOAN
pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)


# In[10]:

#WE USE BOXPLOT TO VISUALIZE SOME OF OUR VARIEBLES
#WE ALSO OBSERVE THAT THERE ARE A LOT OF OUTLIERS
dataset.boxplot(column='ApplicantIncome')


# In[11]:

#HISTOGRAM ABOUT THE ABOVE
dataset['ApplicantIncome'].hist(bins=20)


# In[12]:

#HISTOGRAM ABOUT THE COAPPLICANTINCOME
dataset['CoapplicantIncome'].hist(bins=20)


# In[13]:

#WITH BOXPLOT WE UNDERSTAND THE RELASIONSHIP
dataset.boxplot(column='ApplicantIncome', by='Education')


# In[14]:

#WE EXPLORE WITH BOXPLOT LOANAMOUNT 
dataset.boxplot(column='LoanAmount')


# In[15]:

#HISTOGRAM ABOUT THE LOANAMOUNT
dataset['LoanAmount'].hist(bins=20)


# In[16]:

#NORMALIZING LOANAMOUNT WITH NUMPY
dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[17]:

#FINDING HOW MANY MISSING VALUES ARE
dataset.isnull().sum()


# In[34]:

#FILLING GENDRES MISSING VALUES
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[35]:

#FILLING MARRIED MISSING VALUES
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[36]:

#FILLING DEPENDENTS MISSING VALUES
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[37]:

#FILLING SELF_EMPLOYED MISSING VALUES
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[38]:

#FILLING LOAN_AMOUNT MISSING VALUES
dataset.LoanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log=dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[39]:

#FILLING LOAN_AMOUNT_TERM MISSING VALUES
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[40]:

#FILLING CREDIT_HISTORY MISSING VALUES
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[41]:


dataset.isnull().sum()


# In[42]:

#NORMALIZING APPLICANTINCOME AND COAPPLICANTINCOME
dataset['TotalIncome']=dataset['ApplicantIncome']+ dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[43]:

#HISTOGRAM ABOUT THE TOTALINCOME_LOG
dataset['TotalIncome_log'].hist(bins=20)


# In[44]:


dataset.head()


# In[54]:

#DIVIDING OUR VARIEBLES INTO DEPENDED AND INDEPENDED 
#X->INDEPENDED
#Y->DEPENDED
x=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y=dataset.iloc[:,12].values


# In[55]:

#PRINTING X
x


# In[56]:

#PRINTING Y
y


# In[57]:

#SPLITTING OUR VARIEBLES INTO TEST AND TRAIN WITH SKLEARN
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[58]:

#PRINTIG X_TRAIN
print(x_train)


# In[59]:

#CONVERTING INTO 0 AND 1 
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()


# In[60]:

#APLLYING TO EACH INDEX
for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])


# In[61]:


x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])


# In[63]:


x_train


# In[64]:

#CONVERTING 0 AND 1  THE Y_TRAIN
labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[65]:


y_train


# In[66]:


for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])


# In[67]:


x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])


# In[68]:


labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)


# In[69]:


x_test


# In[70]:


y_test


# In[71]:

#SCALING DATASET
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[73]:

#IMPORTING LIBRABY OF DECISION TREE
from sklearn.tree import DecisionTreeClassifier
DTClassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)


# In[74]:

#PREDICTING THE VALUES OF THE DATASET 
y_pred=DTClassifier.predict(x_test)
y_pred


# In[76]:

#FINDING THE ACCURACY
from sklearn import metrics
print('The accuracy of decision tree is:',metrics.accuracy_score(y_pred,y_test))


# In[77]:

#IMPORTING LIBRABY OF NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[78]:


y_pred=NBClassifier.predict(x_test)


# In[79]:

#PRINTING THE PREDICTION
y_pred


# In[80]:

#PRINTING THE ACCURACY WICH IS BETTER THAN DECISION TREE
print('The accuracy of naive bayes is:',metrics.accuracy_score(y_pred,y_test))


# In[81]:


testdata=pd.read_csv("loan.csv")


# In[82]:


testdata.head()


# In[83]:

#IMPORTING THE DATASET THAT WE WANT TO TEST
testdata=pd.read_csv("loan-test.csv")


# In[84]:


testdata.head()


# In[85]:


testdata.info()


# In[86]:

#FINDING MISSING VALUES
testdata.isnull().sum()


# In[89]:

#FILLING MISSING VALUES
testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[90]:


testdata.isnull().sum()


# In[91]:

#VISUALIZING LOANAMOUNT
testdata.boxplot(column='LoanAmount')


# In[92]:

#VISUALIZING APPLICANTINCOME
testdata.boxplot(column='ApplicantIncome')


# In[93]:

#HANDLING MISSING VALUES OF LOANAMOUNT
testdata.LoanAmount=testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[94]:

#NORMALIZING LOANAMOUNT WITH LOG
testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[95]:


testdata.isnull().sum()


# In[97]:

#FORMING TOTALINCOME BY THE SUM OF APPLICANTINCOME AND COAPPLICANTINCOME
testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[98]:


testdata.head()


# In[100]:


test= testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[103]:


for i in range (0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[104]:


test[:,7]=labelencoder_x.fit_transform(test[:,7])


# In[105]:

#PRINTING THE TEST DATASET WITH 1 AND 0
test


# In[106]:


test=ss.fit_transform(test)


# In[107]:

#MAKING THE PREDICTION OF THE VALUES
pred=NBClassifier.predict(test)


# In[108]:

#PRINTING THE RESULT
pred


# In[ ]:





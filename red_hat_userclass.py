# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:03:02 2016

@author: kiranbv
"""
## This code implements the red hat challenge hosted by kaggle.com. 

# Importing important modules
import os
import numpy
import scipy
import pandas
import sklearn

# Change working directory
wd_present = os.getcwd()
os.chdir('c:\\red_hat_kaggle')


# Getting the data
# Read the training and testing files
training_data_act = pandas.read_csv('act_train.csv')
testing_data_act = pandas.read_csv('act_test.csv')
training_data_people = pandas.read_csv('people.csv')


# joining people file to training and testing activity files
training_data_full = pandas.merge(training_data_people,training_data_act,on=['people_id'],how = 'inner')
testing_data_full = pandas.merge(training_data_people,testing_data_act,on=['people_id'],how = 'inner')
ncol_training = training_data_full.shape[1]
ncol_testing = testing_data_full.shape[1]
training_data_full.fillna('type 999',inplace=True)
testing_data_full.fillna('type 999',inplace=True)


# print number of categories in each variable 
for col in training_data_full.columns:
    print(col, (training_data_full[col].unique()).shape)
    print(col,training_data_full[col].unique().shape[0] - testing_data_full[col].unique().shape[0])

# eliminating some variables and isolating the label for results as well as target variables
labels_test = testing_data_full['activity_id'].values
target_train = training_data_full['outcome'].values
num_feature_train = training_data_full['char_38']
num_feature_test = testing_data_full['char_38']

cols_diff_cats = ['char_3_x','char_1_y','char_2_y','char_5_y']
diff_cat_cols_tr = training_data_full[cols_diff_cats]
diff_cat_cols_te = testing_data_full[cols_diff_cats]

grp1_tr = training_data_full['group_1']
grp1_te = testing_data_full['group_1']



# input correlations
#for col in training_data_full.columns:
#    import scipy.stats
#    print(col,scipy.stats.spearmanr(training_data_full[col],target_train,axis=0))
#    
#
#

##training_data = training_data_full.drop(['people_id','date_x','activity_id','date_y','outcome'],axis=1).values
##testing_data = training_data_full.drop(['people_id','date_x','activity_id','date_y'],axis=1).values
#training_data_full.drop(['char_3_x','char_1_y','char_2_y','char_5_y','people_id','group_1','date_x','date_y','activity_id','char_10_y','char_38','outcome'],axis=1,inplace=True)
#testing_data_full.drop(['char_3_x','char_1_y','char_2_y','char_5_y','people_id','group_1','date_x','date_y','activity_id','char_10_y','char_38'],axis=1,inplace=True)
#
#
## dealing with categorical variables
#
cols1 = ['char_7_x','char_4_x',
         'char_3_y','char_8_y','char_9_y',
         'activity_category'
         ]
cols1 = ['char_11', 'char_12', 'char_13', 'char_14', 'char_15',
       'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21',
       'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27',
       'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33',
       'char_34', 'char_35', 'char_36', 'char_37','activity_category']     
#       
#1. using sklearn
##from sklearn.preprocessing import Imputer
#for col in training_data_full.columns:
#    print(training_data_full[col].dtype)
#    if(training_data_full[col].dtype == 'bool'):
#        training_data_full[col].astype(numpy.int8)
#        testing_data_full[col].astype(numpy.int8)
#    else:
#        training_data_full[col] = training_data_full[col].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#        testing_data_full[col] = testing_data_full[col].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#
#for col in cols_diff_cats:
#    diff_cat_cols_tr[col] = diff_cat_cols_tr[col].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#    diff_cat_cols_te[col] = diff_cat_cols_te[col].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#
#grp1_tr = grp1_tr.apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#grp1_te = grp1_te.apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
#
#
#from sklearn.preprocessing import OneHotEncoder
#var_encode = OneHotEncoder()
#training_data_dummies = var_encode.fit_transform(training_data_full)
#testing_data_dummies = var_encode.transform(testing_data_full)
#
## attach numerical feature and group 1
#from scipy.sparse import hstack,vstack
#from scipy.sparse import csc_matrix
#mat_tr = csc_matrix(numpy.reshape(num_feature_train.values,(num_feature_train.shape[0],1)))
#training_data_dummies = hstack([training_data_dummies,mat_tr])
#
#mat_te = csc_matrix(numpy.reshape(num_feature_test.values,(num_feature_test.shape[0],1)))
#testing_data_dummies = hstack([testing_data_dummies,mat_te])
#
#grp1_tr = csc_matrix(numpy.reshape(grp1_tr.values,(grp1_tr.shape[0],1)))
#training_data_dummies = hstack([training_data_dummies,grp1_tr])
#
#grp1_te = csc_matrix(numpy.reshape(grp1_te.values,(grp1_te.shape[0],1)))
#testing_data_dummies = hstack([testing_data_dummies,grp1_te])
#
#
## encode features with different number of categories
#var_encode1 = OneHotEncoder()
#var_encode1.fit(pandas.concat([diff_cat_cols_tr,diff_cat_cols_te]))
#temp_tr = var_encode1.transform(diff_cat_cols_tr)
#temp_te = var_encode1.transform(diff_cat_cols_te)
#training_data_dummies = hstack([training_data_dummies,temp_tr])
#testing_data_dummies = hstack([testing_data_dummies,temp_te])


#2 using pandas dummies
#
training_data_dummies = pandas.get_dummies(training_data_full,columns = cols1)
testing_data_dummies = pandas.get_dummies(testing_data_full,columns = cols1)


# append column 38 to both vectdf1 and vectdf1test
#training_data_dummies['char38'] = training_data_full['char_38']
#testing_data_dummies['char38'] = testing_data_full['char_38']
#
#training_data_dummies['group_1'] = training_data_full['group_1']
#testing_data_dummies['group_1'] = testing_data_full['group_1']
#
## getting the date variable
#training_data_dummies['year'] = pandas.DatetimeIndex(training_data_full['date_x']).year
#training_data_dummies['month'] = pandas.DatetimeIndex(training_data_full['date_x']).month
#training_data_dummies['day'] = pandas.DatetimeIndex(training_data_full['date_x']).day
#
#testing_data_dummies['year'] = pandas.DatetimeIndex(testing_data_full['date_x']).year
#testing_data_dummies['month'] = pandas.DatetimeIndex(testing_data_full['date_x']).month
#testing_data_dummies['day'] = pandas.DatetimeIndex(testing_data_full['date_x']).day
#
#
ncol_train_dummies = training_data_dummies.shape[1]
ncol_test_dummies = testing_data_dummies.shape[1]
#
#
## getting training and testing data for classification
training_data_dummies = training_data_dummies.iloc[:,ncol_training:ncol_train_dummies]
testing_data_dummies = testing_data_dummies.iloc[:,ncol_testing:ncol_test_dummies]
#size_training = training_data.shape
#size_testing = testing_data.shape


# create small training andtesting data.group 1,char_38,date_x. date_y not very useful.
cols2 = ['group_1','char_38']
training_data = training_data_full[cols2]
testing_data = testing_data_full[cols2]
training_data['group_1'] = training_data_full['group_1'].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
testing_data['group_1'] = testing_data_full['group_1'].apply(lambda x: x.split(' ')[1]).astype(numpy.int32)
training_data['year1'] = pandas.DatetimeIndex(training_data_full['date_x']).year
testing_data['year1'] = pandas.DatetimeIndex(testing_data_full['date_x']).year
training_data['year2'] = pandas.DatetimeIndex(training_data_full['date_y']).year
testing_data['year2'] = pandas.DatetimeIndex(testing_data_full['date_y']).year
training_data['month1'] = pandas.DatetimeIndex(training_data_full['date_x']).month
testing_data['month1'] = pandas.DatetimeIndex(testing_data_full['date_x']).month
training_data['month2'] = pandas.DatetimeIndex(training_data_full['date_y']).month
testing_data['month2'] = pandas.DatetimeIndex(testing_data_full['date_y']).month
training_data['day1'] = pandas.DatetimeIndex(training_data_full['date_x']).day
testing_data['day1'] = pandas.DatetimeIndex(testing_data_full['date_x']).day
training_data['day2'] = pandas.DatetimeIndex(training_data_full['date_y']).day
testing_data['day2'] = pandas.DatetimeIndex(testing_data_full['date_y']).day



training_data = pandas.concat([training_data,training_data_dummies],axis=1)
testing_data = pandas.concat([testing_data,testing_data_dummies],axis=1)



# splitting the training data
from sklearn.cross_validation import train_test_split
tr_cv, te_cv, ytr_cv, yte_cv = train_test_split(training_data,target_train , test_size=0.33, random_state=42)




# classification models

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2',solver='liblinear',max_iter=100)
classifier.fit(training_data_dummies,target_train)


# Implement random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='gini')
classifier.fit(training_data_dummies,target_train)


## implement gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier =  GradientBoostingClassifier(loss='deviance',n_estimators=300)
classifier.fit(training_data_dummies,target_train)


#adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
classifier =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),algorithm="SAMME",n_estimators=100,learning_rate = 0.03)
classifier.fit(tr_cv,ytr_cv)
classifier.fit(training_data,target_train)




# implement xgboost
import xgboost
#classifier = xgboost.XGBClassifier(missing=numpy.nan, max_depth=3, n_estimators=250, learning_rate=0.01, seed=4242)
#classifier.fit(training_data_dummies, target_train)

param = {'max_depth':15, 'eta':0.05, 'silent':1, 'objective':'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree']= 0.90
param['min_child_weight'] = 0
param['booster'] = 'gbtree'
param['alpha'] = 0.4
param['colsample_bylevel'] = 0.8
param['gamma'] = 0.05

num_round = 350

maxauc = []
# cross validation xgboost
for x in range(1,10):
    for y in range(6,10):
        for z in range(1,10):

        
            param = {'max_depth':15, 'eta':0.05, 'silent':1, 'objective':'binary:logistic'}
            param['nthread'] = 4
            param['eval_metric'] = 'auc'
            param['subsample'] = 0.7
            param['colsample_bytree']= 0.9
            param['min_child_weight'] = 0
            param['booster'] = 'gbtree'
            param['alpha'] = 0.4
            param['colsample_bylevel'] = 0.9
            param['gamma'] = 0.005
            
            xgboost_classifier = xgboost.cv(param, xgboost.DMatrix(training_data,label=target_train), nfold=5,metrics={'auc'}, seed = 0)
            print(xgboost_classifier) 
            maxauc.append(numpy.max(xgboost_classifier)[0])

# good params depth = 15, eta = 0.05,





classifier = xgboost.train(param, xgboost.DMatrix(training_data,label=target_train), num_round)





#classifier = xgboost.train(param, xgboost.DMatrix(tr_cv,label=ytr_cv), num_round)


# getting the output on the test data
output_final = classifier.predict_proba(testing_data)[:,1]
output_final = classifier.predict(xgboost.DMatrix(testing_data))


# checking the auc of the cv test data
op = classifier.predict(xgboost.DMatrix(te_cv))
op = classifier.predict_proba(te_cv)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(yte_cv,op, pos_label=1)
metrics.auc(fpr, tpr)



# stratified k fold cv
#from sklearn.cross_validation import StratifiedKFold
#from scipy import interp
#
#cv = StratifiedKFold(target_train, n_folds=6)
#mean_tpr = 0.0
#mean_fpr = numpy.linspace(0, 1, 100)
#all_tpr = []
#for i, (train, test) in enumerate(cv):
#    classifier = xgboost.train(param,xgboost.DMatrix(training_data.values[train], label=target_train[train]),num_round)
#    probas_ = classifier.predict(xgboost.DMatrix(training_data.values[test]))
#    fpr, tpr, thresholds = metrics.roc_curve(target_train[test], probas_)
#    mean_tpr += interp(mean_fpr, fpr, tpr)
#    mean_tpr[0] = 0.0
#    roc_auc = metrics.auc(fpr, tpr)
#    print(roc_auc)

# Creating output dataframe to write in csv
output = {'activity_id':labels_test,
          'outcome':output_final}
outputdf = pandas.DataFrame(output)
#------------------------------------------------------------------------------

# Write to a csv file
outputdf.to_csv('redhat_submission.csv',sep = ',',index=False)


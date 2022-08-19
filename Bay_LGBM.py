# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:07:01 2021

@author: Vaishali Ravi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:01:56 2021

@author: Vaishali Ravi
"""



import pandas as pd
import numpy as np

col_names = ["SRC_ADD","DES_ADD","PKT_ID","FROM_NODE","TO_NODE",
    "PKT_TYPE","PKT_SIZE","FLAGS","FID","SEQ_NUMBER","NUMBER_OF_PKT",
    "NUMBER_OF_BYTE","NODE_NAME_FROM","NODE_NAME_TO","PKT_IN","PKT_OUT",
    "PKT_R","PKT_DELAY_NODE","PKT_RATE","BYTE_RATE",
    "PKT_AVG_SIZE","UTILIZATION","PKT_DELAY","PKT_SEND_TIME","PKT_RESEVED_TIME",
    "FIRST_PKT_SENT","LAST_PKT_RESEVED","label"]
import time


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#start_time = time.time()

df = reduce_mem_usage(pd.read_csv('full.csv',header=None, names = col_names))

#print('Dimensions of the Training set:',df.shape)

newlabeldf=df.replace({ 'Normal' : 0, 'UDP-Flood' : 2 ,'Smurf': 3, 'SIDDOS': 1, 'HTTP_FLOOD': 4})
df['label'] = newlabeldf
#df=df[:int(len(df)/3)]


to_drop_DoS = [0,1]
df=df[df['label'].isin(to_drop_DoS)];
#print(df.shape)
#print('Dimensions of DoS:' ,df.shape)
'''
#print('Unique set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
'''
categorical_columns=['PKT_TYPE', 'FLAGS', 'NODE_NAME_FROM','NODE_NAME_TO']
train_categorical = df[categorical_columns]

# pkt_type
unique_pkt_type=sorted(df.PKT_TYPE.unique())
string1 = 'PKT_TYPE_'
unique_pkt_type1=[string1 + x for x in unique_pkt_type]
#print(unique_pkt_type1)

# flags
unique_flags=sorted(df.FLAGS.unique())
string2 = 'FLAGS_'
unique_flags1=[string2 + x for x in unique_flags]
#print(unique_service2)


# NODE_NAME_FROM
unique_node_from=sorted(df.NODE_NAME_FROM.unique())
string3 = 'NODE_NAME_FROM_'
unique_node_from1=[string3 + x for x in unique_node_from]
#print(unique_flag2)

# NODE_NAME_TO
unique_node_to=sorted(df.NODE_NAME_TO.unique())
string3 = 'NODE_NAME_TO_'
unique_node_to1=[string3 + x for x in unique_node_to]

# put together
dumcols=unique_pkt_type1 + unique_flags1 + unique_node_from1+unique_node_to1

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_encode=train_categorical.apply(LabelEncoder().fit_transform)

enc = OneHotEncoder(categories='auto')
categorical_encenc = enc.fit_transform(categorical_encode)
df_cat_data = pd.DataFrame(categorical_encenc.toarray(),columns=dumcols)

#join categorical column
#print(df_cat_data.head())
#print(df_cat_data.shape)

#join categorical column with existing and drop original categorical column
newdf=df.join(df_cat_data)

newdf.drop('PKT_TYPE', axis=1, inplace=True)
newdf.drop('FLAGS', axis=1, inplace=True)
newdf.drop('NODE_NAME_FROM', axis=1, inplace=True)
newdf.drop('NODE_NAME_TO', axis=1, inplace=True)
'''
print("final column set")
print(newdf.shape)
'''
newdf=newdf.fillna(0)
#newdf=newdf.loc[0:97176]

from sklearn.model_selection import train_test_split
train,test = train_test_split(newdf, test_size = 0.20, random_state = 0)

X_train = train.drop('label',1)
Y_train = train.label

X_test = test.drop('label',1)
Y_test = test.label



'''
print(X_train.shape)    
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
'''
colNames=list(X_train)

#-----------------------DATA PREPROCESSING--------------------

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train) 

scaler1 = preprocessing.StandardScaler().fit(X_test)
X_test=scaler1.transform(X_test) 


#print("---------------FEATURE SELECTION--------------------------------")

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
rfe = RFE(estimator=clf,step=1,n_features_to_select=12)

rfe.fit(X_train, Y_train.astype(int))
X_rfeDoS=rfe.transform(X_train)
X_DoS_test=rfe.transform(X_test)
#ft=(time.time() - start_time)
#print("--- %s seconds ---" %ft )
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)

print('Features selected for DoS:',rfecolname_DoS)
'''
print(X_rfeDoS.shape)
print(X_DoS_test.shape)
'''
#print("-------------LGBM CLASSIFIER--------------------------")

from bayes_opt import BayesianOptimization
import lightgbm as lgb

def bayes_parameter_opt_lgb(train_data, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False):
    # prepare data
    
    # parameters
    def lgb_eval(num_leaves,  max_depth, max_bin, num_iterations):
        params = {'application':'binary', 'metric':'auc'}
        #params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        #params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        #params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['num_iterations'] = int(round(num_iterations))
        #params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        #params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        #params['subsample'] = max(min(subsample, 1), 0)
        
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
     
    lgbBO = BayesianOptimization(lgb_eval, {
                                            'num_leaves': (24, 80),    
                                            'max_depth': (5, 30),
                                            'max_bin':(20,90),
                                            'num_iterations':(5,20)
                                            })
    
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']

train_data = lgb.Dataset(data=X_rfeDoS, label=Y_train, free_raw_data=False)
opt_params = bayes_parameter_opt_lgb(train_data, init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)

opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
opt_params[1]['num_iterations'] = int(round(opt_params[1]['num_iterations']))
opt_params=opt_params[1]
print(opt_params)

'''
from lightgbm import LGBMClassifier
#start_time = time.time()
clf_LGBM_Df=LGBMClassifier()
clf_LGBM_Df.fit(X_rfeDoS, Y_train.astype(int))
Y_Df_pred=clf_LGBM_Df.predict(X_DoS_test)
'''
start_time = time.time()
clf_LGBM_Df = lgb.train(opt_params, train_data, 100)
Y_Df_pred=clf_LGBM_Df.predict(X_DoS_test)

for i in range(0,len(Y_Df_pred)):
    
    if Y_Df_pred[i]>=0.5:       # setting threshold to .5
       Y_Df_pred[i]=1
    else:  
       Y_Df_pred[i]=0

from sklearn.metrics import plot_confusion_matrix
print(pd.crosstab(Y_test, Y_Df_pred, rownames=['Actual attacks'], colnames=['Predicted attacks']))
#plot_confusion_matrix(clf_LGBM_Df, X_DoS_test, Y_test) 



#--------------------------CROSS VALIDATION-----------------------------------

from sklearn.model_selection import cross_val_score
'''
accuracy = cross_val_score(clf_LGBM_Df, X_DoS_test, Y_test.astype(int), cv=5, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
'''


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_Df_pred.astype(int))
print("Accuracy: %0.5f" %accuracy)
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_Df_pred, average='binary')
print('Precision: %.3f' % precision)
from sklearn.metrics import f1_score
score = f1_score(Y_test, Y_Df_pred, average='binary')
print('F-Measure: %.3f' % score)
ct=(time.time() - start_time)
print("--- %s seconds ---" %ct )

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    

     
auc = roc_auc_score(Y_test, Y_Df_pred)
fpr, tpr, thresholds = roc_curve(Y_test, Y_Df_pred)
plot_roc_curve(fpr, tpr)
print('AUC: %.2f' % auc)
import joblib
filename = 'trained_model.sav'
joblib.dump(clf_LGBM_Df, filename)
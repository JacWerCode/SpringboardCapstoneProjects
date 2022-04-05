#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve, f1_score, auc
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import PartialDependenceDisplay

from matplotlib.pyplot import figure

figure(figsize=(20, 20), dpi=80)




from skopt import BayesSearchCV

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

import shap

import os
from datetime import datetime as dt
import joblib
#imblearn stuff
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
#warnings
import warnings
warnings.filterwarnings("ignore") # this will take away the red dialog boxes in the output terminal


# The dataframe from our data wrangling notebook is imported and reduced to married and divorced participants, and divorced is set at the target variable. Then several columns are dropped because they are redundant orthey are too predictive such as 

# In[2]:


df = pd.read_csv('../Data/5_Final.csv',index_col='QKEY')

df = df[df['MaritalStatus'].isin(['Married','Divorced'])]

LOOKING_FOR = 'Divorced'
df[LOOKING_FOR] = df['MaritalStatus'] == LOOKING_FOR
target_names = ['Married',LOOKING_FOR]

print(df.shape)
target_perc = round(sum(df[LOOKING_FOR])/len(df[LOOKING_FOR])*100,2)
print(f'{LOOKING_FOR} makes up {target_perc}% of results')

dropcols = ['MaritalStatus','ClosestAdult']

df.drop(dropcols,axis=1,inplace=True)

object_cols = df.select_dtypes(['object']).columns
hot_coded_df = pd.get_dummies(df,object_cols,drop_first=False) #Quanifty object type columns with one hot encoding
print(hot_coded_df.shape)


# In[ ]:





# # Modeling functions

# In[3]:


def get_matrix_report(model,X_test,y_test,target_names):
    y_pred = model.predict(X_test)
    
    con_mat = confusion_matrix(y_test,y_pred)
    sns.heatmap(con_mat,cmap='Blues',annot=True,fmt='d')
    plt.show()
    class_report = classification_report(y_test,y_pred,target_names=target_names)
    print(class_report)
    return con_mat, class_report


# This function takes in the data and  

# In[4]:


def run_suite(model_name,trainTestDict,target_names,pipeline_dict,smote):
    
    model = model_dict[model_name]['Model']
    
    
    p_d = pipeline_dict

    if p_d['Processed']:
        p_d['Imputer'],p_d['Scaler'] = None,None
    
    if smote:
        pipeline = make_pipeline(p_d['Imputer'],p_d['Scaler'],SMOTE(random_state=p_d['RandomState']),model)
    else:    
        pipeline = make_pipeline(p_d['Imputer'],p_d['Scaler'],model)

    params = model_dict[model_name]['params']
    params = {model_name+"__"+p:params[p] for p in params}        
    cv = BayesSearchCV(pipeline,
                             search_spaces=params,
                             cv=KFold(n_splits=p_d['Folds'], random_state=p_d['RandomState'], shuffle=True),
                             scoring=p_d['Scoring'],
                             return_train_score=True,
                            n_iter = p_d['n_iter'],
                             n_jobs=-1,
                            #verbose=True,
                           random_state=p_d['RandomState'])
    
    cv.fit(trainTestDict['X_train'],trainTestDict['y_train'])
    best_e = cv.best_estimator_
    _,_ = get_matrix_report(best_e,trainTestDict['X_test'],trainTestDict['y_test'],target_names)
    
    return best_e[model_name]


# In[5]:


def runAllSuites(model_dict,trainTestDict,pipeline_dict):
    
    timestamp = int(dt.now().timestamp())
    fp = f'../Models/{timestamp}_{pipeline_dict["Folder_Title"]}'
    os.mkdir(fp)
    best = {}
    for _ in ['X','y']:
        trainTestDict[_].to_csv(f'{fp}/{_}')
    
    get_ipython().system('jupyter nbconvert --to script 2_PreprocessingANDModeling.ipynb')
    os.rename('2_PreprocessingANDModeling.py',f'{fp}/2_PreprocessingANDModeling.py')
    
    
    if pipeline_dict['Smote']:
        smote_list = [True,False]
    else:
        smote_list= [False]
    
    for model in model_dict:
        for smote in smote_list:
            model_name = model
            if smote:
                model_name+="__SMOTE"
                
            print(model_name)
            best[model_name] = run_suite(model,trainTestDict,target_names,pipeline_dict,smote)
            joblib.dump(best[model_name], f'{fp}/{model_name}.joblib')
    return best,fp


# 

# In[ ]:





# In[6]:


def graph_prec_rec_curve(best,model_names,fp,X_test,y_test):
    # predict probabilities # keep probabilities for the positive outcome only
    for i, model_name in enumerate(model_names):
        model = best[model_name]
        model_probs =model.predict_proba(X_test)[:, 1]
        # predict class values
        yhat = model.predict(X_test)
        model_precision, model_recall, _ = precision_recall_curve(y_test, model_probs)
        model_f1 = f1_score(y_test, yhat)
        model_mcc = matthews_corrcoef(y_test, yhat)
        model_pr_auc= auc(model_recall, model_precision)
        # summarize scores
        print('MCC=%.3f PR_auc=%.3f' % (model_mcc,model_pr_auc),model_name)
        # plot the precision-recall curves
        
        style = {0:("-","r"),
                    1:(":","b"),
                    2:("--","k"),
                    3:("-",'g')}
        
        marker,color = style[i]
        plt.plot(model_recall, model_precision, linestyle=marker,color=color,label=model_name)
    
    no_skill = sum(y_test)/ len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='-',color="y",label='No Skill')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(f'{fp}/PRECREC.png',transparent=False)
    plt.show()


# In[ ]:





# In[7]:


def split_data(hot_coded_df,LOOKING_FOR):
    X = hot_coded_df.drop(LOOKING_FOR,axis=1)
    y = hot_coded_df[LOOKING_FOR]

    trainTestDict = {'X':X,"y":y}
    trainTestDict['X_train'],trainTestDict['X_test'], trainTestDict['y_train'], trainTestDict['y_test'] = train_test_split(X, y, random_state=20,stratify=y)
    return trainTestDict

def process_splits(trainTestDict,preProsDict):
    for partition in ['X_train','X_test','X']:
        data = trainTestDict[partition]
        imputed = preProsDict['Imputer'].fit_transform(data)
        scaled = preProsDict['Scaler'].fit_transform(imputed) 
        trainTestDict[partition] = pd.DataFrame(scaled, columns=trainTestDict[partition].columns)
    return trainTestDict


# In[8]:


model_dict = {}

model_dict['logisticregression']={'Model':LogisticRegression(),
               'params':{'penalty':['l1','l2'],'solver':['liblinear']}}


model_dict['xgbclassifier']={'Model':XGBClassifier(),
               'params':{'learning_rate':list(np.arange(.1,.75,.05)),
                         'max_depth':list(np.arange(2,9)),
                         'scale_pos_weight':list(np.arange(3,9)),
                         'n_estimators':list(np.arange(30,210,10)),
                         'colsample_bytree':list(np.arange(.1,1.1,.1))
                        }}

#model_dict.pop('logisticregression');
#model_dict.pop('xgbclassifier');


# In this cell the preprocessinga dn modeling steps are done in one. The classifcation report output is catured and saved as a text file for comparison

# In[9]:


get_ipython().run_cell_magic('capture', 'cap --no-stderr', "\n\npipeline_dict = {'Imputer':SimpleImputer(missing_values=np.nan, strategy='median'),\n                'Scaler':StandardScaler(),\n                'Folds':5,\n                'Smote':True,\n                'Scoring':make_scorer(matthews_corrcoef),\n                'Folder_Title':'Divorced_MCC_No_Drops',\n                 'n_iter':50,\n                'RandomState':42,\n                'Processed':True}\n\n\ntrainTestDict = split_data(hot_coded_df,LOOKING_FOR)\n\nprocessed_TTS = process_splits(trainTestDict,pipeline_dict)\n    \nbest,fp = runAllSuites(model_dict,processed_TTS,pipeline_dict)")


# In[ ]:





# Results of the models are saved to a model_output.txt file

# Here the model metrics are displayed

# In[10]:


print(str(cap))
with open(f'{fp}/model_metrics.txt', 'w') as f:
    f.write(cap.stdout) 


# In[11]:


best.pop('xgbclassifier__SMOTE',None)
graph_prec_rec_curve(best,list(best.keys()),fp,trainTestDict['X_train'], trainTestDict['y_train'])


# In[12]:


graph_prec_rec_curve(best,list(best.keys()),fp,trainTestDict['X_test'], trainTestDict['y_test'])


# In[13]:


pca = PCA(n_components = 2)
use = ''
X_pca = pca.fit_transform(processed_TTS['X'])
#pc1, pc2 = pca.components_
pca_df =pd.DataFrame({'pc1':X_pca[:,0],'pc2':X_pca[:,1],'Divorced':processed_TTS['y']})
sns.scatterplot(data=pca_df,x='pc1',y='pc2',hue='Divorced')


# In[14]:


best['logisticregression__SMOTE']


# In[15]:


best['xgbclassifier']


# In[16]:


top = 8
for model in best:
    try:
        data = best[model].coef_[0]
    except:
        data = best[model].feature_importances_
    feature_rank = pd.Series(data,index=trainTestDict['X_train'].columns).sort_values(ascending=False,key=lambda x:abs(x))
    top_10 = feature_rank.head(top)
    sns.barplot(top_10,top_10.index)
    plt.title(model)
    plt.savefig(f'{fp}/{model}.png',transparent=False)
    plt.show()
    try:
        explainer = shap.Explainer(best[model])
        shap_values = explainer(trainTestDict['X_train'])
        shap.plots.beeswarm(shap_values)
        plt.savefig(f'{fp}/{model}___SHAP.png',transparent=False)
    except:
        pass


# In[17]:


def graphPDP(X,x_axis,y_axis):
    xi = list(X.columns).index(x_axis)
    yi = list(X.columns).index(y_axis)
    PartialDependenceDisplay.from_estimator(best['xgbclassifier'], X.dropna(),(xi,(xi,yi)))
    plt.show()
    
#graphPDP(X,'SEX_Male','SATLIFE_Family')


# In[ ]:





# In[ ]:





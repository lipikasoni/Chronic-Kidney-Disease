#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[233]:


df=pd.read_csv(r"C:\prac\kidney_disease.csv")


# In[234]:


df.head()


# In[235]:


columns=pd.read_csv("C:\prac\data_description.txt",sep='-')
columns=columns.reset_index()
columns.columns=['cols','abb_col_names']


# In[236]:


df.columns=columns['abb_col_names'].values


# In[237]:


df.head()


# In[238]:


df.dtypes


# In[239]:


features=['red blood cell count','packed cell volume','white blood cell count']


# In[240]:


def convert_dtype(df,feature):
    df[feature]=pd.to_numeric(df[feature], errors='coerce')


# In[241]:


for feature in features:
    convert_dtype(df,feature)


# In[242]:


df.drop('id',axis=1,inplace=True)


# In[243]:


def extract_cat_num(df):
    cat_col=[col for col in df.columns if df[col].dtype=='object']
    num_col=[col for col in df.columns if df[col].dtype!='object']
    return cat_col,num_col


# In[244]:


cat_col,num_col=extract_cat_num(df)


# In[245]:


for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')


# In[246]:


df['diabetes mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
df['coronary artery disease'].replace(to_replace={'\tno':'no'},inplace=True)


# In[247]:


df['class'].replace(to_replace={'ckd\t':'ckd'},inplace=True)


# In[248]:


for col in cat_col:
    print('{} has {} values'.format(col, df[col].unique()))
    print('\n')


# In[249]:


len(num_col)


# In[250]:


plt.figure(figsize=(40,30))
for i,feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    df[feature].hist()
    plt.title(feature)


# In[251]:


len(cat_col)


# In[252]:


plt.figure(figsize=(20,20))
for i,feature in enumerate(cat_col):
    plt.subplot(4,3,i+1)
    sns.histplot(df[feature])


# In[253]:


sns.histplot(df['class'])


# In[254]:


plt.figure(figsize=(10,10))
df.corr()
sns.heatmap(df.corr(),annot=True)


# In[255]:


df.groupby(['red blood cells','class'])['red blood cell count'].agg(['count','mean','median','min','max'])


# In[256]:


import plotly.express as px


# In[257]:


df.columns


# In[258]:


px.violin(df,y='red blood cell count',x="class", color="class")


# In[259]:


px.scatter(df,'haemoglobin','packed cell volume')


# In[260]:


grid=sns.FacetGrid(df, hue="class",aspect=2)
grid.map(sns.kdeplot, 'red blood cell count')
grid.add_legend()


# In[261]:


def violin(col):
    fig = px.violin(df, y=col, x="class", color="class", box=True)
    return fig.show()

def scatters(col1,col2):
    fig = px.scatter(df, x=col1, y=col2, color="class")
    return fig.show()


# In[262]:


def kde_plot(feature):
    grid = sns.FacetGrid(df, hue="class",aspect=2)
    grid.map(sns.kdeplot, feature)
    grid.add_legend()


# In[263]:


kde_plot('red blood cell count')


# In[264]:


kde_plot('haemoglobin')


# In[265]:


scatters('red blood cell count', 'packed cell volume')


# In[266]:


scatters('red blood cell count', 'haemoglobin')


# In[267]:


scatters('haemoglobin','packed cell volume')


# In[268]:


violin('red blood cell count')


# In[269]:


violin('packed cell volume')


# In[270]:


scatters('red blood cell count','albumin')


# In[271]:


scatters('packed cell volume','blood urea')


# In[272]:


fig = px.bar(df, x="specific gravity", y="packed cell volume",
             color='class', barmode='group',
             height=400)
fig.show()


# In[273]:


df.head()


# In[274]:


df.isna().sum().sort_values(ascending=False)


# In[275]:


cat_col


# In[276]:


data=df.copy()


# In[277]:


data['red blood cells'].isnull().sum()


# In[278]:


data['red blood cells'].dropna().sample()


# In[293]:


random_sample=data['red blood cells'].dropna().sample(data['red blood cells'].isnull().sum())
random_sample


# In[294]:


random_sample.index


# In[295]:


data[data['red blood cells'].isnull()].index


# In[296]:


random_sample.index=data[data['red blood cells'].isnull()].index


# In[297]:


random_sample.index


# In[298]:


random_sample


# In[299]:


data.loc[data['red blood cells'].isnull(),'red blood cells']=random_sample


# In[300]:


data.head()


# In[301]:


sns.histplot(data['red blood cells'])


# In[302]:


data['red blood cells'].value_counts()/len(data)


# In[303]:


len(df[df['red blood cells']=='normal'])/248


# In[304]:


len(df[df['red blood cells']=='abnormal'])/248


# In[305]:


def Random_value_imputation(feature):
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())               
    random_sample.index=data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature]=random_sample


# In[306]:


Random_value_imputation('pus cell')
Random_value_imputation('red blood cells')


# In[307]:


data[cat_col].isnull().sum()


# In[308]:


mode=data['pus cell clumps'].mode()[0]
mode


# In[309]:


data['pus cell clumps']=data['pus cell clumps'].fillna(mode)


# In[310]:


def impute_mode(feature):
    mode=data[feature].mode()[0]
    data[feature]=data[feature].fillna(mode)


# In[311]:


for col in cat_col:
    impute_mode(col)


# In[312]:


data[cat_col].isnull().sum()


# In[313]:


data[num_col].isnull().sum()


# In[314]:


for col in num_col:
    Random_value_imputation(col)


# In[315]:


data[num_col].isnull().sum()


# In[316]:


for col in cat_col:
    print('{} has {} categories'.format(col, data[col].nunique()))


# In[317]:


from sklearn.preprocessing import LabelEncoder


# In[318]:


le = LabelEncoder()


# In[319]:


for col in cat_col:
    data[col]=le.fit_transform(data[col])


# In[320]:


data.head()


# In[321]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[322]:


ind_col=[col for col in data.columns if col!='class']
dep_col='class'


# In[323]:


X=data[ind_col]
y=data[dep_col]


# In[324]:


ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(X,y)


# In[325]:


ordered_feature


# In[326]:


ordered_feature.scores_


# In[327]:


datascores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
datascores


# In[328]:


dfcolumns=pd.DataFrame(X.columns)
dfcolumns


# In[329]:


features_rank=pd.concat([dfcolumns,datascores],axis=1)
features_rank


# In[339]:


features_rank.columns=['Features','Score']
features_rank


# In[340]:


features_rank.nlargest(10,'Score')


# In[341]:


selected_columns=features_rank.nlargest(10,'Score')['Features'].values


# In[360]:


X_new=data[selected_columns]
X_new


# In[343]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new,y,train_size=0.75)


# In[344]:


print(X_train.shape)
print(X_test.shape)


# In[345]:


y_train.value_counts()


# In[346]:


from xgboost import XGBClassifier
XGBClassifier()


# In[361]:


params={
 "learning_rate"    : [0.05, 0.20, 0.25 ] ,
 "max_depth"        : [ 5, 8, 10],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.7 ]
    
}


# In[348]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[364]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[362]:


from sklearn.model_selection import RandomizedSearchCV


# In[365]:


random_search.fit(X_train, y_train)


# In[353]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[363]:


classifier=XGBClassifier()


# In[355]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[224]:


random_search.fit(X_train, y_train)


# In[381]:


classifier=random_search.best_estimator_
classifier


# In[ ]:





# In[380]:


random_search.best_params_


# In[382]:


classifier.fit(X_train,y_train)


# In[383]:


y_pred=classifier.predict(X_test)


# In[384]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[385]:


confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)


# In[386]:


plt.imshow(confusion)


# In[387]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:





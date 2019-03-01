# -*- coding: utf-8 -*-
"""
项目：Titannic 存活人数预测
作者： 李高俊
修改日期：2018/11/2
版本： V0.0
"""
import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
import matplotlib.pyplot as plt
from sklearn import model_selection
import sklearn
from sklearn import ensemble,gaussian_process,linear_model,naive_bayes,svm,tree
import numpy as np
import seaborn as sns
import xgboost

def load_file(path):  #数据导入
    f = open(path)
    train_set = pd.read_csv(f, index_col=0)
    return train_set


def date_clean(train_set):  #数据清洗
    return


def date_split(train_set):  #数据切分
    return


def creat_model():  #根据数据类型建立模型
    return


def model_val():  #模型评估及调试
    return


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_set = load_file(train_path)
    test_set = load_file(test_path)
    train_set.drop_duplicates(inplace=True)
    test_set.drop_duplicates(inplace=True)
    dataset = pd.concat([train_set,test_set])
    dataset['Age'].fillna(dataset['Age'].median(),inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(),inplace = True)
    drop_col = ['Cabin','Ticket']
    dataset.drop(drop_col,axis=1,inplace = True)
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset['Title'] = dataset['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
    stat_num = 10
    title_name = (dataset["Title"].value_counts() < 10)
    dataset["Title"] = dataset['Title'].apply(lambda x: 'Misc' if title_name.loc[x] == True else x)
    # print(dataset.dtypes)
    print("-" * 32)
    #onehot编码
    label = preprocessing.LabelEncoder()
    dataset['Embarked_code'] = label.fit_transform(dataset['Embarked'])
    dataset['Sex_code'] = label.fit_transform(dataset['Sex'])
    dataset['Title_code'] = label.fit_transform(dataset['Title'])
    dataset['Age'].astype('int64')
    data_x_col = ['Age','Embarked_code','Fare','Parch','Pclass','Sex_code','SibSp','FamilySize','IsAlone','Title_code']
    #切分数据
    data_x = dataset[data_x_col]
    train_dataset = dataset[data_x_col].loc[0:891,:]
    print(train_dataset.head(5))
    print("-" * 32)
    test_dataset = dataset[data_x_col].loc[892:,:]
    print(test_dataset.head(5))
    print("-" * 32)
    data_y = dataset['Survived'].dropna()
    print(data_y.head(5))
    train_set = pd.concat([train_dataset,data_y],axis=1)
    train_x,test_x,train_y,test_y = model_selection.train_test_split(train_dataset,data_y,test_size=0.3)
    #查看相关性
    # for x in train_set:
    #     if train_set[x].dtypes != 'float64':
    #         print('相关性：',x)
    #         print(train_set[[x,'Survived']].groupby(x,as_index=False).mean())
    #         print("-"*32)
    # plt.subplot(231)
    # plt.boxplot(x=train_set['Fare'],meanline=True)
    # plt.title('Fare Boxplot')
    # plt.ylabel('Fare ($)')
    #
    # plt.subplot(232)
    # plt.boxplot(x=train_set['Age'],showmeans=True)
    # plt.title('Age Boxplot')
    # plt.ylabel('Age ($)')
    #
    # plt.subplot(233)
    # plt.boxplot(x=train_set['FamilySize'],showmeans=True,meanline=True)
    # plt.title('FamilySize BoxPlot')
    # plt.ylabel('FamilySize ($)')
    # plt.show()
    # sns.pairplot(train_set,hue='Survived',size=1.5,palette='deep',diag_kind='kde')
    # sns.heatmap(train_set.corr(),cmap='Oranges',square=True,annot=True,annot_kws={'fontsize':8})
    # plt.show()
    #挑选模型
    MLA = [
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        gaussian_process.GaussianProcessClassifier(),

        linear_model.LogisticRegressionCV(),
        linear_model.RidgeClassifier(),
        linear_model.SGDClassifier(),

        naive_bayes.GaussianNB(),
        naive_bayes.BernoulliNB(),

        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        xgboost.XGBClassifier()
    ]
    cv_split = model_selection.ShuffleSplit(n_splits=10,test_size=0.2,train_size=0.6,random_state=0)
    MLA_col = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_com = pd.DataFrame(columns=MLA_col)
    MLA_predict = train_set['Survived']
    row_index = 0
    for alg in MLA:
        MLA_name = alg.__class__.__name__
        MLA_com.loc[row_index,"MLA Name"] = MLA_name
        MLA_com.loc[row_index,'MLA Parameters'] = str(alg.get_params())
        cv_result = model_selection.cross_validate(alg,train_set[data_x_col],train_set['Survived'],cv=cv_split)
        MLA_com.loc[row_index,"MLA Train Accuracy Mean"] = cv_result['train_score'].mean()
        MLA_com.loc[row_index,"MLA Test Accuracy Mean"] = cv_result['test_score'].mean()
        MLA_com.loc[row_index, "MLA Test Accuracy 3*STD"] = cv_result['test_score'].std() * 3
        MLA_com.loc[row_index, "MLA Time"] = cv_result['fit_time'].mean()

        alg.fit(train_set[data_x_col],train_set['Survived'])
        MLA_predict[MLA_name] = alg.predict(train_set[data_x_col])
        row_index += 1
    MLA_com.sort_values(by=['MLA Test Accuracy Mean'],ascending=False,inplace= True)
    test_dataset['Survived'] = MLA[-1].predict(test_dataset)
    test_dataset['Survived'].to_csv('Submission.csv')


    # sns.barplot(x='MLA Test Accuracy Mean',y='MLA Name',data=MLA_com,capsize=20)
    # plt.show()


if __name__ == '__main__':
    main()

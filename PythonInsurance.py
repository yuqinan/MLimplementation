import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn import svm
import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

train_data = DataFrame(pd.read_csv('/Users/qinanyu/Desktop/datapackages/insurance/train.csv'))
test_data = DataFrame(pd.read_csv('/Users/qinanyu/Desktop/datapackages/insurance/test.csv'))

# 确定没有需要清洗的数据
train_data.drop("id",axis=1,inplace=True)
test_data.drop("id",axis=1,inplace=True)
'''
print('---------------------train data info---------------------')
print(train_data.describe())
print('----------------------test data info---------------------')
print(test_data.describe())
'''

#数据替换和清洗
train_data['Gender']=train_data['Gender'].map({'Male':1,'Female':0})
train_data['Vehicle_Age']=train_data['Vehicle_Age'].map({'> 2 Years':2,'1-2 Year':1,'< 1 Year':0})
train_data['Vehicle_Damage']=train_data['Vehicle_Damage'].map({'Yes':1,'No':0})

test_data['Gender']=test_data['Gender'].map({'Male':1,'Female':0})
test_data['Vehicle_Age']=test_data['Vehicle_Age'].map({'> 2 Years':2,'1-2 Year':1,'< 1 Year':0})
test_data['Vehicle_Damage']=test_data['Vehicle_Damage'].map({'Yes':1,'No':0})


'''
#热力值关联分析
sns.countplot(train_data['Response'],label="Count")
plt.show()
#用热力图呈现相关性
corr = train_data[list(train_data.columns[0:10])].corr()
plt.figure(figsize=(11,11))
annot=True #显示每个方格的数据
sns.heatmap(corr,annot=True)
plt.show()
'''
#数据拆分
features = ['Gender', 'Age', 'Driving_License', 'Region_Code',
                                                         'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
                                                         'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
train_x, test_x, train_y, test_y = train_test_split(train_data.loc[:,
                                                        ['Gender', 'Age', 'Driving_License', 'Region_Code',
                                                         'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
                                                         'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']],
                                                        train_data.loc[:, ['Response']], test_size=0.25,
                                                        random_state=33)


#决策树算法
def reg_tree(train_x, test_x, train_y, test_y,train_data,test_data):
    #特征值选择

    train_features= train_data[features]
    train_labels = train_data['Response']


    dvec=DictVectorizer(sparse=False)
    # 代码中使用了 fit_transform 这个函数，它可以将特征向量转化为特征值矩阵
    train_features=dvec.fit_transform(train_features.to_dict(orient='record'))

    from sklearn.tree import DecisionTreeClassifier
    #构建ID3决策树
    clf= DecisionTreeClassifier(criterion='entropy')
    #决策树训练
    clf.fit(train_x, train_y)
    '''
    #决策树预测
    pred_labels = clf.predict(test_ss_x)

    from sklearn.model_selection import cross_val_score
     #使用K折交叉验证 统计决策树准确率
    print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
    mse = mean_squared_error(test_y, pred_labels)
    print("决策树均方误差 = ", round(mse, 2))
    print("决策树预测结果",pred_labels)

    sns.countplot(test_y['Response'], label="Count")
    plt.title("test-y")
    plt.show()
'''

    test_result = clf.predict(test_data)

    sns.countplot(test_result, label="Count")
    plt.title("reg-tree")
    plt.show()

reg_tree(train_x, test_x, train_y, test_y,train_data,test_data)

#KNN算法
def KNN(train_x, test_x, train_y, test_y,train_data,test_data):
    KNeighborsClassifier(n_neighbors=5000, weights='uniform', algorithm='auto', leaf_size=50)

    #采用Z-Score规范化
    ss = StandardScaler()
    train_ss_x = ss.fit_transform(train_x)
    test_ss_x = ss.transform(test_x)

    #创建KNN分类器
    knn = KNeighborsClassifier()
    knn.fit(train_ss_x, train_y)
   # predict_y = knn.predict(test_ss_x)
   # mse = mean_squared_error(test_y, predict_y)

    '''print("KNN准确率: %.4lf" % accuracy_score(test_y, predict_y))
    print("KNN均方误差 = ", round(mse, 2))
    print("KNN预测结果",predict_y)
'''
    test_result = knn.predict(test_data)
    sns.countplot(test_result, label="count")
    plt.title("KNN")
    plt.show()

'''
    data = pd.DataFrame({'KNN': test_result})
    datatoexcel = pd.ExcelWriter("KNNresult.xlsx", engine='xlsxwriter')
    data.to_excel(datatoexcel, sheet_name='Sheet1')
    datatoexcel.save()
'''


#KNN(train_x, test_x, train_y, test_y,train_data,test_data)

#Adaboost算法
def Adaboost(train_x, test_x, train_y, test_y,train_data,test_data):

    # 采用Z-Score规范化
    ss = StandardScaler()
    train_ss_x = ss.fit_transform(train_x)
    test_ss_x = ss.transform(test_x)

    # 使用AdaBoost分类模型
    ada = AdaBoostClassifier(n_estimators=2000, random_state=0)
    ada.fit(train_ss_x, train_y)
    '''
    pred_y = ada.predict(test_ss_x)
    mse = mean_squared_error(test_y, pred_y)
    print("均方误差 = ", round(mse, 2))
    print("预测数值",pred_y)
'   '''

    test_result = ada.predict(test_data)

    sns.countplot(test_result, label="count")
    plt.title("Ababoost")
    plt.show()
    '''
    data = pd.DataFrame({'Adaboost':pred_y})
    datatoexcel = pd.ExcelWriter("Insurance_result1.xlsx", engine='xlsxwriter')
    data.to_excel(datatoexcel, sheet_name='Sheet1')
    datatoexcel.save()
    '''

#Adaboost(train_x, test_x, train_y, test_y,train_data,test_data)

#SVM
def SVM(train_x, test_x, train_y, test_y,train_data,test_data):

    #创建SVM分类器
    model = svm.SVC()
    # 用训练集做训练
    model.fit(train_x, train_y)

    # 用测试集做预测
  #  prediction = model.predict(test_x)
    test_result = model.predict(test_data)
    sns.countplot(test_result, label="count")
    plt.title("SVM")
    plt.show()

    #print('准确率: ', metrics.accuracy_score(prediction, test_y))

#SVM(train_x, test_x, train_y, test_y,train_data,test_data)

#随机森林
def Randforest(train_x, test_x, train_y, test_y,train_data,test_data):
    rf = RandomForestClassifier(random_state=1, criterion='gini')

    '''
    parameters = {"n_estimators": range(1, 11)}
    # 使用GridSearchCV进行参数调优
    clf = GridSearchCV(estimator=rf, param_grid=parameters)
    # 对数据集进行分类
    clf.fit(train_x, train_y)
    print("最优分数： %.4lf" % clf.best_score_)
    print("最优参数：", clf.best_params_)
    '''

    rf.fit(train_x,train_y)
 #   pred_y= rf.predict(test_x)
    test_result= rf.predict(test_data)

    sns.countplot(test_result, label="count")
    plt.title("Random Forest")
    plt.show()


#Randforest(train_x, test_x, train_y, test_y,train_data,test_data)
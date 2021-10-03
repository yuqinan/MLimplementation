import pymysql
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

server = '127.0.0.1'
user = 'root'
passwd = 'gmcc1234'
databases = 'chnldm'
mydb = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="gmcc1234",
    database="chnldm",
    port=3306
)
mycursor = mydb.cursor()
mycursor.execute("select * from tw_usr_up_89_139_dm ")
result = mycursor.fetchall()
# 获取列名
cols = [i[0] for i in mycursor.description]
# sql内表转换pandas的DF
train_data = pd.DataFrame(np.array(result), columns=cols)
train_data['arpu_202008_10'].fillna(0, inplace=True)
train_data['mou_202008_10'].fillna(0, inplace=True)
train_data['b20dou_a5dou_fax'].fillna(0, inplace=True)
train_data['voice_fee_202008_10'].fillna(0, inplace=True)
train_data['bal_fee_202008_10'].fillna(0, inplace=True)
train_data['arpu_202010'].fillna(0, inplace=True)

print("sucess")
mydb.close()

###################################################################################################################

# 提取模型特征
features = ['innet_mo', 'arpu_202008_10', 'arpu_202010', 'mou_202008_10', 'dou_202008_10', 'is_arpu_up',
            'is_mou_up', 'is_dou_up', 'b20dou_a5dou_fax', 'gprs_fee_202008_10', 'voice_fee_202008_10',
            'bal_fee_202008_10', 'is_home',
            'is_family', 'is_vpmn']

# train_x-- feature
# train_y--labels
# 现有数据分组，分为75%训练集，25%测试集
train_x, test_x, train_y, test_y = train_test_split(train_data.loc[:,
                                                    ['innet_mo', 'arpu_202008_10', 'arpu_202010', 'mou_202008_10',
                                                     'dou_202008_10', 'is_arpu_up',
                                                     'is_mou_up', 'is_dou_up', 'b20dou_a5dou_fax', 'gprs_fee_202008_10',
                                                     'voice_fee_202008_10',
                                                     'bal_fee_202008_10', 'is_home',
                                                     'is_family', 'is_vpmn']],
                                                    train_data.loc[:, ['is_up']], test_size=0.25,
                                                    random_state=33)

'''
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)
'''
print("success")


###################################################################################################################

# 决策树算法
def reg_tree(train_x, test_x, train_y, test_y, train_data):
    # 特征值选择
    import matplotlib.pyplot as plt
    print("run!")
    train_features = train_data[features]
    train_labels = train_data['is_up']

    dvec = DictVectorizer(sparse=False)
    # 代码中使用了 fit_transform 这个函数，它可以将特征向量转化为特征值矩阵
    train_features = dvec.fit_transform(train_features.to_dict(orient='record'))

    from sklearn import tree

    # 构建ID3决策树
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=200000)
    # 决策树训练
    clf.fit(train_x, train_y)

    # 决策树预测
    pred_labels = clf.predict(test_x)

    from sklearn.model_selection import cross_val_score
    # 使用K折交叉验证 统计决策树准确率
    print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
    mse = mean_squared_error(test_y, pred_labels)
    print("决策树均方误差 = ", round(mse, 2))
    print("决策树预测结果", pred_labels)

    sns.countplot(test_y['is_up'], label="Count")
    plt.title("test-y")
    plt.show()

    sns.countplot(pred_labels, label="Count")
    plt.title("reg-tree")
    plt.show()

    from sklearn.tree import export_graphviz  # 通过graphviz绘制决策树
    with open('E:/pro/upgrade6.dot', 'w')as f:
        f = export_graphviz(clf, feature_names=['innet_mo', 'arpu_202008_10', 'arpu_202010', 'mou_202008_10',
                                                'dou_202008_10', 'is_arpu_up',
                                                'is_mou_up', 'is_dou_up', 'b20dou_a5dou_fax', 'gprs_fee_202008_10',
                                                'voice_fee_202008_10',
                                                'bal_fee_202008_10', 'is_home',
                                                'is_family', 'is_vpmn'], out_file=f)

    # export_graphviz第一个参数填入决策树的模型，feature_names填入参与的特征名，out_file即指定输出文件

    print("success")


reg_tree(train_x, test_x, train_y, test_y, train_data)

###################################################################################################################

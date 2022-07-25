#1读取数据
import pandas as pd
df = pd.read_excel('客户信息及违约表现.xlsx')

#2提取特征变量和目标变量
X = df.drop(columns='是否违约')
y = df['是否违约']

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型训练及搭建
from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(X_train, y_train)

#5模型预测及评估
#对测试集数据进行预测
y_pred = model.predict(X_test)
print(y_pred)

#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a)

#模型整体的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('预测准确度：' + str(score))

#预测属于各个分类的概率
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba)

#绘制roc曲线来评估模型的预测效果
from sklearn.metrics import roc_curve
fpr, tpr, thers = roc_curve(y_test, y_pred_proba[:, 1])
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
plt.plot(fpr, tpr)
plt.title('ROC')
plt.xlabel('假警报率')
plt.ylabel('命中率')
plt.show()

#计算模型的ACU值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('AUC:' + str(score))

#6分析数据特征的重要性，并对特征名称和特征重要性进行汇总
features = X.columns  # 获取特征名称
importances = model.feature_importances_  # 获取特征重要性

importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))

#7模型参数调优
from sklearn.model_selection import GridSearchCV
parameters = {'num_leaves': [10, 15, 31], 'n_estimators': [10, 20, 30], 'learning_rate': [0.05, 0.1, 0.2]}  #模型待调优参数的候选值范围


clf = LGBMClassifier()  #构建LGBMClassifier模型
grid_search = GridSearchCV(clf, parameters, scoring='roc_auc', cv=5)#将模型和待调优的参数候选值范围传入，模型的评估标准，交叉验证次数
grid_search.fit(X_train, y_train)   #传入训练集数据
print(grid_search.best_params_)     #输出参数最优值 

#获得参数最优值重新搭建模型
model = LGBMClassifier(n_estimators=10, num_leaves=10, learning_rate=0.2)
model.fit(X_train, y_train)

#查看新模型的AUC值
y_pred_proba = model.predict_proba(X_test)
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('新模型：' + str(score))


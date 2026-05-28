"""
项目：泰坦尼克号生存预测完整流程
核心能力：机器学习全流程、特征工程、模型评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# 1. 数据加载
df = pd.read_csv('data/titanic_data.csv')


# 2. 特征工程
# 2.1. 提取姓名中的头衔
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs'
})

# 2.2. 计算家庭规模
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 2.3. 创建是否独自旅行特征
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 2.4. 填充缺失值
# Age：使用头衔分组的中位数填充，更精细
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
# Fare：测试集有1个缺失值，用整体中位数填充
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# Embarked：训练集有2个缺失值，用众数填充
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# 创建新特征, ---是否有舱房记录（Cabin非空为1）
df['HasCabin'] = df['Cabin'].notna().astype(int)

# 2.5. 分箱处理
# 年龄分箱
df['AgeBin'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 30, 50, 100],
    labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
)

# 票价分箱（按四分位数分为4个等级）
df['FareBin'] = pd.qcut(
    df['Fare'],
    q=4,
    labels=['Low', 'Medium', 'High', 'VeryHigh']
)

# 2.6. 删除无关特征
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin',
             'Age', 'Fare', 'SibSp', 'Parch']
df.drop(columns=drop_cols, inplace=True)

# 2.7分离训练集和测试集
# 训练集：Survived 列非空的样本
train_processed = df[df['Survived'].notna()].copy()
# 测试集：Survived 列为空的样本，并删除该列
test_processed = df[df['Survived'].isna()].drop(columns='Survived').copy()

# 提取特征矩阵 (X) 和标签 (y)
X = train_processed.drop(columns='Survived')
y = train_processed['Survived'].astype(int)

# 划分训练集和验证集（80%训练，20%验证，分层抽样保持标签比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_test.shape[0]}')


# 3.数据预处理管道
# 3.1 特征分类
numeric_features = ['FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone','AgeBin', 'FareBin', 'HasCabin']
# 3.2 ColumnTransformer 对不同类型特征做不同处理：
# - 数值特征：标准化（去除量纲，均值为0，标准差为1）
# - 类别特征：独热编码（避免引入顺序关系），drop='first' 避免虚拟变量陷阱
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# 4. 多模型对比
# 4.1 定义模型
models = {
    '逻辑回归': LogisticRegression(random_state=42),
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
}
# 4.2 训练和评估模型
result = {}
for name, model in models.items():
    # 创建管道
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # 训练
    pipe.fit(X_train, y_train)
    
    # 预测
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]  # 正类（存活）概率
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    result[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    print(f"\n{name} 评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
# 5. 模型优化：随机森林网格搜索
# 5.1 以随机森林作为优化对象，定义超参数搜索空间
param_grid = {
    'classifier__n_estimators': [100, 200],      # 决策树数量
    'classifier__max_depth': [None, 10, 20],     # 树的最大深度
    'classifier__min_samples_split': [2, 5],     # 内部节点再划分所需最小样本数
    'classifier__min_samples_leaf': [1, 2]       # 叶子节点最少样本数
}
# 4.2 使用随机森林作为分类器的 Pipeline
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# 4.3 网格搜索 + 交叉验证
grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
print(f'最佳交叉验证准确率: {grid_search.best_score_:.4f}')

# 获取最佳模型
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print('\n优化后模型在验证集的表现:')
print(f'准确率: {accuracy_score(y_test, y_pred_best):.4f}')
print(classification_report(y_test, y_pred_best))


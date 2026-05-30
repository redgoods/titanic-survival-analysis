
#  泰坦尼克号生存预测——完整机器学习项目

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目基于 Kaggle 经典竞赛 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)，完整展示了数据科学项目的标准流程：  
**探索性数据分析（EDA） → 特征工程 → 多模型对比 → 超参数优化 → 模型评估**。

---

##  项目亮点
-  全面的 EDA：单变量/双变量/多变量可视化，发现性别、舱位、年龄等关键因素
-  丰富的特征工程：头衔提取、家庭规模、缺失值智能填充、分箱处理
- 三种分类器对比：逻辑回归、决策树、随机森林
- 网格搜索 + 交叉验证优化随机森林超参数
- 详尽的评估报告与特征重要性分析
- 可一键运行的脚本 `titanic-prediction.py`

---

## 仓库结构
```
titanic-survival-analysis/
├── data/
│   └── titanic_data.csv          # 训练数据集（891条）
├── notebooks/
│   └── data_preprocess.ipynb     # EDA 笔记本
├── images/                       # EDA 图表（可存放）
├── titanic-prediction.py         # 机器学习全流程脚本
├── README.md
└── requirements.txt
```

---

##  快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/redgoods/titanic-survival-analysis.git
cd titanic-survival-analysis
```

### 2. 安装依赖
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
或使用 requirements.txt（如有）：
```bash
pip install -r requirements.txt
```

### 3. 运行分析
- **EDA 分析**：打开 `notebooks/data_preprocess.ipynb` 并逐步执行，即可重现所有可视化图表和统计结论。
- **建模与评估**：直接运行脚本：
  ```bash
  python titanic-prediction.py
  ```
  脚本将自动完成数据加载、特征工程、模型训练、对比和超参数优化，并在终端打印详细的评估结果。

---

## 数据说明
| 特征 | 描述 | 备注 |
|------|------|------|
| PassengerId | 乘客编号 | 唯一标识 |
| Survived | 是否幸存 | 0=遇难, 1=幸存（目标变量） |
| Pclass | 客舱等级 | 1/2/3 |
| Name | 姓名 | 用于提取头衔 |
| Sex | 性别 | male / female |
| Age | 年龄 | 有缺失值 |
| SibSp | 船上兄弟姐妹/配偶数 | |
| Parch | 船上父母/子女数 | |
| Ticket | 船票编号 | 格式复杂，未直接使用 |
| Fare | 票价 | 有缺失值 |
| Cabin | 舱房号 | 缺失严重，已转为“是否有记录”特征 |
| Embarked | 登船港口 | C/Q/S，有缺失值 |

---

##  EDA 主要发现
| 影响因素 | 重要结论 |
|----------|----------|
| **性别** | 女性生存率 74%，男性仅 19% —— 妇孺优先原则体现明显 |
| **船舱等级** | 头等舱生存率 63%，三等舱仅 24% —— 阶层与生存机会强相关 |
| **年龄** | 儿童生存率更高，年轻人有一定优势 |
| **票价** | 与生存率正相关（实际映射社会地位） |
| **家庭大小** | 独自旅行者生存率相对较低 |

> 详细的可视化图表和统计检验，请查看 `notebooks/data_preprocess.ipynb`。

---

## 特征工程一览
| 新特征 | 构造方法 | 作用 |
|--------|----------|------|
| `Title` | 从 Name 中用正则提取，合并稀有头衔 | 代表社会身份 |
| `FamilySize` | SibSp + Parch + 1 | 家庭规模 |
| `IsAlone` | FamilySize == 1 | 是否独自出行 |
| `HasCabin` | Cabin 非空为 1 | 是否有舱房记录 |
| `AgeBin` | 年龄分 5 箱：Child/Teen/YoungAdult/Adult/Senior | 年龄段离散化 |
| `FareBin` | 票价四分位分 4 箱：Low/Medium/High/VeryHigh | 票价等级化 |

所有连续特征在建模前均经过标准化或独热编码处理，确保了模型稳定性。

---

## 模型表现
运行 `titanic-prediction.py` 后，在验证集（20% 数据）上得到以下指标：

| 模型 | Accuracy | Precision | Recall | F1-score | AUC |
|------|----------|-----------|--------|----------|-----|
| 逻辑回归 | **0.8324** | 0.7826 | 0.7826 | 0.7826 | 0.8613 |
| 决策树 | 0.7877 | 0.7541 | 0.6667 | 0.7077 | 0.7720 |
| 随机森林（默认） | 0.8045 | 0.7742 | 0.6957 | 0.7328 | 0.8140 |
| **随机森林（优化后）** | 0.7933 | 0.77 | 0.67 | 0.71 | - |

> 注意：本脚本仅使用训练集划分的训练/验证数据进行评估，未加载 Kaggle 提供的测试集，因此优化后模型在验证集上的准确率略低于默认逻辑回归属于正常波动。如需生成 Kaggle 提交文件，可扩展脚本读取 `test.csv`，用最佳模型预测并输出。

---

## 技术栈
- **数据分析**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **机器学习**：Scikit-learn（Pipeline, ColumnTransformer, GridSearchCV）
- **模型**：LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
- **运行环境**：Python 3.7+, Jupyter Notebook

---

## 改进方向
- [ ] 载入真实测试集并生成 `submission.csv`
- [ ] 添加 XGBoost / LightGBM 进行对比
- [ ] 尝试投票融合（VotingClassifier）或堆叠（Stacking）
- [ ] 使用 SHAP 进行模型解释
- [ ] 将 EDA 和建模流程合并为一个可交互的 Notebook

---

## 许可
本项目代码采用 [MIT License](https://opensource.org/licenses/MIT)。  
数据来源于 Kaggle，版权归原始提供者所有。

---


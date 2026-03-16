---
name: automl-skill
description: >
  AutoML 自动化机器学习技能 | Automated Machine Learning Skill. 
  基于 PyCaret 进行低代码机器学习建模，支持分类、回归、聚类、异常检测、时间序列预测、自然语言处理和关联规则挖掘等任务。
  未来将集成更多 AutoML 库（如 AutoGluon、FLAML 等）。
  当用户需要快速构建机器学习模型、自动化模型选择、超参数调优、模型集成、特征工程或进行 AutoML 实验时使用此技能。
  适用于数据科学家、公民数据科学家、机器学习工程师和希望快速原型开发的人员。
  触发关键词：AutoML、机器学习自动化、PyCaret、分类模型、回归模型、聚类、异常检测、时间序列、文本分类、模型调优、模型比较、特征选择。
  Trigger keywords in English: AutoML, automated machine learning, PyCaret, classification, regression, clustering, anomaly detection, time series forecasting, NLP, text mining, model tuning, model comparison, feature engineering.
---

# PyCaret AutoML 技能指南 | PyCaret AutoML Skill Guide

本技能帮助用户使用 PyCaret 快速构建端到端的机器学习工作流。PyCaret 是一个开源的低代码机器学习库，可以将数百行代码简化为几行。

This skill helps users build end-to-end machine learning workflows using PyCaret, an open-source low-code ML library that simplifies hundreds of lines of code into just a few lines.

## 核心功能 | Core Capabilities

- **自动化模型选择** - 自动比较多个模型并选择最佳模型
- **自动化超参数调优** - 使用 Optuna/Hyperopt 自动优化模型参数
- **自动化特征工程** - 自动进行数据预处理、特征转换和特征选择
- **模型集成** - 支持 Bagging、Boosting、Stacking、Blending
- **模型可解释性** - 支持 SHAP、Permutation Importance 等解释方法
- **模型部署就绪** - 生成可复现的生产级 Pipeline

## 支持的机器学习任务 | Supported ML Tasks

| 模块 | Module | 任务类型 | Task Type | 参考文档 |
|------|--------|----------|------------|----------|
| pycaret.classification | Classification | 二分类、多分类 | Binary, Multi-class | [classification.md](references/classification.md) |
| pycaret.regression | Regression | 回归预测 | Regression | [regression.md](references/regression.md) |
| pycaret.clustering | Clustering | 无监督聚类 | Unsupervised Clustering | [clustering.md](references/clustering.md) |
| pycaret.anomaly | Anomaly Detection | 异常检测 | Outlier Detection | [anomaly.md](references/anomaly.md) |
| pycaret.time_series | Time Series | 时间序列预测 | Time Series Forecasting | [time_series.md](references/time_series.md) |
| pycaret.nlp | NLP | 文本分类、主题建模 | Text Classification, Topic Modeling | [nlp.md](references/nlp.md) |
| pycaret.arules | Association Rules | 关联规则挖掘 | Market Basket Analysis | [association_rules.md](references/association_rules.md) |

---

## 快速开始 | Quick Start

### 1. 选择您的任务类型

根据您的机器学习任务，选择相应的模块：

- **分类问题** → 使用 `pycaret.classification`
- **回归问题** → 使用 `pycaret.regression`  
- **客户分群** → 使用 `pycaret.clustering`
- **异常检测** → 使用 `pycaret.anomaly`
- **时间预测** → 使用 `pycaret.time_series`
- **文本分析** → 使用 `pycaret.nlp`
- **购物篮分析** → 使用 `pycaret.arules`

### 2. 标准工作流 | Standard Workflow

每个 PyCaret 模块遵循相似的工作流程：

```python
# 1. 导入模块
from pycaret.classification import *

# 2. 加载数据（可选：使用内置数据集）
data = get_data('breast_cancer')

# 3. 初始化环境 (setup) - 数据预处理
clf = setup(data, target='target', session_id=42)

# 4. 比较模型 (compare_models)
best_model = compare_models()

# 5. 创建模型 (create_model)
model = create_model('lr')

# 6. 调优模型 (tune_model)
tuned_model = tune_model(model)

# 7. 集成模型 (ensemble_model)
ensemble = ensemble_model(tuned_model)

# 8. 模型评估 (evaluate_model)
evaluate_model(ensemble)

# 9. 预测 (predict_model)
predictions = predict_model(ensemble, data=test_data)

# 10. 保存模型 (save_model)
save_model(ensemble, 'best_model_pipeline')
```

---

## 通用 API 参考 | Common API Reference

详细内容请参考 [utilities.md](references/utilities.md)

### 数据加载

```python
from pycaret.classification import get_data

# 列出数据集
all_datasets = get_data('index')

# 加载数据集
data = get_data('breast_cancer')
```

### 配置管理

```python
from pycaret.classification import get_config, set_config

# 获取配置
X_train = get_config('X_train')

# 设置配置
set_config('seed', 123)
```

### 模型操作

```python
# 比较模型
best = compare_models()

# 创建模型
model = create_model('rf')

# 调优模型
tuned = tune_model(model)

# 集成
ensemble = ensemble_model(model)

# 预测
predictions = predict_model(model, data=new_data)

# 保存/加载
save_model(model, 'my_model')
loaded = load_model('my_model')
```

---

## 详细文档索引 | Detailed Documentation Index

| 模块 | 包含内容 | 文件 |
|------|----------|------|
| 参数深度分析 | setup参数选择指南、决策树、实战配置 | [setup_parameters_deep_dive.md](references/setup_parameters_deep_dive.md) |
| Classification | setup 参数、模型列表、评估指标、工作流 | [classification.md](references/classification.md) |
| Regression | setup 参数、回归模型、评估指标、工作流 | [regression.md](references/regression.md) |
| Time Series | 时间序列特有参数、预测、季节性 | [time_series.md](references/time_series.md) |
| Clustering | 聚类算法、轮廓系数、分配标签 | [clustering.md](references/clustering.md) |
| Anomaly | 异常检测算法、可视化 | [anomaly.md](references/anomaly.md) |
| NLP | 主题模型、文本处理、词云 | [nlp.md](references/nlp.md) |
| Association Rules | 关联规则、支持度、置信度 | [association_rules.md](references/association_rules.md) |
| Utilities | 通用函数、部署、应用生成 | [utilities.md](references/utilities.md) |

---

## 代码模板 | Code Templates

### 分类任务模板

```python
from pycaret.classification import *

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

clf = setup(data, target='target', train_size=0.8)
best = compare_models()
tuned = tune_model(best)
ensemble = ensemble_model(tuned)
predictions = predict_model(ensemble, data=test)
save_model(ensemble, 'classifier')
```

### 回归任务模板

```python
from pycaret.regression import *

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

reg = setup(data, target='price', normalize=True)
best = compare_models()
tuned = tune_model(best, optimize='RMSE')
predictions = predict_model(tuned, data=test)
save_model(tuned, 'regressor')
```

### 时间序列模板

```python
from pycaret.time_series import *

data = get_data('airline')
ts = setup(data, fh=12, seasonal_period=12)
best = compare_models()
model = create_model('arima')
predictions = predict_model(model, fh=24)
```

---

## 最佳实践 | Best Practices

1. **数据预处理**: 使用 `normalize=True`, `remove_outliers=True` 等参数
2. **模型选择**: 用 `compare_models(turbo=True)` 快速验证
3. **超参数调优**: 根据时间预算设置 `n_iter`
4. **模型集成**: 复杂任务使用 `ensemble_model` 或 `stack_models`
5. **生产部署**: 使用 `finalize_model()` 在全量数据上训练

---

## 故障排除 | Troubleshooting

- **内存不足**: 减少 `n_iter`, 使用 `turbo=True`, 减少 `fold`
- **特征维度太高**: 使用 `pca=True` 或 `feature_selection=True`
- **类别不平衡**: 使用 `fix_imbalance=True`
- **文本处理失败**: 确保文本列是字符串类型

---

## PyCaret 版本信息 | Version Information

当前文档基于 PyCaret 3.0 版本。

- 官方文档: https://pycaret.gitbook.io/docs
- API 文档: https://pycaret.readthedocs.io/

如需了解特定模块的详细参数，请参考 references 目录下的相应文档。

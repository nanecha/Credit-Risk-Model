# Task-1

# Credit Scoring Business Understanding

In this task, I attempted to gain a solid fundamental grasp of credit risk by thoroughly examining a variety of reference resources supplied.  During this process, I learned about fundamental concepts such as the definition of credit risk, risk assessment methods, RFM, the role of regulatory frameworks such as the Basel II Accord, and various modeling approaches in the sector.

- Credit Risk : According to the Basel III framework, credit risk is defined as the potential that a bank borrower or counterparty will fail to meet its obligations in accordance with agreed terms

 The development of a credit scoring model is guided by both regulatory requirements and business needs, addressing risk measurement, data limitations, and model interpretability. Below, we summarize key considerations for our approach:

## 1. Regulatory Influence: Basel II and Model Interpretability

In order to guarantee sufficient capital buffers, the Basel II Accord highlights how crucial it is to measure credit risk precisely.  Credit scoring models that are not only predictive but also comprehensible, accessible, and thoroughly documented are required by this regulatory focus.  In order to ensure compliance and support efficient risk management, models must enable auditors and financial institutions to comprehend the process used to make credit decisions.  Model selection and documentation procedures are thus greatly impacted by the requirement for lucid thinking and outputs that can be justified.

## 2. Proxy Variable for Default Risk and Its Implications

In order to guarantee sufficient capital buffers, the Basel II Accord highlights how crucial it is to measure credit risk precisely.  Credit scoring models that are not only predictive but also comprehensible, accessible, and thoroughly documented are required by this regulatory focus.  In order to ensure compliance and support efficient risk management, models must enable auditors and financial institutions to comprehend the process used to make credit decisions.  Model selection and documentation procedures are thus greatly impacted by the requirement for lucid thinking and outputs that can be justified.

## 3. Trade-offs: Interpretability vs. Predictive Performance

In a regulated environment, there's a critical balance between model performance and interpretability:

- Simple models (e.g., Logistic Regression with Weight of Evidence) offer transparency, explainability, and regulatory acceptance, but may sacrifice predictive power.

- Complex models (e.g., Gradient Boosting Machines) often achieve higher accuracy, yet pose challenges in terms of explainability, validation, and governance.

Ultimately, the choice depends on the institution’s risk tolerance, regulatory environment, and the need for model governance. In many cases, a hybrid approach—using complex models for exploration and simple models for deployment—strikes the best balance.

## References

- [Statistica Paper: Credit Scoring](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)

- [Alternative Credit Scoring (HKMA)](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)


# 📊 Task-2: Exploratory Data Analysis (EDA)
This task performs an in-depth exploratory data analysis on the raw credit risk dataset to understand its structure, quality, and underlying patterns. EDA is a critical step that helps guide feature selection, engineering, and modeling decisions in the credit scoring pipeline.

## 1. Install requirement

- see requirements

## 🗂️  2.Initial Data Overview
Check the structure, size, and data samples

## 🔢  3.Data Types and Summary Statistics

## 📈 4. Distribution of Numerical Features

## ➤ 5.Summary Statistics (Numerical)

These statistics include:

Mean, standard deviation

Min, max values

Quartiles (25%, 50%, 75%)

# 🧩 6 Identify Numerical and Categorical Features


# 🔍 7 Correlation Matrix (Numerical Features)

# 🧩 8 Missing Values

# 🚨 9 Outlier Detection

# Summary of EDA Insights
- The dataset includes both numerical and categorical features.

- Certain columns may have skewed distributions or contain outliers.

- Some features are highly correlated and may need transformation or selection.

- Missing data and class imbalance (if any) will require treatment.

- Categorical features exhibit varying levels of cardinality, affecting encoding strategies.
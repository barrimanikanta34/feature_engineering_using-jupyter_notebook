
# 🧠 Feature Engineering with Python

🚀 Transforming Raw Data into Machine Learning-Ready Features

A practical, hands-on collection of Python notebooks covering essential Feature Engineering techniques for Machine Learning.

# 📌 About This Repository

Feature Engineering is one of the most important steps in a Machine Learning workflow.

Real-world datasets are rarely clean and ready to use. They often contain:

❌ Missing values  
⚠️ Imbalanced classes  
📊 Outliers  
🔤 Categorical variables  
📉 Uneven data distributions

This repository demonstrates how to identify and handle these problems using Python and popular Machine Learning libraries.

The notebooks are designed to be practical and beginner-friendly, with examples, visualizations, explanations, and implementation using real and simulated datasets.

# 🎯 What You'll Learn

By working through this repository, you will learn how to:

🔍 Identify missing values  
🧹 Handle missing data using imputation techniques  
⚖️ Understand imbalanced datasets  
⬆️ Perform up-sampling  
⬇️ Understand down-sampling  
🧬 Generate synthetic minority samples using SMOTE  
📦 Detect and understand outliers  
📐 Calculate Q1, Q3 and IQR  
🚨 Identify lower and upper fences  
🔤 Convert categorical variables into numerical features  
🏷️ Apply One-Hot Encoding  
🤖 Prepare datasets for Machine Learning models

# 📚 Notebook Collection

## 1️⃣ Handling Missing Data

📓 Notebook: [01. Feature Engineering- Handling Missing Data ](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-/blob/main/01.%20Feature%20Engineering-%20Handling%20Missing%20Data.ipynb)

Missing values are extremely common in real-world datasets.

This notebook demonstrates how to:

* Detect missing values using isnull()  
* Calculate the number of missing values  
* Understand the impact of dropping missing records  
* Remove missing values using dropna()  
* Perform column-wise missing-value handling  
* Apply Mean Value Imputation  
* Visualize distributions using Seaborn  

### 🛠️ Libraries Used  

import pandas as pd  
import seaborn as sns 

### 📊 Dataset

The notebook uses the Titanic dataset available through Seaborn.

## 2️⃣ Handling Imbalanced Dataset

📓 Notebook: [02. Feature Engineering- Handling Imbalanced Dataset](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-/blob/main/02.%20Feature%20Engineering-%20Handling%20Imbalanced%20Dataset.ipynb)

A classification dataset is called imbalanced when one class contains significantly more observations than another.

For example:

Class 0 → 900 samples  
Class 1 → 100 samples

This creates a 9:1 class imbalance.

The notebook demonstrates:

* Creating an imbalanced dataset  
* Understanding majority and minority classes  
* Inspecting class distribution  
* Separating majority and minority classes  
* Performing Up-Sampling  
* Using sklearn.utils.resample

🔑 Key Concept  

 ### 🔑 Key Concept

```text
Imbalanced Dataset
        │
        ├── Majority Class
        │
        └── Minority Class
                 │
                 ▼
          Resampling Techniques

```


## 3️⃣ SMOTE — Synthetic Minority Oversampling Technique

📓 Notebook: [03. Feature Engineering-SMOTE](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-/commit/ae148e356e53e7df6a18c20c7e13fb8b2cc6289f)

SMOTE is a powerful technique used to handle imbalanced datasets.

### 🔬 Workflow

               Imbalanced Dataset  
                       ↓  
             Identify Minority Class  
                       ↓
              Find Nearest Neighbor
                       ↓
            Generate Synthetic Samples  
                       ↓
                Balanced Dataset

The notebook covers:

* Creating classification datasets
* Visualizing class distributions
* Understanding synthetic data generation
* Installing and using imbalanced-learn
* Applying SMOTE
* Checking the resulting dataset shape

### 🛠️ Main Library

```python
from imblearn.over_sampling import SMOTE
```

### ⚙️ Example

```python
oversample = SMOTE(
    k_neighbors=2,
    random_state=42
)

X, y = oversample.fit_resample(
    final_df[['f1', 'f2']],
    final_df['target']
)
```
## 4️⃣ Handling Outliers with Python

📓 Notebook: [04. Handling Outliers with Python](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-/blob/main/04.%20Handling%20Outliers%20with%20Python.ipynb)

Outliers are observations that are significantly different from the majority of the data.

This notebook introduces the Interquartile Range (IQR) method for detecting outliers.

### 📐 Important Concepts

Q1 = First Quartile   
Q2 = Median   
Q3 = Third Quartile  

IQR = Q3 - Q1  

### 🚨 Outlier Detection

Lower Fence = Q1 - 1.5 × IQR

Upper Fence = Q3 + 1.5 × IQR

Values outside these boundaries can be considered potential outliers.

The notebook also uses visualization to better understand the distribution of the data.

## 5️⃣ Data Encoding — Nominal, Label, Ordinal & Target Guided Encoding

📓 Notebook: [05. Data Encoding-Nomianl_One Hot Encoding](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-/blob/main/05.%20Data%20Encoding-Nomianl_One%20Hot%20Encoding%20.ipynb)

Machine Learning algorithms generally work with numerical data.  
However, real-world datasets often contain **categorical features** such as colors, sizes, cities, gender, smoker status, etc.

**Data Encoding** is the process of converting categorical values into numerical representations that can be used by Machine Learning models. :contentReference[oaicite:1]{index=1}

---

### 🎯 What You'll Learn

This notebook covers three important encoding techniques:

- 🔤 **Nominal / One-Hot Encoding**
- 🏷️ **Label Encoding**
- 📊 **Ordinal Encoding**
- 🎯 **Target Guided Ordinal Encoding**

---

### 🔤 1. Nominal / One-Hot Encoding

One-Hot Encoding converts each categorical value into a separate binary feature.

For example, a `color` feature containing:

```text
Red
Green
Blue
```

can be represented as:

```text
Red   → [1, 0, 0]
Green → [0, 1, 0]
Blue  → [0, 0, 1]
```

Each category gets its own binary column. :contentReference[oaicite:2]{index=2}

### 🔑 Key Concept

```text
        Categorical Data
               │
               ▼
       ┌─────────────────┐
       │ One-Hot Encoder │
       └────────┬────────┘
                │
                ▼
       Numerical Features
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
      Blue    Green     Red
       0        0        1
```

### 🛠️ Main Library

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
```

The notebook creates a simple DataFrame containing `red`, `blue`, and `green` categories and applies `OneHotEncoder`. :contentReference[oaicite:3]{index=3}

### ⚙️ Example

```python
encoder = OneHotEncoder()

encoded = encoder.fit_transform(
    df[['color']]
).toarray()
```

The encoded columns are generated using:

```python
encoder.get_feature_names_out()
```

producing columns such as:

```text
color_blue
color_green
color_red
```

The notebook then combines the encoded features with the original DataFrame using `pd.concat()`. :contentReference[oaicite:4]{index=4}

---
### ⚠️ Limitation of One-Hot Encoding

One-Hot Encoding can create a large number of features when a categorical variable contains many unique categories.

For example:

```text
100 Categories
      │
      ▼
100 Binary Features
      │
      ▼
Sparse Matrix
```

The notebook specifically highlights the increase in features and sparsity as a drawback. :contentReference[oaicite:5]{index=5}

---

### 🆕 Encoding New Data

Once the encoder has been fitted, it can also transform new categorical data.

```python
encoder.transform(
    [['blue']]
).toarray()
```

This allows new observations to be converted using the same encoding scheme. :contentReference[oaicite:6]{index=6}

---

### 🏷️ 2. Label Encoding

**Label Encoding** assigns a numerical value to each category.

For example:

```text
Red   → 2
Green → 1
Blue  → 0
```

In the notebook, `LabelEncoder` is used to convert the `color` feature into numerical labels. :contentReference[oaicite:7]{index=7}

### 🛠️ Main Library

```python
from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()
```

### ⚙️ Example

```python
lbl_encoder.fit_transform(
    df[['color']]
)
```

The notebook produces encoded values such as:

```text
red   → 2
blue  → 0
green → 1
```

and demonstrates transforming individual values such as `red` and `green`. :contentReference[oaicite:8]{index=8}

### ⚠️ Important Consideration

Assigning numerical values to categories can introduce an apparent numerical relationship between categories.

For example:

```text
Blue  → 0
Green → 1
Red   → 2
```

A Machine Learning model could potentially interpret `Red (2)` as greater than `Green (1)` and `Blue (0)`.

Therefore, the notebook notes that label encoding is appropriate when a **rank-based assignment** is meaningful. :contentReference[oaicite:9]{index=9}

---

### 📊 3. Ordinal Encoding

**Ordinal Encoding** is used when categorical values have an **intrinsic order or ranking**. :contentReference[oaicite:10]{index=10}

For example:

```text
Small
  ↓
Medium
  ↓
Large
```

can be represented as:

```text
Small  → 0
Medium → 1
Large  → 2
```

### 🛠️ Main Library

```python
from sklearn.preprocessing import OrdinalEncoder
```

### ⚙️ Example

```python
encoder = OrdinalEncoder(
    categories=[['small', 'medium', 'large']]
)

encoded = encoder.fit_transform(
    df[['size']]
)
```

The notebook explicitly defines the category order as:

```text
small → 0
medium → 1
large → 2
```

and demonstrates transforming a new value such as `small`. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

---

### 🎯 4. Target Guided Ordinal Encoding

**Target Guided Ordinal Encoding** encodes a categorical variable based on its relationship with a target variable.

The notebook describes replacing each category with a numerical value based on the **mean or median of the target variable for that category**. :contentReference[oaicite:13]{index=13}

### 🔬 Workflow

```text
        Categorical Feature
                │
                ▼
          Target Variable
                │
                ▼
      Calculate Group Mean
                │
                ▼
        Create Mapping
                │
                ▼
      Replace Categories
                │
                ▼
       Numerical Feature
```

### 📊 Example

The notebook uses:

```text
City       Price
----------------
New York    200
London      150
Paris       300
Tokyo       250
New York    100
Paris       320
```

Here:

- `city` → categorical feature
- `price` → target variable :contentReference[oaicite:14]{index=14}

The mean target value for each city is calculated:

```python
mean_price = df.groupby(
    'city'
)['price'].mean().to_dict()
```

Result:

```text
London   → 150.0
New York → 150.0
Paris    → 310.0
Tokyo    → 250.0
```

:contentReference[oaicite:15]{index=15}

The categorical city values are then mapped to their corresponding numerical values:

```python
df['city_encoded'] = df['city'].map(
    mean_price
)
```

This creates a new numerical feature called `city_encoded`. :contentReference[oaicite:16]{index=16}

---

### 🧪 Practical Example with the Tips Dataset

The notebook also applies target-guided encoding concepts to the **Seaborn `tips` dataset**. :contentReference[oaicite:17]{index=17}

It selects:

```python
new_df = df[['time', 'total_bill']]
```

and calculates the mean `total_bill` for each `time` category:

```python
mean_price_of_total_bill = (
    df.groupby('time')['total_bill'].mean()
)
```

The resulting means shown in the notebook are approximately:

```text
Lunch   → 17.168676
Dinner  → 20.797159
```

:contentReference[oaicite:18]{index=18}

---

### 🧠 Encoding Techniques at a Glance

| Encoding Technique | Suitable For | Example |
|---|---|---|
| 🔤 One-Hot Encoding | Nominal categories | Color |
| 🏷️ Label Encoding | Converting categories to labels | Color |
| 📊 Ordinal Encoding | Ordered categories | Small → Medium → Large |
| 🎯 Target Guided Encoding | Categories related to a target | City → Mean Price |

---

### 🔄 Data Encoding Workflow

```text
                 CATEGORICAL DATA
                        │
                        ▼
              Identify Category Type
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        No Natural Order       Has Natural Order
             │                     │
             ▼                     ▼
       One-Hot Encoding      Ordinal Encoding
             │
             ▼
       Multiple Binary
          Features

                 OR

          Target Relationship
                 │
                 ▼
      Target Guided Encoding
                 │
                 ▼
        Numerical Feature
```

---

### 🌟 Key Takeaways

After completing this chapter, you will understand:

- 🔤 How One-Hot Encoding converts categories into binary features
- 🏷️ How Label Encoding assigns numerical labels
- 📊 When Ordinal Encoding should be used
- 🎯 How Target Guided Encoding uses a target variable
- ⚠️ Why One-Hot Encoding can create many features
- 🧠 Why numerical labels can introduce unintended relationships
- 🐍 How to implement these techniques using `scikit-learn` and `pandas`

**Data Encoding transforms categorical information into numerical features that can be used in Machine Learning workflows.** 🚀

## 🗂️ Repository Structure

📦 01.Feature Engineering with python  
│  
├── 📓 01. Feature Engineering- Handling Missing Data.ipynb  
│  
├── 📓 02. Feature Engineering- Handling Imbalanced Dataset.ipynb  
│  
├── 📓 03. Feature Engineering-SMOTE.ipynb  
│  
├── 📓 04. Handling Outliers with Python.ipynb  
│  
├── 📓 05. Data Encoding-Nomianl_One Hot Encoding .ipynb  
│  
└── 📄 README.md

## 🧰 Technologies & Libraries

| Technology | Purpose |
|---|---|
| 🐍 **Python** | Programming language |
| 📓 **Jupyter Notebook** | Interactive development |
| 🐼 **Pandas** | Data manipulation |
| 🎨 **Seaborn** | Dataset loading and data analysis |
| 🤖 **Scikit-learn** | Data encoding and Machine Learning utilities |

## 🧭 Feature Engineering Roadmap

```text
                         RAW DATA
                            │
                            ▼
                ┌─────────────────────┐
                │ Data Understanding  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Missing Value      │
                │      Check          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Missing Data        │
                │ Handling            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Outlier Detection   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Categorical         │
                │ Encoding            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Class Imbalance     │
                │ Handling            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  ML-Ready Dataset   │
                └─────────────────────┘
```

## 📋 Key Feature Engineering Techniques

| Technique | Problem Addressed | Approach |
|---|---|---|
| 🧹 **Missing Value Imputation** | Missing observations | Mean Imputation |
| ⬆️ **Up-Sampling** | Class imbalance | Resampling |
| 🧬 **SMOTE** | Class imbalance | Synthetic samples |
| 🚨 **IQR Method** | Outliers | Statistical boundaries |
| 🔤 **One-Hot Encoding** | Categorical data | Binary features |

## 🚀 Getting Started

1. Clone the Repository
git clone [📘 View Repository](https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-) https://github.com/Ganesh-Ganga/Feature-Engineering-Techniques-.git  
2. Navigate to the Project  
***cd "01.Feature Engineering with python"***
3. Install Required Libraries  
***pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter***


## 💡 Why Feature Engineering Matters

A Machine Learning model is only as good as the data provided to it.

Good feature engineering can help models:

* 📈 Improve predictive performance
* 🎯 Learn meaningful patterns
* 🧹 Work with cleaner datasets
* ⚖️ Handle class imbalance
* 🚫 Reduce the impact of extreme values
* 🔢 Understand categorical information

Feature engineering bridges the gap between raw data and machine learning.

## 🤝 Contributions

Suggestions, improvements, and additional feature-engineering techniques are welcome!

If you find something useful in this repository, feel free to ⭐ Star the project.

## 👨‍💻 Author

Ganesh Ganga 


### 📌 Data Science & Machine Learning Enthusiast

*Turning raw data into meaningful features — one dataset at a time.* 🚀

---


⭐ **If this repository helped you learn Feature Engineering, consider giving it a star!**

**Happy Learning & Keep Building! 🚀🐍🤖**
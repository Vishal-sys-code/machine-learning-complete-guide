# **FEATURE ENGINEERING**
This is the process of transforming the raw data into a format that is easier to understand and use.
It has different process. Some of them are:
1. **Feature Transformation**
    a. *Missing Value Imputation*
    b. *Handling Categorical Features*
    c. *Outlier Detection*
    d. *Scaling*
    e. *Column Transformer*
    f. *Function Transformer*
    g. *Binning and Binarization*
2. **Feature Construction**
3. **Feature Splitting**

## Feature Transformation
Feature Transformation is the process of modifying or changing the original features of a dataset in order to improve the performance of the ML models.
Some of the techniques used in feature transformation are:
* Missing Value Imputation
* Handling Categorical Features
* Outlier Detection
* Scaling

**Missing Value Imputation :**
To choose the right method is always important.
Missing Value Imputation can be done by:
* Using Mean or Median if the data is not too skewed
* Using Median if the data is skewed or outliers
* Using Mode Imputation for categorical data
* Use Forward Fill or Backward Fill for time series data
* KNN Imputation if relationship between missing values and other features is complex.

**Handling Categorical Features :**
Categorical Features can be handled by:
* Label Encoding
* One Hot Encoding
* Ordinal Encoding

In deep, Categorical Features can be handled by:
* **Nominal Encoding:** One Hot Encoding, One Hot Encoding (Multiple Categories) , Mean Encoding
* **Ordinal Encoding:** Label Encoding, Target Guided Ordinal Encoding

**Outlier Detection :**
Outliers can be treated by:
* Trimming
* Capping
* Treating Outliers as a missing value
* Discretization

Outliers can be detected by:
* **Visual Method:** Box Plot and Scatter Plot
* **Statistical Method:** Z-Score(Standard Score), IQR (Inter-quartile Range), Winsorization.

**Scaling :**
Scaling can be done by:
* Standardization
* Normalization
* Min Max Scaling

**Column Transformer :**
There is a scikit library called ColumnTransformer which can be used to apply different transformations to different columns.
It is used to create and apply seperate transformers for numarical and categorical data.

```python
transformer = ColumnTransformer(transformers = [
    ('tnf-1', SimpleImputer, ['column_name']),
    ('tnf-2', OrdinalEncoder, categories = [['CN1', 'CN2']], ['column_name'])
    .....
    .....
    .....
])
```

**Function Transformer :**
It is used to normally distribute the data. If your data is skewed, you can use this transformer to make it normally distributed.

Some of the methods to do it: 
* Log transformation
* Reciprocal transformation
* Square root transformation
* Box-Cox transformation
* Yeo-Johnson transformation

In sklearn, Function Transformer can be used for: log, reciprocal, square root. Power Transformer can be used for Box-Cox and Yeo-Johnson transformation.

**Binning and Binarization :**
Some of the datasets do contain irregular values in number. There are two techniques for handling this numerical to the categorical data:
* Binning or Discretization
* Binarization
* K-means Binning
---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.3
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
```{=html}
<h2 align=center>Tumor Diagnosis (Part 1): Exploratory Data Analysis</h2>
```
`<img src="https://storage.googleapis.com/kaggle-datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-cover.jpg">`{=html}
:::

::: {.cell .markdown}
### About the Dataset:
:::

::: {.cell .markdown}
The [Breast Cancer Diagnostic
data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
is available on the UCI Machine Learning Repository. This database is
also available through the [UW CS ftp
server](http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/).

Features are computed from a digitized image of a fine needle aspirate
(FNA) of a breast mass. They describe characteristics of the cell nuclei
present in the image. n the 3-dimensional space is that described in:
\[K. P. Bennett and O. L. Mangasarian: \"Robust Linear Programming
Discrimination of Two Linearly Inseparable Sets\", Optimization Methods
and Software 1, 1992, 23-34\].
:::

::: {.cell .markdown}
**Attribute Information**:

-   ID number
-   Diagnosis (M = malignant, B = benign) 3-32)
:::

::: {.cell .markdown}
Ten real-valued features are computed for each cell nucleus:

1.  radius (mean of distances from center to points on the perimeter)
2.  texture (standard deviation of gray-scale values)
3.  perimeter
4.  area
5.  smoothness (local variation in radius lengths)
6.  compactness (perimeter\^2 / area - 1.0)
7.  concavity (severity of concave portions of the contour)
8.  concave points (number of concave portions of the contour)
9.  symmetry
10. fractal dimension (\"coastline approximation\" - 1)

The mean, standard error and \"worst\" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field 13
is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant
:::

::: {.cell .markdown}
### Task 1: Loading Libraries and Data
:::

::: {.cell .code execution_count="3"}
``` python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library  
import matplotlib.pyplot as plt
import time
```
:::

::: {.cell .code execution_count="5"}
``` python
data = pd.read_csv('data/data.csv')
```
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
```{=html}
<h2 align=center> Exploratory Data Analysis </h2>
```

------------------------------------------------------------------------
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 2: Separate Target from Features

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="6"}
``` python
data.head()
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
```
:::
:::

::: {.cell .code execution_count="7"}
``` python
col = data.columns
print (col)
```

::: {.output .stream .stdout}
    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
          dtype='object')
:::
:::

::: {.cell .code execution_count="10"}
``` python
y = data.diagnosis
drop_cols = ['Unnamed: 32','id','diagnosis']
x = data.drop(drop_cols, axis = 1)
x.head()
```

::: {.output .execute_result execution_count="10"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 3: Plot Diagnosis Distributions

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="12"}
``` python
ax = sns.countplot(y, label = "Count")
B, M = y.value_counts()
print('Number of Benign Trumors', B)
print('Number of Malignant Tumors', M)
```

::: {.output .stream .stdout}
    Number of Benign Trumors 357
    Number of Malignant Tumors 212
:::

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/7eaaa365ca9c5abe9b3219fa76922aadf8a89ae1.png)
:::
:::

::: {.cell .code execution_count="13"}
``` python
x.describe()
```

::: {.output .execute_result execution_count="13"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>
```
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
```{=html}
<h2 align=center> Data Visualization </h2>
```

------------------------------------------------------------------------
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 4: Visualizing Standardized Data with Seaborn

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="17"}
``` python
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/27989ebc7ca71487c7e2cdc82aec86460c8eabd5.png)
:::
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 5: Violin Plots and Box Plots

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="18"}
``` python
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/b363237c43d7103547c91523b253dae29796830e.png)
:::
:::

::: {.cell .code execution_count="19"}
``` python
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 20:30]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/45b083d89a90c155a07470c5db6f5e9d7f8f460b.png)
:::
:::

::: {.cell .code execution_count="20"}
``` python
sns.boxplot(x='features', y='value', hue='diagnosis', data=data)
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/db2ea207a56bcb069fcb80464b4cabbdd0691222.png)
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 6: Using Joint Plots for Feature Comparison

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="21"}
``` python
sns.jointplot(x.loc[:, 'concavity_worst'],
            x.loc[:, 'concave points_worst'],
            kind='regg',
            color='#ce1414');
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/c8ae2132452fda72a923ae0f88de49c8d5c45b60.png)
:::
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 7: Observing the Distribution of Values and their Variance with Swarm Plots

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="23"}
``` python
sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/9406b7334d109901279611757101a9067d8867a1.png)
:::
:::

::: {.cell .code execution_count="24"}
``` python
sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/e9cb2477cf5e43150bb6bd72daa2b65b4153054d.png)
:::
:::

::: {.cell .code execution_count="26"}
``` python
sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 20:30]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/be82c094bfa267e43a178e8e3bcf9ddee28a6d07.png)
:::
:::

::: {.cell .markdown}
:::

::: {.cell .markdown}
### Task 8: Observing all Pair-wise Correlations

------------------------------------------------------------------------

Note: If you are starting the notebook from this task, you can run cells
from all the previous tasks in the kernel by going to the top menu and
Kernel \> Restart and Run All \*\*\*
:::

::: {.cell .code execution_count="29"}
``` python
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(x.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax);
```

::: {.output .display_data}
![](vertopal_103f7d03929c46dfa85ac0e2befb0a53/aebb95966d53b1f6baf620a62f1b08b385382b1d.png)
:::
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .code}
``` python
```
:::

::: {.cell .code}
``` python
```
:::

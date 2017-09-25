---
layout: post
title:  "Python Data Analysis with pandas"
author: "Mubaris NK"
---

Python is a great language for data analysis. **pandas** is a Python package providing fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python.

In this post we'll get to know more about doing data analysis using pandas.

Mainly pandas has two data structures, **series** and **dataframes**.

## pandas Series

pandas series can be used for *one-dimensional labeled array*.


```python
import pandas as pd
index_list = ['MIT', 'Stanford', 'Harvard', 'Caltech', 'Cambridge', 'Oxford', 'UCL']
a = pd.Series([100, 98.7, 98.4, 97.7, 95.6, 95.3, 94.6],
              index=index_list)
```


```python
print(a)
```




    MIT          100.0
    Stanford      98.7
    Harvard       98.4
    Caltech       97.7
    Cambridge     95.6
    Oxford        95.3
    UCL           94.6
    dtype: float64



Labels can accessed using `index` attribute


```python
print(a.index)
```




    Index(['MIT', 'Stanford', 'Harvard', 'Caltech', 'Cambridge', 'Oxford', 'UCL'], dtype='object')



You can use array indexing or labels to access data in the series


```python
print(a[1])
print(a['Cambridge'])
```

    98.7
    95.6


You can also apply mathematical operations on pandas series.


```python
b = a * 2
c = a ** 1.5
print(b)
print()
print(c)
```

    MIT          200.0
    Stanford     197.4
    Harvard      196.8
    Caltech      195.4
    Cambridge    191.2
    Oxford       190.6
    UCL          189.2
    dtype: float64
    
    MIT          1000.000000
    Stanford      980.563513
    Harvard       976.096258
    Caltech       965.699142
    Cambridge     934.731414
    Oxford        930.334981
    UCL           920.103546
    dtype: float64


You can even create a series of heterogeneous data.


```python
s = pd.Series(['random', 1.2, 3, 'data'], index=['any', 'thing', 2, '4.3'])
print(s)
```




    any      random
    thing       1.2
    2             3
    4.3        data
    dtype: object



## pandas DataFrame

pandas DataFrame is a *2-dimensional labeled data structure*. There are many methods to create DataFrames. We'll see each one by one.

### Creating DataFrame from dictionary of Series

The following method can used to create DataFrames from a dictionary of pandas series.


```python
index_list = ['MIT', 'Stanford', 'Harvard', 'Caltech', 'Cambridge']
u = {
    'citations': pd.Series([99.9, 99.4, 99.9, 100, 78.4], index=index_list),
    'employer': pd.Series([100, 100, 100, 85.4, 100], index=index_list)
}

df = pd.DataFrame(u)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>citations</th>
      <th>employer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MIT</th>
      <td>99.9</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Stanford</th>
      <td>99.4</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Harvard</th>
      <td>99.9</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Caltech</th>
      <td>100.0</td>
      <td>85.4</td>
    </tr>
    <tr>
      <th>Cambridge</th>
      <td>78.4</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.index)
```




    Index(['MIT', 'Stanford', 'Harvard', 'Caltech', 'Cambridge'], dtype='object')




```python
print(df.columns)
```




    Index(['citations', 'employer'], dtype='object')



### Creating DataFrame from list of dictionaries


```python
l = [{'orange': 32, 'apple': 42}, {'banana': 25, 'carrot': 44, 'apple': 34}]
df = pd.DataFrame(l, index=['value1', 'value2'])
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apple</th>
      <th>banana</th>
      <th>carrot</th>
      <th>orange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>value1</th>
      <td>42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>value2</th>
      <td>34</td>
      <td>25.0</td>
      <td>44.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



You might have noticed that we got a DataFrame with `NaN` values in it. This is because we didn't the data for that particular row and column.

### Creating DataFrame from Text/CSV files

pandas tool come in handy when you want to load data from a csv or a text file. It has built in functions to do this for use.


```python
df = pd.read_csv('happiness.csv')
```

Yes we created a DataFrame from a csv file. This dataset contains outcome of European quality of life survey. This dataset is available [here](https://perso.telecom-paristech.fr/eagan/class/igr204/data/happiness.csv). Now we have stored the DataFrame in `df`, we want to see what's inside. First we will see the size of the DataFrame.


```python
print(df.shape)
```




    (105, 4)



It has 105 Rows and 4 Columns. Instead of printing out all the data, we will see the first 10 rows.


```python
df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Gender</th>
      <th>Mean</th>
      <th>N=</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AT</td>
      <td>Male</td>
      <td>7.3</td>
      <td>471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Female</td>
      <td>7.3</td>
      <td>570</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Both</td>
      <td>7.3</td>
      <td>1041</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BE</td>
      <td>Male</td>
      <td>7.8</td>
      <td>468</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Female</td>
      <td>7.8</td>
      <td>542</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>Both</td>
      <td>7.8</td>
      <td>1010</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BG</td>
      <td>Male</td>
      <td>5.8</td>
      <td>416</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>Female</td>
      <td>5.8</td>
      <td>555</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>Both</td>
      <td>5.8</td>
      <td>971</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CY</td>
      <td>Male</td>
      <td>7.8</td>
      <td>433</td>
    </tr>
  </tbody>
</table>
</div>



There are many more methods to create a DataFrames. But now we will see about basic operation on DataFrames.

### Operations on DataFrame

We'll recall the DataFrame we made earlier.


```python
index_list = ['MIT', 'Stanford', 'Harvard', 'Caltech', 'Cambridge']
u = {
    'citations': pd.Series([99.9, 99.4, 99.9, 100, 78.4], index=index_list),
    'employer': pd.Series([100, 100, 100, 85.4, 100], index=index_list)
}

df = pd.DataFrame(u)
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>citations</th>
      <th>employer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MIT</th>
      <td>99.9</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Stanford</th>
      <td>99.4</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Harvard</th>
      <td>99.9</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Caltech</th>
      <td>100.0</td>
      <td>85.4</td>
    </tr>
    <tr>
      <th>Cambridge</th>
      <td>78.4</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



Now we want to create a new row column from current columns. Let's see how it is done.


```python
df['score'] = (2 * df['citations'] + 3 * df['employer'])/5
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>citations</th>
      <th>employer</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MIT</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>99.96</td>
    </tr>
    <tr>
      <th>Stanford</th>
      <td>99.4</td>
      <td>100.0</td>
      <td>99.76</td>
    </tr>
    <tr>
      <th>Harvard</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>99.96</td>
    </tr>
    <tr>
      <th>Caltech</th>
      <td>100.0</td>
      <td>85.4</td>
      <td>91.24</td>
    </tr>
    <tr>
      <th>Cambridge</th>
      <td>78.4</td>
      <td>100.0</td>
      <td>91.36</td>
    </tr>
  </tbody>
</table>
</div>



We have created a new column `score` from `citations` and `employer`. We'll create one more using boolean.


```python
df['flag'] = df['citations'] > 99.5
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>citations</th>
      <th>employer</th>
      <th>score</th>
      <th>flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MIT</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>99.96</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Stanford</th>
      <td>99.4</td>
      <td>100.0</td>
      <td>99.76</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Harvard</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>99.96</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Caltech</th>
      <td>100.0</td>
      <td>85.4</td>
      <td>91.24</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Cambridge</th>
      <td>78.4</td>
      <td>100.0</td>
      <td>91.36</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



We can also remove columns.


```python
score = df.pop('score')
print(score)
```




    MIT          99.96
    Stanford     99.76
    Harvard      99.96
    Caltech      91.24
    Cambridge    91.36
    Name: score, dtype: float64




```python
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>citations</th>
      <th>employer</th>
      <th>flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MIT</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Stanford</th>
      <td>99.4</td>
      <td>100.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Harvard</th>
      <td>99.9</td>
      <td>100.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Caltech</th>
      <td>100.0</td>
      <td>85.4</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Cambridge</th>
      <td>78.4</td>
      <td>100.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Descriptive Statistics using pandas

It's very easy to view descriptive statistics of a dataset using pandas. We are gonna use, Biomass data collected from this [source](https://vincentarelbundock.github.io/Rdatasets/datasets.html). Let's load the data first.


```python
url = 'https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/DAAG/biomass.csv'
df = pd.read_csv(url)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>dbh</th>
      <th>wood</th>
      <th>bark</th>
      <th>root</th>
      <th>rootsk</th>
      <th>branch</th>
      <th>species</th>
      <th>fac26</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>90</td>
      <td>5528.0</td>
      <td>NaN</td>
      <td>460.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>E. maculata</td>
      <td>z</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>106</td>
      <td>13650.0</td>
      <td>NaN</td>
      <td>1500.0</td>
      <td>665.0</td>
      <td>NaN</td>
      <td>E. pilularis</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>112</td>
      <td>11200.0</td>
      <td>NaN</td>
      <td>1100.0</td>
      <td>680.0</td>
      <td>NaN</td>
      <td>E. pilularis</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>34</td>
      <td>1000.0</td>
      <td>NaN</td>
      <td>430.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>E. pilularis</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>130</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3000.0</td>
      <td>1030.0</td>
      <td>NaN</td>
      <td>E. maculata</td>
      <td>z</td>
    </tr>
  </tbody>
</table>
</div>



We are not interested in the unnamed column. So, let's delete that first. Then we'll see the statistics with one line of code.


```python
del df['Unnamed: 0']
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dbh</th>
      <th>wood</th>
      <th>bark</th>
      <th>root</th>
      <th>rootsk</th>
      <th>branch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>153.000000</td>
      <td>133.000000</td>
      <td>17.000000</td>
      <td>54.000000</td>
      <td>53.000000</td>
      <td>76.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>26.352941</td>
      <td>1569.045113</td>
      <td>513.235294</td>
      <td>334.383333</td>
      <td>113.802264</td>
      <td>54.065789</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.273679</td>
      <td>4071.380720</td>
      <td>632.467542</td>
      <td>654.641245</td>
      <td>247.224118</td>
      <td>65.606369</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>0.300000</td>
      <td>0.050000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.000000</td>
      <td>29.000000</td>
      <td>59.000000</td>
      <td>11.500000</td>
      <td>2.000000</td>
      <td>10.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15.000000</td>
      <td>162.000000</td>
      <td>328.000000</td>
      <td>41.000000</td>
      <td>11.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36.000000</td>
      <td>1000.000000</td>
      <td>667.000000</td>
      <td>235.000000</td>
      <td>45.000000</td>
      <td>77.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>145.000000</td>
      <td>25116.000000</td>
      <td>1808.000000</td>
      <td>3000.000000</td>
      <td>1030.000000</td>
      <td>371.000000</td>
    </tr>
  </tbody>
</table>
</div>



It's simple as that. We can see all the statistics. Count, mean, standard deviation and other statistics. Now we are gonna find some other metrics which are not available in the `describe()` summary.

### Mean


```python
print(df.mean())
```




    dbh         26.352941
    wood      1569.045113
    bark       513.235294
    root       334.383333
    rootsk     113.802264
    branch      54.065789
    dtype: float64



### Min and Max


```python
print(df.min())
```




    dbh                      3
    wood                     3
    bark                     7
    root                   0.3
    rootsk                0.05
    branch                   4
    species    Acacia mabellae
    dtype: object




```python
print(df.max())
```




    dbh          145
    wood       25116
    bark        1808
    root        3000
    rootsk      1030
    branch       371
    species    Other
    dtype: object



### Pairwise Correlation


```python
df.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dbh</th>
      <th>wood</th>
      <th>bark</th>
      <th>root</th>
      <th>rootsk</th>
      <th>branch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dbh</th>
      <td>1.000000</td>
      <td>0.905175</td>
      <td>0.965413</td>
      <td>0.899301</td>
      <td>0.934982</td>
      <td>0.861660</td>
    </tr>
    <tr>
      <th>wood</th>
      <td>0.905175</td>
      <td>1.000000</td>
      <td>0.971700</td>
      <td>0.988752</td>
      <td>0.967082</td>
      <td>0.821731</td>
    </tr>
    <tr>
      <th>bark</th>
      <td>0.965413</td>
      <td>0.971700</td>
      <td>1.000000</td>
      <td>0.961038</td>
      <td>0.971341</td>
      <td>0.943383</td>
    </tr>
    <tr>
      <th>root</th>
      <td>0.899301</td>
      <td>0.988752</td>
      <td>0.961038</td>
      <td>1.000000</td>
      <td>0.936935</td>
      <td>0.679760</td>
    </tr>
    <tr>
      <th>rootsk</th>
      <td>0.934982</td>
      <td>0.967082</td>
      <td>0.971341</td>
      <td>0.936935</td>
      <td>1.000000</td>
      <td>0.621550</td>
    </tr>
    <tr>
      <th>branch</th>
      <td>0.861660</td>
      <td>0.821731</td>
      <td>0.943383</td>
      <td>0.679760</td>
      <td>0.621550</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning

We need to clean our data. Our data might contain missing values, NaN values, outliers, etc. We may need to remove or replace that data. Otherwise our data might make any sense.

We can find null values using following method.


```python
print(df.isnull().any())
```




    dbh        False
    wood        True
    bark        True
    root        True
    rootsk      True
    branch      True
    species    False
    fac26       True
    dtype: bool



We have to remove these null values. This can done by method shown below.


```python
newdf = df.dropna()

print(newdf.shape)
```




    (4, 8)



But, sadly our datset reduced to a small one. But, you get the point.

There are many more useful tools in pandas. We'll see more about them in upcoming posts.

Discuss this post on [Hacker News](https://news.ycombinator.com/item?id=15331975)

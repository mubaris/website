---
layout: post
title:  "Introduction to Data Visualization using Python"
author: "Mubaris NK"
---

Data visualization is one of primary skills of any data scientist. It's also a large field in itself. There are many courses available just focused on Data Visualization. This post is just an introduction to this much broader topic.

In this post first we will look at data visualization **conceptually**, then we will explore more using Python libraries.

## What is Data Visualization?

> **By visualizing information, we turn it into a landscape that you can explore with your eyes, a sort of information map. And when you’re lost in information, an information map is kind of useful.** ―David McCandless

**Data visualizations is the process of turning large and small datasets into visuals that are easier for the human brain to understand and process.**

When we have a dataset, it will take some time to make the meaning of that data. But, when we represent this data in graphs or other visualizations, it is much more easier for us to understand. That's the power of data visualization.

### Examples of Data Visualizations

* **Countries with largest defense budget**

![Defense Budget](https://i.imgur.com/SJwm2Kg.png)

You can clearly see that US defense budget almost equal as the combined budget of other countries.

* **Largest Occupations in the United States**

![US Occupation](https://i.imgur.com/BSmMBo3.png)

* **Atheists in Europe**

![Europe Atheist](https://i.redd.it/shyvnpt2kamx.png)

* **Death by Heart Decease in US by Nick Usoff**

![Heart Decease](https://i.redd.it/0kjvfx55vody.png)

I can show you many more here. There are endless supply of Data Visualizations available on internet.


### Principles of Good Data Visualization

These principles are directly taken from [Data Visualisation: A Handbook for Data Driven Design by Andy Kirk](https://www.amazon.com/Data-Visualisation-Handbook-Driven-Design/dp/1473912148). I highly recommend reading this book.

* **Trustworthy**

This means that the data presented is honestly portrayed, or the visualization is not misleading. Trust is hard to earn and easy to lose. This is very important.

* **Accessible**

Accessible is about focusing on your target audience and ability to use your visualization.

* **Elegant**

It's important to have stylish and beautiful visualization when you present them. If you are exploring data, it might not be critical. But, if you presenting your visualization to a particular audience or submitting on some platform, you will need beautiful visualizations.

## Data Visualization in Python using Matplotlib

[Matplotlib](https://matplotlib.org/) is a widely used visualization package in Python. It's very easy to create and present data visualizations using Matplotlib. There are other visualization libraries available in Python.

* [Seaborn](https://seaborn.pydata.org/)
* [ggplot](http://ggplot.yhathq.com/)
* [Altair](https://altair-viz.github.io/)
* [Bokeh](https://bokeh.pydata.org/en/latest/)
* [pygal](http://pygal.org/en/stable/)
* [Plotly](https://plot.ly/)
* [geoplotlib](https://github.com/andrea-cuttone/geoplotlib)
* and many more

We are going to learn how to create Bar plots, Line plots and Histograms using Matplotlib in this post. The entire code created is using [Jupyter Notebooks](https://jupyter.org/).

### Line Plots

Line plots are very simple plots. It represents frequency of data along a number lines. You can learn more about Line charts and Spline charts from Data Viz Project.

* [Line Chart](http://datavizproject.com/data-type/line-chart/)
* [Spline Chart](http://datavizproject.com/data-type/spline-graph/)

We'll use [Bitcoin Historical Price Dataset](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory) from Kaggle to draw line plots here.

First we'll import all **numpy**, **pandas** and **matplotlib**. Then we read the data using `read_csv` function from pandas.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bitcoin_dataset.csv')
data.head()
```

You will get an output table with 24 columns and 5 rows(Too long to print here).


```python
data.shape
```




    (1590, 24)



We'll need to convert the Date string to pandas datetime.


```python
data['Date'] = pd.to_datetime(data['Date'].values)
```

Now we extract date and price from our data set.


```python
date = data['Date'].values
price = data['btc_market_price'].values
```

Now we can plot using these values.


```python
plt.plot(date, price)
plt.show()
```


![png](https://mubaris.com/files/images/output_12_0.png)


This plot is not labelled. And the axes are not perfect. We'll fix that now.


```python
plt.plot(date, price, c='magenta')

# Add title
plt.title("BTC Price over time")

# Axis labels
plt.xlabel("Year")
plt.ylabel("Price in USD")

# Axes Range
plt.axis(['2009', '2018', 0, 5000])

plt.show()
```


![png](https://mubaris.com/files/images/output_14_0.png)


### Bar Plots

Bar Plot is chart that represents categorical data with rectangular bars. More about bar plots at [Data Viz Project](http://datavizproject.com/data-type/bar-chart/)

We'll use European Developers Salary data to plot bar graph. Get this data from [here](https://mubaris.com/files/salary.csv)

At first we read the data from csv file.


```python
salary = pd.read_csv('salary.csv')
salary.columns = ['Experience', 'Salary', 'Country']
salary.head()
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
      <th>Experience</th>
      <th>Salary</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>27930</td>
      <td>Austria</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.0</td>
      <td>28000</td>
      <td>Austria</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>39200</td>
      <td>Austria</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>39200</td>
      <td>Austria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>40000</td>
      <td>Austria</td>
    </tr>
  </tbody>
</table>
</div>



We will be plotting mean salary by each country. So we'll get mean value by each country.


```python
salary = salary.groupby(['Country']).mean()
salary
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
      <th>Experience</th>
      <th>Salary</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austria</th>
      <td>7.980000</td>
      <td>53385.200000</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>6.952381</td>
      <td>55803.047619</td>
    </tr>
    <tr>
      <th>Bulgaria</th>
      <td>10.264706</td>
      <td>42017.647059</td>
    </tr>
    <tr>
      <th>Croatia</th>
      <td>6.600000</td>
      <td>30275.900000</td>
    </tr>
    <tr>
      <th>Cyprus</th>
      <td>3.000000</td>
      <td>26093.333333</td>
    </tr>
    <tr>
      <th>Czech Republic</th>
      <td>8.562500</td>
      <td>46110.750000</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>9.562500</td>
      <td>83223.666667</td>
    </tr>
    <tr>
      <th>Estonia</th>
      <td>7.153846</td>
      <td>37526.153846</td>
    </tr>
    <tr>
      <th>Finland</th>
      <td>6.117647</td>
      <td>45642.647059</td>
    </tr>
    <tr>
      <th>France</th>
      <td>5.507843</td>
      <td>49085.176471</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>6.607735</td>
      <td>66540.110497</td>
    </tr>
    <tr>
      <th>Greece</th>
      <td>8.769231</td>
      <td>31716.153846</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>7.722222</td>
      <td>26873.666667</td>
    </tr>
    <tr>
      <th>Ireland</th>
      <td>7.414894</td>
      <td>62754.510638</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>7.526923</td>
      <td>34007.692308</td>
    </tr>
    <tr>
      <th>Latvia</th>
      <td>5.333333</td>
      <td>32666.666667</td>
    </tr>
    <tr>
      <th>Lithuania</th>
      <td>7.200000</td>
      <td>34333.333333</td>
    </tr>
    <tr>
      <th>Luxembourg</th>
      <td>12.750000</td>
      <td>61250.000000</td>
    </tr>
    <tr>
      <th>Malta</th>
      <td>7.400000</td>
      <td>48400.000000</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>6.890000</td>
      <td>54096.537500</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>8.342105</td>
      <td>107457.421053</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>6.322222</td>
      <td>36655.111111</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>5.300000</td>
      <td>30148.500000</td>
    </tr>
    <tr>
      <th>Romania</th>
      <td>6.183333</td>
      <td>35043.133333</td>
    </tr>
    <tr>
      <th>Serbia</th>
      <td>7.375000</td>
      <td>33450.000000</td>
    </tr>
    <tr>
      <th>Slovakia</th>
      <td>4.400000</td>
      <td>24618.000000</td>
    </tr>
    <tr>
      <th>Slovenia</th>
      <td>8.000000</td>
      <td>37380.000000</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>7.070755</td>
      <td>38556.452830</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>6.792453</td>
      <td>77481.000000</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>7.250000</td>
      <td>93962.250000</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>6.080214</td>
      <td>68270.550802</td>
    </tr>
  </tbody>
</table>
</div>



Now we extract these values to plot. We are only taking first 5 countries.


```python
country = salary.index[:5]
country_array = np.arange(5)
mean_salary = salary['Salary'].values[:5]
```


```python
# Basic Plot
plt.bar(country_array, mean_salary, color='#f44c44')

# X-Axis Tick Labels
plt.xticks(country_array, country)

# Title
plt.title("European Developers Salary")

# Y-Axis Label
plt.ylabel("Salary in €")

plt.show()
```


![png](https://mubaris.com/files/images/output_21_0.png)


We can clearly see that Belgium has the highest average salary and Cyprus has least average salary among these five countries.

### Histogram

Histogram a diagram consisting of rectangles whose area is proportional to the frequency of a variable and whose width is equal to the class interval. More about [Histograms](http://datavizproject.com/data-type/histogram/)

We are going to generate some random numbers using **numpy**. Then we will plot histogram of these random numbers.


```python
data = np.random.randint(low=-100, high=250, size=400)

#Plot. bins=no. of bins
plt.hist(data, bins=15, color='#988659')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

#Using Grids
plt.grid()

plt.show()
```


![png](https://mubaris.com/files/images/output_24_0.png)


This is not showing any kind of special data. We can generate **Gaussian(Normal)** random numbers using numpy to create better histograms. These are just random numbers, this doesn't represent any data.


```python
# Mean = 5, Standard Deviation = 2, Number of points = 1000
data = np.random.normal(5, 2, 1000)

#Plot. bins=no. of bins
plt.hist(data, bins=10, color='#8cdcb4')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

#Using Grids
plt.grid()

plt.show()
```


![png](https://mubaris.com/files/images/output_26_0.png)


## Conclusion

So far we have learned how to create **Line plots, Bar plots and Histograms** using **Matplotlib** library. In the future posts we will learn more about how to create more plots. Also, we will use data science methods for a particular case study.

Discuss this post on [Hacker News](https://news.ycombinator.com/item?id=15337132)
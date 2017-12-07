---
layout: post
title:  "DataViz Mastery Part 1 - Treemaps"
author: "Mubaris NK"
comments: true
tags: python dataviz tutorial
twimg: https://i.imgur.com/SJwm2Kg.png
image: https://i.imgur.com/SJwm2Kg.png
---

DataViz Mastery will be a series blog posts which aims to master data visualizations using Python. I am aiming to cover all visualizations in [DataViz Project](http://datavizproject.com/). In this part 1 of the series we will cover how to create [Treemaps](http://datavizproject.com/data-type/treemap/) with Python.

## Treemap

Treemaps display hierarchical data as a set of nested rectangles. Each group is represented by a rectangle, which area is proportional to its value. Using color schemes, it is possible to represent several dimensions: groups, subgroups… Treemaps have the advantage to make efficient use of space, what makes them useful to represent a big amount of data.

### Examples

![Bloomberg Force Awakens](https://i.imgur.com/6qYatVp.png)

![Defense Budget](https://i.imgur.com/SJwm2Kg.png)

![Third Example](https://i.imgur.com/VSpdfU8.png)

## The Code

We will use the data of Star Wars Movie Franchise Revenue from [Statistic Brain](http://www.statisticbrain.com/star-wars-total-franchise-revenue/). `squarify` is a Python module helps you plot Treemaps with Matplotlib backend. Seaborn is another data visualization library with Matplotlib backend. Seaborn helps you create beautiful visualizations. You can interact with Seaborn in 2 ways.

1) Activate and Seaborn and Use Matplotlib

2) Use Seaborn API

Since, Seaborn doesn't have Treemaps API, we will use 1st option.

If you are unfamiliar with Matplotlib, read this [Introductory Post](https://mubaris.com/2017-09-26/introduction-to-data-visualizations-using-python)


```python
# Data Manipulation
import pandas as pd
# Treemap Ploting
import squarify
# Matplotlib and Seaborn imports
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
# Activate Seaborn
sns.set()
%matplotlib inline
# Large Plot
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)
# Use ggplot style
style.use('ggplot')
```

We have imported necessary modules to generate Treemap. Now let's import out dataset.


```python
# Reading CSV file
df = pd.read_csv("starwars-revenue.csv")
# Sort by Revenue
df = df.sort_values(by="Revenue", ascending=False)
# Find Percentage
df["Percentage"] = round(100 * df["Revenue"] / sum(df["Revenue"]), 2)
# Create Treemap Labels
df["Label"] = df["Label"] + " (" + df["Percentage"].astype("str") + "%)"
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
      <th>Movie</th>
      <th>Revenue</th>
      <th>Label</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Episode 7 – The Force Awakens</td>
      <td>4068223624</td>
      <td>The Force Awakens (37.68%)</td>
      <td>37.68</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rogue One</td>
      <td>2450000000</td>
      <td>Rogue One (22.69%)</td>
      <td>22.69</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Episode 1 – The Phantom Menace</td>
      <td>924317558</td>
      <td>The Phantom Menace (8.56%)</td>
      <td>8.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Episode 3 – Revenge of the Sith</td>
      <td>848754768</td>
      <td>Revenge of the Sith (7.86%)</td>
      <td>7.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Episode 4 – A New Hope</td>
      <td>775398007</td>
      <td>A New Hope (7.18%)</td>
      <td>7.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Episode 2 – Attack of the Clones</td>
      <td>649398328</td>
      <td>Attack of the Clones (6.01%)</td>
      <td>6.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Episode 5 – Empire Strikes Back</td>
      <td>538375067</td>
      <td>Empire Strikes Back (4.99%)</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Episode 6 – Return of the Jedi</td>
      <td>475106177</td>
      <td>Return of the Jedi (4.4%)</td>
      <td>4.40</td>
    </tr>
    <tr>
      <th>8</th>
      <td>The Clone Wars</td>
      <td>68282844</td>
      <td>The Clone Wars (0.63%)</td>
      <td>0.63</td>
    </tr>
  </tbody>
</table>
</div>



That's out dataframe. Now Let's Plot it.


```python
# Get Axis and Figure
fig, ax = plt.subplots()
# Our Colormap
cmap = matplotlib.cm.coolwarm
# Min and Max Values
mini = min(df["Revenue"])
maxi = max(df["Revenue"])
# Finding Colors for each tile
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in df["Revenue"]]
# Plotting
squarify.plot(sizes=df["Revenue"], label=df["Label"], alpha=0.8, color=colors)
# Removing Axis
plt.axis('off')
# Invert Y-Axis
plt.gca().invert_yaxis()
# Title
plt.title("Revenue from Star Wars Franchise Movies", fontsize=32)
# Title Positioning
ttl = ax.title
ttl.set_position([.5, 1.05])
# BG Color
fig.set_facecolor('#eeffee')
```


![png](https://mubaris.com/files/images/output_5_0_1.png)


If you want to try different colormap, find a colormap of your choice from [Matplotlib Docs](https://matplotlib.org/examples/color/colormaps_reference.html) and replace 2nd line in this snippet. Now Let's try plotting World's top 10 Billionaires net worth.


```python
# Reading CSV file
df = pd.read_csv("rich.csv")
# Label
df["Label"] = df["Name"] + " - $" + df["Net Worth in Billion $"].astype("str") + "B"
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
      <th>Name</th>
      <th>Net Worth in Billion $</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bill Gates</td>
      <td>86.0</td>
      <td>Bill Gates - $86.0B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Warren Buffett</td>
      <td>75.6</td>
      <td>Warren Buffett - $75.6B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jeff Bezos</td>
      <td>72.8</td>
      <td>Jeff Bezos - $72.8B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Amancio Ortega</td>
      <td>71.3</td>
      <td>Amancio Ortega - $71.3B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mark Zuckerberg</td>
      <td>56.0</td>
      <td>Mark Zuckerberg - $56.0B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Carlos Slim Helu</td>
      <td>54.5</td>
      <td>Carlos Slim Helu - $54.5B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Larry Ellison</td>
      <td>52.2</td>
      <td>Larry Ellison - $52.2B</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Charles Koch</td>
      <td>48.3</td>
      <td>Charles Koch - $48.3B</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Davis Koch</td>
      <td>48.3</td>
      <td>Davis Koch - $48.3B</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Michael Bloomberg</td>
      <td>47.5</td>
      <td>Michael Bloomberg - $47.5B</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change Style
style.use('fivethirtyeight')
fig, ax = plt.subplots()
# Manually Entering Colors
colors = ["#248af1", "#eb5d50", "#8bc4f6", "#8c5c94", "#a170e8", "#fba521", "#75bc3f"]
# Plot
squarify.plot(sizes=df["Net Worth in Billion $"], label=df["Label"], alpha=0.9, color=colors)
plt.axis('off')
plt.gca().invert_yaxis()
plt.title("Net Worth of World's Top 10 Billionaires", fontsize=32, color="Black")
ttl = ax.title
ttl.set_position([.5, 1.05])
fig.set_facecolor('#effeef')
```


![png](https://mubaris.com/files/images/output_8_0_1.png)


That concludes the part 1 of DataViz Mastery. Let me know if you have any questions. In the next DataViz Mastery post we will learn how to create Word Clouds using Python

Checkout this [Github Repo](https://github.com/mubaris/dataviz-gallery/) for more visualizations.

## Data Visualization Books

1) <a href="http://amzn.to/2iww3ab" target="_blank">Storytelling with Data: A Data Visualization Guide for Business Professionals</a>

2) <a href="http://amzn.to/2zf20vp" target="_blank">The Truthful Art: Data, Charts, and Maps for Communication</a>

3) <a href="http://amzn.to/2ixm7gC" target="_blank">Data Visualization: a successful design process</a>

4) <a href="http://amzn.to/2herZ1K" target="_blank">Data Visualisation: A Handbook for Data Driven Design</a>

<div id="mc_embed_signup">
<form action="//mubaris.us16.list-manage.com/subscribe/post?u=f9e9a4985cce81e89169df2bf&amp;id=3654da5463" method="post" id="mc-embedded-subscribe-form" name="mc-embedded-subscribe-form" class="validate" target="_blank" novalidate>
    <div id="mc_embed_signup_scroll">
    <label for="mce-EMAIL">Subscribe for more Awesome!</label>
    <input type="email" value="" name="EMAIL" class="email" id="mce-EMAIL" placeholder="email address" required>
    <!-- real people should not fill this in and expect good things - do not remove this or risk form bot signups-->
    <div style="position: absolute; left: -5000px;" aria-hidden="true"><input type="text" name="b_f9e9a4985cce81e89169df2bf_3654da5463" tabindex="-1" value=""></div>
    <div class="clear"><input type="submit" value="Subscribe" name="subscribe" id="mc-embedded-subscribe" class="button"></div>
    </div>
</form>
</div>

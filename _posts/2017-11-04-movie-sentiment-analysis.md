---
layout: post
title:  "Analyzing Movie Subtitles"
author: "Mubaris NK"
comments: true
catalog: true
tags: python data-science nlp
header-img: files/images/post12.jpg
twimg: https://i.imgur.com/etpqMur.png
image: https://i.imgur.com/etpqMur.png
---

In this post we will analyze a movie using sentiment analysis of its subtitles. This is little bit tricky. Because, the subtitles are time dependant data. Unlike other time series data, subtitles don't have constant time intervals. We will need to change them to equal time intervals. We will use various Python modules to analyze the subtitles. Let's start with importing the subtitle.

## `pysrt` - Python Module for Handling srt Subtitles

[pysrt](https://github.com/byroot/pysrt) is a great Python module to handle srt files. Let's see `pysrt` in action.

```python
import pysrt

# Loading the Subtitle
subs = pysrt.open('some/file.srt')

sub = subs[0]

# Subtitle text
text = sub.text
text_without_tags = sub.text_without_tags

# Start and End time
start = sub.start.to_time()
end = sub.end.to_time()

# Removing line and saving
del subs[index]
subs.save('some/file.srt')
```

## `TextBlob` - Simplified Python Module for Text Processing

[TextBlob](https://textblob.readthedocs.io/en/dev/) is a Python library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more. Let's see TextBlob in action.

```python
from textblob import TextBlob

text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
blob.tags           # [('The', 'DT'), ('titular', 'JJ'),
                    #  ('threat', 'NN'), ('of', 'IN'), ...]

blob.noun_phrases   # WordList(['titular threat', 'blob',
                    #            'ultimate movie monster',
                    #            'amoeba-like mass', ...])

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
# 0.060
# -0.341

blob.translate(to="es")  # 'La amenaza titular de The Blob...'

# Sentiment Analysis
blob = TextBlob(text)
sentiment_polarity = blob.sentiment.polarity # -0.1590909090909091
sentiment_subjectivity = blob.sentiment.subjectivity # 0.6931818181818182
```

## The Method

We can divide the method in to 4 steps.

* Divide total running time to constant time intervals
* Collect and combine all the text in each time interval
* Find the sentiment polarity of text in each time interval
* Visualize our analysis


```python
from datetime import date, datetime, timedelta, time
import pysrt
from textblob import TextBlob
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (16.0, 9.0)
style.use('fivethirtyeight')
```


```python
# Helper Function to create equally divided time intervals
# start - Starting Time
# end - Ending Time
# delta - Interval Period
def create_intervals(start, end, delta):
    curr = start
    while curr <= end:
        curr = (datetime.combine(date.today(), curr) + delta).time()
        yield curr

# Main Function to Get Sentiment Data
# file - srt file location
# delta - time interval in minutes
def get_sentiment(file, delta=2):
    # Reading Subtitle
    subs = pysrt.open(file, encoding='iso-8859-1')
    n = len(subs)
    # List to store the time periods
    intervals = []
    # Start, End and Delta
    start = time(0, 0, 0)
    end = subs[-1].end.to_time()
    delta = timedelta(minutes=delta)
    for result in create_intervals(start, end, delta):
        intervals.append(result)
    # List to store sentiment polarity
    sentiments = []
    
    index = 0
    m = len(intervals)
    # Collect and combine all the text in each time interval
    for i in range(m):
        text = ""
        for j in range(index, n):
            # Finding all subtitle text in the each time interval
            if subs[j].end.to_time() < intervals[i]:
                text += subs[j].text_without_tags + " "
            else:
                break
        # Sentiment Analysis
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        sentiments.append(pol)
        index = j
    # Adding Initial State
    intervals.insert(0, time(0, 0, 0))
    sentiments.insert(0, 0.0)
    return (intervals, sentiments)

# Utility to find average sentiment
def average(y):
    avg = float(sum(y))/len(y)
    return avg
```

We have written our function to find sentiment of subtitle over interval of time. Now let's try to plot this. We'll use "Thor" movie subtitles here.


```python
x, y = get_sentiment("Thor.srt")
fig, ax = plt.subplots()
plt.plot(x, y)
plt.title("Thor (2011)", fontsize=32)
plt.ylim((-1, 1))
plt.ylabel("Sentiment Polarity")
plt.xlabel("Running Time")
plt.text(.5, 1.03, "Average Sentiment - " + str(round(average(y), 4)), color="green")
ttl = ax.title
ttl.set_position([.5, 1.05])
```


![png](https://mubaris.com/files/images/output_4_0.png)


There we have our plot of Sentiment Analysis of subtitle against the running time. All code is available on [GitHub](https://github.com/mubaris/dataviz-gallery/tree/master/movie-subtitles)

## More Examples

#### Iron Man (2008)

![Iron Man](https://i.imgur.com/zFUGY0v.png)

#### The Incredible Hulk (2008)

![The Incredible Hulk](https://i.imgur.com/FGATJPH.png)

#### Iron Man 2 (2010)

![Iron Man 2](https://i.imgur.com/2hOSCf0.png)

#### Thor (2011)

![Thor](https://i.imgur.com/etpqMur.png)

#### Captain America: The First Avenger (2011)

![Captain America: The First Avenger](https://i.imgur.com/dXtIEov.png)

#### The Avengers (2012)

![The Avengers](https://i.imgur.com/e7vRVuo.png)

#### Iron Man 3 (2013)

![Iron Man 3](https://i.imgur.com/hzhy5GQ.png)

#### Captain America: The Winter Soldier (2014)

![Captain America: The Winter Soldier](https://i.imgur.com/2OrU8IE.png)

#### Guardians of the Galaxy (2014)

![Guardians of the Galaxy](https://i.imgur.com/yyyyR1E.png)

#### Avengers: Age of Ultron (2015)

![Avengers: Age of Ultron ](https://i.imgur.com/8ib9nEu.png)

#### Ant-Man (2015)

![Ant-Man](https://i.imgur.com/2dRuMe4.png)

#### Captain America: Civil War (2016)

![Captain America: Civil War](https://i.imgur.com/zASrxVo.png)

#### Doctor Strange (2016)

![Doctor Strange](https://i.imgur.com/WSN4it7.png)

#### Guardians of the Galaxy Vol. 2 (2017)

![Guardians of the Galaxy Vol. 2](https://i.imgur.com/UT8NvMl.png)

#### Spider-Man: Homecoming (2017)

![Spider-Man: Homecoming](https://i.imgur.com/7T357iY.png)

#### Man of Steel (2013)

![Man of Steel](https://i.imgur.com/wXWlxtH.png)

#### Batman v Superman: Dawn of Justice (2016)

![Batman v Superman: Dawn of Justice](https://i.imgur.com/8ySq3bs.png)

#### Suicide Squad (2016)

![Suicide Squad](https://i.imgur.com/tOB9eQw.png)

#### Wonder Woman (2017)

![Wonder Woman](https://i.imgur.com/wxlU4Pv.png)

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
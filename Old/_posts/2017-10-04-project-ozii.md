---
layout: post
title:  "Project Ozii - Generating PulseGraphs from Text"
author: "Mubaris NK"
comments: true
tags: python project
---

Ever since I watched the movie **'Arrival'**, I wanted to create my own logographs. Just like the one they show in the movie. So, I started searching about the symbols they shown in the movie. I found that the movie company hired Wolfram to produce the logographs. Later, they did a live coding broadcast related to this. You can watch the video [here](https://www.youtube.com/watch?v=r8nTifCIr0c). 38 of these logographs and related documents are available in this [Github Repo](https://github.com/WolframResearch/Arrival-Movie-Live-Coding). These logographs look like this:

![Time Logograph](https://i.imgur.com/jxnQVu3.jpg)

This image means 'Time'.

Then I thought about creating my own set of logographs. I started with creating some set of rules and procedure.

1) Only English letters will be considered.

2) Assigned weights to each English alphabet according to their freqency ranging from 0 to 10.

3) Applied standard mathematical function to each alphabet in the letter.

4) Plot the final function.

The frequency values were like this:

```python
    'e': 10.0,
    't': 9.62,
    'a': 9.23,
    'o': 8.85,
    'i': 8.46,
    'n': 8.08,
    's': 7.69,
    'r': 7.31,
    'h': 6.92,
    'd': 6.54,
    'l': 6.15,
    'u': 5.77,
    'c': 5.34,
    'm': 5.00,
    'f': 4.62,
    'y': 4.23,
    'w': 3.85,
    'g': 3.46,
    'p': 3.08,
    'b': 2.69,
    'v': 2.31,
    'k': 1.92,
    'x': 1.54,
    'q': 1.15,
    'j': 0.77,
    'z': 0.34
```

I tried functions \\(sin(x)\\), \\(cos(x)\\), \\(log(x)\\), \\(e^x\\), and \\(x^n\\) with different values of n. Tried summing up, multiplying and composite of functions. But nothing worked. Everything lead to unstable and weird plots.

When I tried composite function for some word, I got an interesting plot like this one below:

![Demo1](https://i.imgur.com/0opRmuw.png)

It looks like pulses. When I checked for the function, it was the following one:

\\[\frac{1}{sin(cos(x))}\\]

But of course, it had different coeffiecients. Then I checked for the actual plot of this function. That looked like this.

![Transformer](https://i.imgur.com/CPsaVDf.png)

It looks very interesting. So, I created a method to generate pulsegraphs from text using the frequency of the alphabet and this function. For now on in this we call this function \\(ozii\\).

\\[ozii(x) = \frac{1}{sin(cos(x))}\\]

Here \\(x\\) is in degrees.

## The Method

\\[ x \in (0, 1] \\]

\\[ p = \sum_{i=1}^n weight(a_i) * x^i \\]

\\[ y = ozii(p) \\]

\\(a_i\\) is the i<sup>th</sup> alphabet and p is a polynomial generated from weight values defined the above table. For each unique word we will get unique polynomial.

I scaled down \\(y\\) value to maximum of 0.5, Then plot a graph between x and y. There you have your pulse graph.

For example, lets take the input as 'time':

\\[ p = 9.62x + 8.46x^2 + 5x^3 + 10x^4 \\]

\\[ y = ozii(p) \\]

Scale down \\(y\\) to maximum of 0.5

If you plot this you will get the following pulsegraph:

![time pulsegraph](https://i.imgur.com/SC21sZw.png)

And for sentences I took the sum of each \\(y\\) to make the plot.

## Full Code

```python
import os
import numpy as np
import matplotlib.pyplot as plt

# Weight Values
alphabet = {
    'e': 10.0,
    't': 9.62,
    'a': 9.23,
    'o': 8.85,
    'i': 8.46,
    'n': 8.08,
    's': 7.69,
    'r': 7.31,
    'h': 6.92,
    'd': 6.54,
    'l': 6.15,
    'u': 5.77,
    'c': 5.34,
    'm': 5.00,
    'f': 4.62,
    'y': 4.23,
    'w': 3.85,
    'g': 3.46,
    'p': 3.08,
    'b': 2.69,
    'v': 2.31,
    'k': 1.92,
    'x': 1.54,
    'q': 1.15,
    'j': 0.77,
    'z': 0.34,
    'E': 10.0+(1e-7),
    'T': 9.62+(1e-7),
    'A': 9.23+(1e-7),
    'O': 8.85+(1e-7),
    'I': 8.46+(1e-7),
    'N': 8.08+(1e-7),
    'S': 7.69+(1e-7),
    'R': 7.31+(1e-7),
    'H': 6.92+(1e-7),
    'D': 6.54+(1e-7),
    'L': 6.15+(1e-7),
    'U': 5.77+(1e-7),
    'C': 5.34+(1e-7),
    'M': 5.00+(1e-7),
    'F': 4.62+(1e-7),
    'Y': 4.23+(1e-7),
    'W': 3.85+(1e-7),
    'G': 3.46+(1e-7),
    'P': 3.08+(1e-7),
    'B': 2.69+(1e-7),
    'V': 2.31+(1e-7),
    'K': 1.92+(1e-7),
    'X': 1.54+(1e-7),
    'Q': 1.15+(1e-7),
    'J': 0.77+(1e-7),
    'Z': 0.34+(1e-7),
    '.': 4.9e-7,
    '?': 5.1e-7,
    ' ': 0
}

def cos(x):
    return np.cos(180 * x / np.pi)

def sin(x):
    return np.sin(180 * x / np.pi)

def inverse(x):
    return 1/x

# Ozii Function
def transformer(x):
    y = cos(x)
    y = sin(y)
    y = inverse(y)
    return y

# X
x = np.linspace(0, 1, 1001)
x = x[1:]

# Return y for a single word
def transform(text):
    n = len(text)
    y = 0
    for i in range(len(text)):
        y += alphabet[text[i]] * (x ** (i+1))
    y = transformer(y)
    max_y = np.max(np.abs(y))
    y = (0.5/max_y) * y
    return y

# y for a sentence
def sentence_transformer(sentence):
    words = sentence.split()
    y = np.zeros(x.shape)
    for i, word in enumerate(words):
        y += transform(word)
    max_y = np.max(np.abs(y))
    y = (0.5/max_y) * y
    return y

# Create plot and generate image.
def generate_image(sentence, pixels=500, dir="output"):
    y = sentence_transformer(sentence)
    size = pixels / 10
    fig = plt.figure(figsize=(10, 10))
    plt.plot(x, y, linewidth=1, c='k')
    plt.axis([0, 1, -0.5, 0.5])
    plt.axis('off')
    words = sentence.split()
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filename = dir + "/" + sentence + ".png"
    plt.savefig(filename, dpi=size)
    plt.close('all')
    return filename
```

You might have noticed that I have added weight values to capital letters too.

## More Examples

#### ozii

![ozii](https://i.imgur.com/OdY8DbM.png)

#### Time

![Time](https://i.imgur.com/7rFdrBq.png)

#### There is no linear time

![There is no linear time](https://i.imgur.com/kgP2lWK.png)

#### Batman

![Batman](https://i.imgur.com/Ypj5K3Q.png)

#### I am Batman

![I am Batman](https://i.imgur.com/Y6Lfuw5.png)

#### Human

![Human](https://i.imgur.com/fnob73l.png)

#### Humanity

![Humanity](https://i.imgur.com/KqCpsPi.png)

Checkout this [Github Repo](https://github.com/mubaris/ozii) for examples and full code.

Let me know what you think.

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

---
layout: post
title:  "What is Machine Learning?"
author: "Mubaris NK"
---

*Learning*, like intelligence, covers such a broad range of processes that it is difficult to define precisely. According to Wikipedia, Learning is the act of acquiring new or modifying and reinforcing existing knowledge, behaviors, skills, values, or preferences which may lead to a potential change in synthesizing information, depth of the knowledge, attitude or behavior relative to the type and range of experiences.

In *Machine Learning*, computers apply **statistical learning** techniques to automatically identify patterns in data. These techniques can be used to make highly accurate predictions. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. **The primary aim is to allow the computers learn automatically** without human intervention or assistance and adjust actions accordingly.

## Types of Learning

There are four types of learning methods in Machine Learning.

* **Supervised Learning**: This method can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.

    Ex: Face Detection from Images, Spam Filter

* **Unsupervised Learning**: This is used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data.

    Ex: Social Network Analysis

* **Semi-Supervised Learning**: This falls somewhere in between supervised and unsupervised learning, since they use both labeled and unlabeled data for training – typically a small amount of labeled data and a large amount of unlabeled data. The systems that use this method are able to considerably improve learning accuracy. Usually, semi-supervised learning is chosen when the acquired labeled data requires skilled and relevant resources in order to train it / learn from it. Otherwise, acquiringunlabeled data generally doesn’t require additional resources.

* **Reinforcement Learning**: This is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.

    Ex: Game Bots

## Classification, Regression and Clustering

### Classification

The main goal of classification is to predict the target class (Yes/ No). If the trained  model is for predicting any of two target classes. It is known as binary classification. Considering the student profile to predict whether the student will pass or fail. Considering the customer, transaction details to predict whether he will buy the new product or not. These kind problems will be addressed with binary classification. If we have to predict more the two target classes it is known as multi-classification. Considering all subject details of a student to  predict which subject the student will score more. Identifying the object in an image. These kind problems are known as multi-classification problems.

![Classification](https://i.imgur.com/MZVSnu3.png)

### Regression

The main goal of regression algorithms is the predict the discrete or a continues value. In some cases, the predicted value can be used to identify the linear relationship between the attributes. Suppose the increase in the product advantage budget will increase the product sales.  Based on the problem difference regression algorithms can be used. some of the basic regression algorithms are linear regression, polynomial regression … etc

![Regression](https://i.imgur.com/Qu5WuSV.png)

### Clustering

Clustering is an unsupervised machine learning task that automatically divides the data into clusters, or groups of similar items. It does this without having been told how the groups should look ahead of time. As we may not even know what we're looking for, clustering is used for knowledge discovery rather than prediction. It provides an insight into the natural groupings found within data.

![Clustering](https://i.imgur.com/WLtE9F5.png)

## Books for Beginners

### <a target="_blank" href="http://amzn.to/2ysyqRL">Programming Collective Intelligence: Building Smart Web 2.0 Applications</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=morningdata-20&l=am2&o=1&a=0596529325" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

Programming Collective Intelligence, PCI as it is popularly known, is one of the best books to start learning machine learning. This book uses Python.

### <a target="_blank" href="http://amzn.to/2y8Talt">Machine Learning for Hackers: Case Studies and Algorithms to Get You Started</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=morningdata-20&l=am2&o=1&a=1449303714" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

This is a great introduction to Machine Learning using R.

### <a target="_blank" href="http://amzn.to/2xFfDot">Machine Learning in Action</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=morningdata-20&l=am2&o=1&a=1617290181" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

Machine Learning in Action is unique book that blends the foundational theories of machine learning with the practical realities of building tools for everyday data analysis using Python.

## Conclusion

![ML Classes](https://i.imgur.com/emDnmdY.png)

I highly recommend reading [Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/).

Discuss this post on [Hacker News](https://news.ycombinator.com/item?id=15346778)


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
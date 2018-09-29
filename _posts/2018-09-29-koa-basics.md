---
layout: post
title:  "Introduction to Koa.js"
author: "Mubaris NK"
comments: true
catalog: true
tags: js node koa tutorial
header-img: files/images/post16.jpg
twimg: https://i.imgur.com/gJXT4iu.png
image: https://i.imgur.com/gJXT4iu.png
---

[Koa.js](https://koajs.com/) is a minimal Node.js web framework developed by the team behind [Express.js](https://expressjs.com/). Koa uses async functions, this gives you advantage over callback functions. By default Koa does not come with any middlewares. That makes Koa very minimal and elegant. In this post we'll get started with building an API using Koa.js

Koa requires **node v7.6.0** or higher for **ES2015** and **async** function support.

### Prerequisites

- Node.js Knowledge
- ES6 Syntax Familiarity

## What are we building?

With the help of Koa, we'll build a simple [sentiment analysis](https://mubaris.com/2017/11/04/movie-sentiment-analysis/) API which takes a text as input and provides sentiment score as output. We'll use following NPM packages to build this API.

- [Koa](https://www.npmjs.com/package/koa) - Web Framework
- [Koa Router](https://www.npmjs.com/package/koa-router) - For routing
- [Koa Body Parser](https://www.npmjs.com/package/koa-bodyparser) - To parse request body
- [Sentiment](https://www.npmjs.com/package/sentiment) - Analysing the text

Let's get started building this API.

## Hello World

We'll start with a simplest example. First off, we need to install Koa. Create a new directory and we'll run the following to install Koa.

```sh
yarn add koa
```

The hello world example is simple as it gets,

```js
const Koa = require('koa');
const app = new Koa();

app.use(async ctx => {
    ctx.body = 'Hello World';
});

app.listen(3000, () => {
    console.log('Server started on localhost:3000');
});
```

First line is importing Koa. In the next line, we initialize the Koa application.

`app.use(function)` is a middleware. This gets called for every request sent to the server. And we are setting the body as "Hello World". Hence on every route, we'll get the response "Hello World". And finally we are listening on port number 3000.

## Koa Middleware

It's very easy to create a custom middleware in Koa. In the last section we used `app.use(function)`, this function can be used to create a Koa middleware. Koa middleware flows in a stack like manner, allowing you to perform actions downstream then filter and manipulate the response upstream. Koa middleware are simple functions which return a `MiddlewareFunction` with signature `(ctx, next)`. When the middleware is run, it must manually invoke `next()` to run the “downstream” middleware.

```js
const Koa = require('koa');
const app = new Koa();

app.use(async (ctx, next) => { 
    console.log('1'); 
    await next(); 
    console.log('2');
});
app.use(async (ctx, next) => {
    console.log('3'); 
    await next(); 
    console.log('4');
});
app.use(async (ctx, next) => { 
    console.log('5');
    ctx.body = 'Hello World';
    console.log('6');
});

app.listen(3000, function(){ 
    console.log('Server started on localhost:3000');
});
```

If you hit `localhost:3000` on the browser, you'll get following console output. The process goes like this,

- Browser sends the request to the server
- First middleware gets called, Prints "1"
- First middleware calls the next middleware using `next()`. First one pauses, execution goes to the next one
- Second middleware gets called, Prints "3"
- `next()`, Second pauses
- Third middleware gets called, Prints "5"
- Third middleware sends response back to the Browser "Hello World"
- Third continues, Prints "6", and execution goes upwards.
- Second middleware continues, Prints "4", execution goes upwards.
- First middleware continues, Prints "2".

```bash
Server started on localhost:3000
1
3
5
6
4
2
```

Koa Middlewares can be used for Logging, Exception Handling, Authentication, and many more things. Here's [a list of middlewares from Koa Wiki](https://github.com/koajs/koa/wiki#middleware).

Let's move on to building sentiment analysis API.

## Enter Sentiment

We'll use a Node.js library called [`sentiment`](https://www.npmjs.com/package/sentiment) to calculate sentiment scores. This library performs AFINN-based sentiment analysis. It comes with a list of words with its predefined scores. For every sentence, it finds average sentiment scores of all words in the sentiment. It gives the score in the range of -5 to 5, here -5 being most negative and 5 being most positive. We'll start with installing `sentiment`.

```sh
yarn add sentiment
```

Let's see an example of how it works

```js
const Sentiment = require('sentiment');
const sentiment = new Sentiment();
let result = sentiment.analyze('Cats are amazing.');
console.log(result);
/*
{ score: 4,
    comparative: 1.3333333333333333,
    tokens: [ 'cats', 'are', 'amazing' ],
    words: [ 'amazing' ],
    positive: [ 'amazing' ],
    negative: [] }
*/
result = sentiment.analyze('Cats are lazy');
console.log(result);
/*
{ score: -1,
    comparative: -0.3333333333333333,
    tokens: [ 'cats', 'are', 'lazy' ],
    words: [ 'lazy' ],
    positive: [],
    negative: [ 'lazy' ] }
*/
```

Here's `score` is the sum of sentiment scores of all words, and `comparative` is the average score. We're interested in `comparative` score.

Let's integrate sentiment analysis with our Koa application.

## Koa + Sentiment

We need to install `koa-router` middleware for using routes in Koa and `koa-bodyparser` for parsing request body. Let's install these with,

```sh
yarn add koa-router koa-bodyparser
```

Now we are building the final API. We'll use the following configuration for the API.

- POST request on `/analyze`
- JSON request body of the format `{"text": "The text to be analyzed"}`
- JSON response of the format `{"text": "The text to be analyzed", "score": 0.3}`
- Sentiment score in the range of -1 to 1 instead of -5 to 5

```js
const Koa = require('koa');
const Router = require('koa-router');
const Sentiment = require('sentiment');
const bodyParser = require('koa-bodyparser');

const app = new Koa();
const router = new Router();
const sentiment = new Sentiment();


// Analyze a text and return sentiment score in the range of -1 to 1
function analyze(text) {
    const result = sentiment.analyze(text);
    const comp = result.comparative;
    const out = comp / 5;
    return out;
}

// Use bodyparser middleware to parse JSON request
app.use(bodyParser());

// Define POST request route to analyze the text
router.post('/analyze', async (ctx, next) => {
    // Look for text property on request body
    const text = ctx.request.body.text;
    if (text) {
        // Analyze the given text
        const score = analyze(text);
        // Send response
        ctx.body = {
            text,
            score
        };
    } else {
        // Send error if there's not text property on the body
        ctx.status = 400;
        ctx.body = {
            "error": "Please provide a text to analyze"
        };
    }
});

// Use Koa Router middleware
app
    .use(router.routes())
    .use(router.allowedMethods());

// Finally, start the server
app.listen(3000, function(){
    console.log('Server started on localhost:3000');
});
```

That's our Sentiment Analysis API. We'll go through it line by line.

- First we import necessary libraries and initialize them.
- `analyze()` takes a text input and returns it sentiment score in the range -1 to 1
- `app.use(bodyParser())` , we tell Koa to use bodyparser middleware to parse JSON requests
- We define `/analyze` route to analyze the text. This route only accepts POST requests.
- The function in `/analyze` route tries to get text property from the request. If it's available, send the response to client with score and text. If not, we send back an error.
- We tell Koa to use Router Middleware
- And finally, start the server with `listen`

That wraps up our Sentiment Analysis API using Koa.js. Full code is available on this [Github Repo](https://github.com/mubaris/koa-playground/blob/master/part1/index.js). In Part 2 of Koa Tutorial we'll cover Logging and adding Analytics to our Sentiment Analysis API. Subscribe to Newsletter to receive Part 2.

<div id="mc_embed_signup">
<form action="//mubaris.us16.list-manage.com/subscribe/post?u=f9e9a4985cce81e89169df2bf&amp;id=3654da5463" method="post" id="mc-embedded-subscribe-form" name="mc-embedded-subscribe-form" class="validate" target="_blank" novalidate>
    <div id="mc_embed_signup_scroll">
    <label for="mce-EMAIL">Subscribe for more Awesome Posts!</label>
    <input type="email" value="" name="EMAIL" class="email" id="mce-EMAIL" placeholder="email address" required>
    <!-- real people should not fill this in and expect good things - do not remove this or risk form bot signups-->
    <div style="position: absolute; left: -5000px;" aria-hidden="true"><input type="text" name="b_f9e9a4985cce81e89169df2bf_3654da5463" tabindex="-1" value=""></div>
    <div class="clear"><input type="submit" value="Subscribe" name="subscribe" id="mc-embedded-subscribe" class="button"></div>
    </div>
</form>
</div>

# %% [markdown]
# # Assignment 3, D7015B, Industrial AI and eMaintenance - Part I: Theories & Concepts #
# 
# Isak Jonsson, isak.jonsson@gmail.com

# %% [markdown]
# ## Imports ##

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import statistics
import json
from wordcloud import WordCloud
import string
from itertools import filterfalse
import nlu
import os

# %% [markdown]
# Environment-specific settings, for getting Spark to work

# %%
os.environ["JAVA_HOME"] = "C:\\Users\\isakj\\.jdks\\corretto-11.0.22"
os.environ["HADOOP_HOME"] = "C:\\projects\\winutils\\hadoop-3.2.0"
os.environ["SPARK_HOME"] = "C:\\projects\\spark-3.2.3-bin-hadoop3.2"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["HADOOP_HOME"] + "/bin:" + os.environ["SPARK_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "jupyter"
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# %% [markdown]
# ## Open file ##

# %%
df = pd.read_excel('openDamagesinTrains.xls')

print(df.shape)
print(dict(df).keys())


# %% [markdown]
# ## Reparsing date ##
# 
# The code below is not used, but it plots the day-of-month values for all reporting dates cells where the cell actually is a date cell. Statistically, the days should be evenly spread.

# %%
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(df[df['Damage reporting date']
        .apply(lambda x : isinstance(x, datetime.datetime))]['Damage reporting date']
        .apply(lambda x : x.day), 
        bins=np.linspace(1,31,31))

ax.set_xlabel('Day of month')
ax.set_ylabel('Occurrences')

# %% [markdown]
# Reparsing
# * If cell is a datetime, create a new datetime object with day and months swapped.
# * If cell is text and can be parsed as M/D/YYYY, create a new datetime value with those elements.
# * If cell value is `'Öppen'`, store as `None`

# %%
def parse_date(value):
    if isinstance(value, datetime.datetime):
        return datetime.datetime(value.year, value.day, value.month)
    if value=='Öppen':
        return None
    m=re.fullmatch('(\\d{1,2})/(\\d{1,2})/(\\d{4})', value)
    return datetime.datetime(int(m.group(3)),int(m.group(1)),int(m.group(2)))

df['Damage reporting date'] = df['Damage reporting date'].apply(parse_date)
df['Damage closing date'] = df['Damage closing date'].apply(parse_date)

# %% [markdown]
# ## TTF and TTR extraction ##
# 
# Two arrays (DataFrames) are created. The axes are the vehicle code and the damage category, respectively. For each axis, a special value `'*'` is used to hold values for all vehicles and categories. The value in the TTF array is for each combination a list of the difference to last reported failure. The value in the TTR array is for each combination the difference between the reporting and closing date for those rows in the dataset where a closing date appears.
# 
# Note on storage. This, rather lazy, way of storing the TTF and TTR values are $O(\#failures)$, where $\#failures$ is the number of rows in the original dataset. If only mean and variance is required, a more memory-efficient way would be to store $n,\sum{x},\sum{x^2}$ for each combination, bring memory requirements down to $O(\#vehicles \times \#categories)$.

# %%

df.sort_values("Damage reporting date", inplace=True) # to make diff work

vehicles = ['*'] + list(df['Vehicle'].sort_values().unique())
categories = ['*'] + list(df['Damage category'].sort_values().unique())


ttf = pd.DataFrame('', vehicles, categories)
ttr = pd.DataFrame('', vehicles, categories)

for vehicle in vehicles:
    for category in categories:
        # find the applicable rows
        rows = df.apply(lambda x : 
                        (vehicle == '*' or x['Vehicle'] == vehicle) and
                        (category == '*' or x['Damage category'] == category), axis=1)
        ttf.loc[vehicle,category] = list(df[rows]['Damage reporting date'].diff()[1:].map(lambda x:x.days))
        closed = df[rows & df.notnull()['Damage closing date']]
        ttr.loc[vehicle,category] = list((closed['Damage closing date']-closed['Damage reporting date']).map(lambda x:x.days))


# %%
vehicle = 'F26005'
category = 'ALLMÄN FORDONSINFORMATION'
vehicles = [ vehicle ]
categories = [ category ]

rows = df.apply(lambda x : 
                        (vehicle == '*' or x['Vehicle'] == vehicle) and
                        (category == '*' or x['Damage category'] == category), axis=1)
df[rows]
list(df[rows]['Damage reporting date'].diff()[1:].map(lambda x:x.days))
print(list(df[rows]['Damage reporting date'].diff()[1:].map(lambda x:x.days)), sep=', ')

# %% [markdown]
# ## Data analysis ##

# %% [markdown]
# ### Plotting data ###

# %%

# the values below can be changed
vehicle = 'F26005'
category = 'ALLMÄN FORDONSINFORMATION'
#category = 'BROMSSYSTEM'
logarithmic = False

fig, ax = plt.subplots()
values = ttf.loc[vehicle,category]
ax.hist(values, bins=20)
if logarithmic:
    plt.yscale('log')
ax.set_xlabel('Days ($\\bar{{x}}={0:.2f}$, $\\tilde{{x}}={1:.1f}$, $\\sigma={2:.2f}$, $N={3}$)'.format(statistics.mean(values), statistics.median(values), statistics.stdev(values), len(values)))
ax.set_ylabel('Occurrences')
ax.set_title('Time-to-failure; ' + ('all vehicles' if vehicle=='*' else vehicle) + ', ' + ('all damage categories' if category=='*' else category))

fig, ax = plt.subplots()
values = ttr.loc[vehicle,category]
ax.hist(values, bins=20)
if logarithmic:
    plt.yscale('log')
ax.set_xlabel('Days ($\\bar{{x}}={0:.2f}$, $\\tilde{{x}}={1:.1f}$, $\\sigma={2:.2f}$, $N={3}$)'.format(statistics.mean(values), statistics.median(values), statistics.stdev(values), len(values)))
ax.set_ylabel('Occurrences')
ax.set_title('Time-to-repair; ' + ('all vehicles' if vehicle=='*' else vehicle) + ', ' + ('all damage categories' if category=='*' else category))


# %% [markdown]
# ### Data in tabular form ###
# 
# With every thing as lists in dataframe, it is very easy to do statistical analyses for the entire dataset.

# %%
ttf.applymap(len)
ttf.applymap(statistics.mean)
ttf.applymap(statistics.stdev)

# %% [markdown]
# ## Word cloud ##
# 
# The data category field is rather "noisy", when it comes to create a word cloud. 
# In order to clean it up a bit, we want to remove everything that is _too_ specific,
# like the number of millimeter a door gap is.
# Also, we want to remove words that are not specific at all, so called _stop words_.
# 
# One of the challenges is that the text is in Swedish, and as expected, the amount of NLP libraries
# for Swedish is much lower than English NLP libraries. The strategy I ended up with is:
# 
# 1. split everything into words, just by splitting on space
# 2. make everything lowercase
# 3. remove every "word" that has a digit it it
# 4. remove every word that starts with a `/` (it is a signature)
# 5. remove leading and trailing punctuation, which can lead to empty words, which are deleted
# 6. remove all stopwords in https://github.com/stopwords-iso/stopwords-sv
# 7. convert all words to a string and lemmatize the string using https://sparknlp.org/2020/05/05/lemma_sv.html
# 8. remove stopwords, again, using https://sparknlp.org/2020/07/14/stopwords_sv.html
# 
# One would think that the correct order might be to first lemmatize, then remove stopwords.
# But one specific problem I found was the word "för" in the the adverb meaning "too": "gapet är för stort".
# However, the lemmatizer treated it as the verb "för" ("bring"), replacing it with the infinitive form
# "föra", which was not caught by the stopword algorithm.

# %%
stopwords_sv = json.load(open('stopwords-sv/stopwords-sv.json'))
rows = df.apply(lambda x : isinstance(x['Damage description'], str), axis=1)
words = df[rows]['Damage description'].apply(str.split).sum()
# everything lowercase
words = [x.lower() for x in words]
# remove all numbers
words = list(filter(lambda x : not re.match('.*[0-9].*', x), words))
# remove signatures '/...'
words = list(filter(lambda x : not re.match('^/.*', x), words))
# remove remaining leading and trailing punctuation
words = [x.strip(string.punctuation) for x in words]
# remove empty words
words = list(filter(None, words))
# remove stopwords from JSON list
words = list(filterfalse(stopwords_sv.__contains__, words))

text = ' '.join(words)

nlu_stopwords = nlu.load("sv.stopwords")
nlu_lemma = nlu.load('sv.lemma')

# lemmatize words
words = nlu_lemma.predict([text], output_level='document').lem[0]
# remove stopwords, again
words = list(nlu_stopwords.predict(' '.join(words))['stopword_less'].dropna())

# %%
wordcloud = WordCloud(
    stopwords={}, 
    normalize_plurals=False, 
    random_state=1,
    width=1200,
    height=600).generate(' '.join(words))
oldsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (wordcloud.width/100,wordcloud.height/100)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.rcParams["figure.figsize"] = oldsize

# %% [markdown]
# ## Machine learning for category prediction ##
# 
# We use a [TD-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) classifier for a simple way of matching words with labels (categories).
# 
# Before vectorizing the data, we clean it by using pretty much the same operation as for the word cloud. When it comes to lemmatizing, I have tested both the WordNet lemmatizer and the NLU lemmatizer.
# 
# * The WordNet lemmatizer is for English, so it should not perform very well.
# * The NLU lemmatizer is for Swedish. However, it is very slow per call, so I build a larger string with delimiters (`'/#/'`) for values, run the lemmatizer, and then split the result.
# 
# Not suprisingly, the Swedish lemmatizer is better than the English one. But the difference is marginal:
# 
# * No lemmatizer: 52% accuracy
# * WordNet lemmatizer: 54% accuracy
# * NLU lemmatizer: 55% accuracy

# %%
use_nlu_lemmatizer = True
use_wordnet_lemmatizer = False

import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Tokenization och preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('swedish'))

def preprocess_text(text):
    words = text.split()
    words = list(filter(lambda x : not re.match('.*[0-9].*', x), words))
    # remove signatures '/...'
    words = list(filter(lambda x : not re.match('^/.*', x), words))
    # remove remaining leading and trailing punctuation
    words = [x.strip(string.punctuation) for x in words]
    # remove empty words
    words = list(filter(None, words))
    # remove stopwords from JSON list
    words = list(filterfalse(stop_words.__contains__, words))
    words = list(filterfalse(stopwords_sv.__contains__, words))
    text = ' '.join(words)
    tokens = word_tokenize(text.lower())
    if use_wordnet_lemmatizer:
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

df_clean = df[df['Damage description'].apply(lambda x : isinstance(x, str))]

if use_nlu_lemmatizer:
    sss = ' /#/ '.join(df_clean['Damage description'].apply(preprocess_text))
    lem = nlu_lemma.predict(sss, output_level='document')
    X = ' '.join(lem['lem'][0]).split('/#/')
else:
    X = list(df_clean['Damage description'].apply(preprocess_text))

y = list(df_clean['Damage category'])
# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Multinomial Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# predict the test set
y_pred = classifier.predict(X_test)

# find the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %% [markdown]
# ### Predictions results ###
# 
# First some interpretations from the classifier, what are the most "important" words for each damage category?

# %%
terms = vectorizer.get_feature_names_out()
DD = pd.DataFrame()
for i, doc in enumerate(classifier.classes_):
    D = pd.DataFrame({'word':vectorizer.get_feature_names_out(), 'score':classifier.feature_count_[i,:]})
    D.sort_values('score', inplace=True, ascending=False)
    DD[doc] = list(D['word'])
DD[0:10]

# %% [markdown]
# And finally, some test results for made up sentences.

# %%
def predict(sentence):
    sentence = preprocess_text(sentence)
    if use_nlu_lemmatizer:
        lem = nlu_lemma.predict(sentence, output_level='document')
        sentence = ' '.join(lem['lem'][0])
    new_sentence_vectorized = vectorizer.transform([sentence])
    predicted_label = classifier.predict(new_sentence_vectorized)[0]
    return predicted_label

new_sentence = "Dörr 41 har väldigt högt pip. Måste sänkas en aning, skär i öronen."
print("Predicted Label for '{}': {}".format(new_sentence, predict(new_sentence)))

new_sentence = "Handfat har spruckit mitt itu!"
print("Predicted Label for '{}': {}".format(new_sentence, predict(new_sentence)))



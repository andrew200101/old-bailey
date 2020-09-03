

from sklearn import svm
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import requests
import time
from bs4 import BeautifulSoup
from math import nan
from collections import Counter
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim import corpora, models

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import itertools
import scipy as sp
from matplotlib import cm

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import seaborn as sn
from nltk import RegexpParser
from textblob import TextBlob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# additional feature
def occupation(case_soup):
    occupations = []
    for rs in case_soup.find_all('rs'):
        if rs.get('type') == 'occupation':
            occupations.append(rs.text)
    return occupations

# the 3 features below are taken from the lab


def case_date(case_soup):
    for element in case_soup.findAll('interp'):
        if element.attrs['type'] == 'date':
            return element.attrs['value']


def people_in_case(case_soup):
    people = []
    for persName in case_soup.find_all('persname'):
        person = {}
        person["type"] = persName.attrs.get("type")
        for interp in persName.find_all('interp'):
            fieldName = interp.attrs["type"]
            fieldValue = interp.attrs["value"]
            person[fieldName] = fieldValue
        people.append(person)
    return people


def case_descriptions(case_soup):
    descriptions = {}
    for rs in case_soup.find_all('rs'):
        desc = {}
        for interp in rs.find_all('interp'):
            fieldName = interp.attrs["type"]
            fieldValue = interp.attrs["value"]
            desc[fieldName] = fieldValue
        desc["text"] = rs.text.strip()
        descriptions[rs.attrs['type']] = desc
    return descriptions


trials = []
for file in os.listdir('data/OBO_XML_72/sessionsPapers'):
    trials.append(file)


def table_of_cases(xml_file_names):
    rows = []
    for xml_file in xml_file_names:
        with open(f'data/OBO_XML_72/sessionsPapers/'+xml_file, "r") as xml_file:
            cases = BeautifulSoup(xml_file)

        for case in cases.findAll('div1'):
            people = people_in_case(case)
            date = case_date(case)
            descriptions = case_descriptions(case)
            occupations = occupation(case)
            row = {
                "date": date,
                "id": case.attrs["id"],
                # split on all whitespace, then join on " ", to remove long sequences of whitespace
                "text": " ".join(case.text.split()),
                "any_defendant_female": False,
                "any_defendant_male": False,
                "any_victim_female": False,
                "any_victim_male": False,
            }
            if "offenceDescription" in descriptions:
                # `dictionary.get(key, default)` is the same as `dictionary[key] if key in dictionary else default`
                row["offenceText"] = descriptions["offenceDescription"].get(
                    "text", nan)
                row["offenceCategory"] = descriptions["offenceDescription"].get(
                    "offenceCategory", nan)
                row["offenceSubcategory"] = descriptions["offenceDescription"].get(
                    "offenceSubcategory", nan)
            if "verdictDescription" in descriptions:
                row["verdictText"] = descriptions["verdictDescription"].get(
                    "text", nan)
                row["verdictCategory"] = descriptions["verdictDescription"].get(
                    "verdictCategory", nan)
            if "punishmentDescription" in descriptions:
                row["punishmentText"] = descriptions["punishmentDescription"].get(
                    "text", nan)
                row["punishmentCategory"] = descriptions["punishmentDescription"].get(
                    "punishmentCategory", nan)
                row["punishmentSubcategory"] = descriptions["punishmentDescription"].get(
                    "punishmentSubcategory", nan)

            row['occupations'] = occupations

            for person in people:
                if person.get("type") == "defendantName" and person.get("gender") == "female":
                    row["any_defendant_female"] = True
                if person.get("type") == "defendantName" and person.get("gender") == "male":
                    row["any_defendant_male"] = True
                if person.get("victim") == "victimName" and person.get("gender") == "female":
                    row["any_victim_female"] = True
                if person.get("victim") == "victimName" and person.get("gender") == "male":
                    row["any_victim_male"] = True

                if person.get("type") == "defendantName" and person.get("gender") == "female":
                    row["any_defendant_female"] = True
            rows.append(row)
    return pd.DataFrame(rows)


trials_df = table_of_cases(trials[0:40])


def drop_Na(table, column):  # drops all NaN values in specified string columns/return new table
    return table[table[column].notna()]


trials_df = drop_Na(trials_df, 'offenceText')
trials_df = drop_Na(trials_df, 'verdictCategory')
trials_df = trials_df.reset_index(drop=True)

target_classes = ['guilty', 'notGuilty']
trials_df = trials_df[trials_df['verdictCategory'].isin(target_classes)]


stop = stopwords.words('english')
punctuation = string.punctuation
more_stops = ['--', '``', "''", "s'", "\'s", "n\'t", "...", "\'m", "-*-", "-|"]
stemmer = SnowballStemmer("english")


def preprocess(text):
    lowered_string = text.lower()
    tokens = nltk.word_tokenize(lowered_string)
    filtered_tokens = [
        x for x in tokens if x not in punctuation and x not in more_stops]
    stopped_tokens = [x for x in filtered_tokens if not x in stop]
    stemmed_tokens = [stemmer.stem(i) for i in stopped_tokens]

    return stemmed_tokens


trials_df['processed_text'] = trials_df['offenceText'].apply(preprocess)


grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    PP: {<IN><NP>}
    VP: {<VB.*><NP|PP|CLAUSE>+}
    CLAUSE: {<NP><VP>}
    """


def chunked(tokens):
    cp = RegexpParser(grammar)
    tagged = nltk.pos_tag(tokens)
    chunked = cp.parse(tagged)
    return str(chunked)


trials_df['chunked'] = trials_df['processed_text'].apply(chunked)


text_list = list(trials_df['chunked'])

tf = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 3),
                     min_df=0,
                     stop_words='english')

tf_idf_matrix = tf.fit_transform(text_list)


document_frequency = {}
for i in range(len(text_list)):
    tokens = text_list[i].split(' ')
    tokens = [t for t in tokens if t != '']
    for w in tokens:
        try:
            document_frequency[w].add(i)
        except:
            document_frequency[w] = {i}
for i in document_frequency:
    document_frequency[i] = len(document_frequency[i])
document_frequency_r = {k: v for k, v in sorted(
    document_frequency.items(), key=lambda item: item[1], reverse=True)}
document_frequency_a = {k: v for k, v in sorted(
    document_frequency.items(), key=lambda item: item[1], reverse=False)}


keys1 = list(document_frequency_r.keys())[0:20]
values1 = list(document_frequency_r.values())[0:20]
keys2 = list(document_frequency_a.keys())[0:20]
values2 = list(document_frequency_a.values())[0:20]

fig1 = plt.figure(figsize=(30, 30))

ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_title('Document Frequency - Most Frequent', size=40)

ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_title('Document Frequency - Least Frequent', size=40)

ax1.barh(keys1, values1, color='red')
ax2.barh(keys2, values2)

trials_df['verdictCategory'].value_counts()


y = trials_df['verdictCategory']

X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, y,
                                                    train_size=.80,
                                                    test_size=.20)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                            train_size=.75,
                                                            test_size=.25)


def metric(predicted, actual, title):
    actual_1 = list(actual)
    FN = 0
    FP = 0
    for i in range(len(predicted)):
        if (predicted[i] == 'notGuilty') & (actual_1[i] == 'guilty'):
            FN += 1
        if (predicted[i] == 'guilty') & (actual_1[i] == 'notGuilty'):
            FP += 1

    print('Accuracy: ', np.mean(predicted == actual_1))
    print('Falsely Convicted: ', FP/len(predicted))
    print('Falsely Let Go: ', FN/len(predicted))

    lables_dict = {}
    values = actual.value_counts().axes[0].tolist()
    for i in range(0, len(values)):
        lables_dict[i] = values[i]
    cf_matrix = confusion_matrix(actual, predicted)
    df_cm = pd.DataFrame(cf_matrix, range(len(values)),
                         range(len(values)))
    df_cm = df_cm.rename(index=str, columns=lables_dict)
    df_cm.index = values
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size

    sn.heatmap(df_cm,
               annot=True,
               annot_kws={"size": 16})
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


svm = svm.LinearSVC(class_weight="balanced")
svm_model = svm.fit(X_train, y_train)
svm_pred = svm_model.predict(X_validate)


title = 'Linear SVM Prediction Results'
metric(svm_pred, y_validate, title)


log_reg = LogisticRegression(class_weight="balanced")
log_reg_model = log_reg.fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_validate)


title = 'Logistic Regression Prediction Results'
metric(log_reg_pred, y_validate, title)


rfm = RandomForestClassifier(class_weight="balanced")
rfm.fit(X_train, y_train)
rfm_pred = rfm.predict(X_validate)


title = 'RFM Prediction Results'
metric(rfm_pred, y_validate, title)


feature_names = tf.get_feature_names()
feature_importances = pd.DataFrame({'feature': feature_names,
                                    'importance': rfm.feature_importances_})\
    .sort_values('importance', ascending=False)
feature_importances.head(15)


svm_pred_test = svm_model.predict(X_test)
title = 'CVM Test Set Results'
metric(svm_pred_test, y_test, title)


trials_df.reset_index(drop=True)


def unknown_occupation(occupation):
    if len(occupation) == 0:
        x = '[unknown]'
        return (x)
    else:
        return str(occupation)


trials_df['occupations'] = trials_df['occupations'].apply(unknown_occupation)
trials_df = trials_df.reset_index(drop=True)


dummies = pd.get_dummies(trials_df['occupations'], dtype=bool)
occupation_all = list(dummies.columns)
trials_df = pd.concat([trials_df, dummies], axis=1)
features = ['any_defendant_female', 'any_defendant_male',
            'any_victim_female', 'any_victim_male'] + occupation_all
feature_names += features
trials_df.head(5)


new_tf_idf = sp.sparse.hstack((tf_idf_matrix, trials_df[features]))
y = trials_df['verdictCategory']

X_train, X_test, y_train, y_test = train_test_split(new_tf_idf, y,
                                                    train_size=.80,
                                                    test_size=.20)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                            train_size=.75,
                                                            test_size=.25)

svm = svm.LinearSVC(class_weight="balanced")
svm_model = svm.fit(X_train, y_train)
svm_pred = svm_model.predict(X_validate)


title = 'Linear SVM Prediction Results'
metric(svm_pred, y_validate, title)


log_reg = LogisticRegression(class_weight="balanced")
log_reg_model = log_reg.fit(X_train, y_train)
log_reg_pred = log_reg_model.predict(X_validate)
title = 'Logistic Regression Prediction Results'
metric(log_reg_pred, y_validate, title)


rfm = RandomForestClassifier(class_weight="balanced")
rfm.fit(X_train, y_train)
rfm_pred = rfm.predict(X_validate)
title = 'RFM Prediction Results'
metric(rfm_pred, y_validate, title)


svm_test = svm_model.predict(X_test)
title = 'SVM Test Set Results'
metric(svm_test, y_test, title)


coefs = svm_model.coef_[0]
coefficients = []
for i in coefs:
    coefficients.append(i)


len(feature_names)


len(coefficients)


new_feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': coefficients})\
    .sort_values('importance', ascending=False)


new_feature_importances.head(10)

first_200 = list(new_feature_importances['feature'])[0:200]
first_5000 = list(new_feature_importances['feature'])[0:5000]


matched_features_200 = []
matched_features_5000 = []
for i in range(0, len(feature_names)):
    if feature_names[i] in first_200:
        matched_features_200.append(i)
    if feature_names[i] in first_5000:
        matched_features_5000.append(i)


tf_idf_matrix_200 = new_tf_idf.tocsr()[:, matched_features_200]
tf_idf_matrix_5000 = new_tf_idf.tocsr()[:, matched_features_5000]


y = trials_df['verdictCategory']

# Train/Test Split
X_train_200, X_test_200, y_train_200, y_test_200 = train_test_split(tf_idf_matrix_200, y,
                                                                    train_size=.80,
                                                                    test_size=.20)

# Train/Validation Split
X_train_200, X_validate_200, y_train_200, y_validate_200 = train_test_split(X_train_200, y_train_200,
                                                                            train_size=.75,
                                                                            test_size=.25)

X_train_5000, X_test_5000, y_train_5000, y_test_5000 = train_test_split(tf_idf_matrix_5000, y,
                                                                        train_size=.80,
                                                                        test_size=.20)

# Train/Validation Split
X_train_5000, X_validate_5000, y_train_5000, y_validate_5000 = train_test_split(X_train_5000, y_train_5000,
                                                                                train_size=.75,
                                                                                test_size=.25)


svm = svm.LinearSVC(class_weight="balanced")
svm_model = svm.fit(X_train_200, y_train_200)
svm_pred_200 = svm_model.predict(X_validate_200)


title = 'SVM - 200 Features'
metric(svm_pred, y_validate_200, title)


svm = svm.LinearSVC(class_weight="balanced")
svm_model = svm.fit(X_train_5000, y_train_5000)
svm_pred_5000 = svm_model.predict(X_validate_5000)


title = 'SVM - 5000 Features'
metric(svm_pred, y_validate_5000, title)

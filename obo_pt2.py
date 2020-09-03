

import warnings
from textblob import TextBlob
import seaborn as sns
import re


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
from scipy import stats
import matplotlib.pyplot as plt
import math
import itertools
import scipy as sp
from nltk import RegexpParser
import matplotlib.style as style


trials = []
for file in os.listdir('data/OBO_XML_72/sessionsPapers'):
    trials.append(file)


def case_date(case_soup):
    for element in case_soup.findAll('interp'):
        if element.attrs['type'] == 'date':
            return element.attrs['value']


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


def table_of_cases(xml_file_names):
    rows = []
    for xml_file in xml_file_names:
        with open(f'data/OBO_XML_72/sessionsPapers/'+xml_file, "r") as xml_file:
            cases = BeautifulSoup(xml_file)
        for case in cases.findAll('div1'):
            date = case_date(case)
            descriptions = case_descriptions(case)
            row = {
                "date": date,
                "id": case.attrs["id"],
                # split on all whitespace, then join on " ", to remove long sequences of whitespace
                "text": " ".join(case.text.split()),
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
                row["verdictSubcategory"] = descriptions["verdictDescription"].get(
                    "verdictSubcategory", nan)
            if "punishmentDescription" in descriptions:
                row["punishmentText"] = descriptions["punishmentDescription"].get(
                    "text", nan)
                row["punishmentCategory"] = descriptions["punishmentDescription"].get(
                    "punishmentCategory", nan)
                row["punishmentSubcategory"] = descriptions["punishmentDescription"].get(
                    "punishmentSubcategory", nan)
            rows.append(row)
    return pd.DataFrame(rows)


def trial_files(start, end):
    dates = []
    for year in range(start, end):
        for month in range(1, 13):
            if month < 10:
                month = str(0) + str(month)
            else:
                month = str(month)
            for day in range(1, 32):
                if day < 10:
                    day = str(0) + str(day)
                else:
                    day = str(day)
                dates.append((str(year)+str(month)+str(day)))
    trials_all = []
    for trial in trials:
        if trial[0:8] in dates:
            trials_all.append(trial)
    return trials_all


trials_bloody_era = trial_files(1753, 1758)  # extended years
trials_control = trial_files(1824, 1825)


bloody_df = table_of_cases(trials_bloody_era)
control_df = table_of_cases(trials_control)


def pie_visualization(series, title):
    size = list(series.values)
    label = list(series.index)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    theme = plt.get_cmap('hsv')
    ax1.set_prop_cycle("color", [theme(1. * i / len(size))
                                 for i in range(len(size))])
    ax1.axis('equal')
    title = plt.title(title)
    pie = plt.pie(size)
    labels = label
    plt.legend(label, fontsize=10, bbox_to_anchor=(-0.05, 1.))


pie_visualization(
    bloody_df['punishmentCategory'].value_counts(), 'Punishment Types')


pie_visualization(bloody_df['punishmentSubcategory'].value_counts(
), 'Punishment Subcategory Types')


def get_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


bloody_df['text_polarity'] = bloody_df['text'].apply(get_polarity)
control_df['text_polarity'] = control_df['text'].apply(get_polarity)


def drop_Na(table, column):  # drops rows with NaN values in 'text_polarity' column/return a new table
    return table[table[column].notna()]


bloody_df_death = drop_Na(bloody_df, 'text_polarity')
control_df_death = drop_Na(bloody_df, 'text_polarity')


def boolean_vsub(verd):
    if type(verd) != str:
        return 0
    else:
        return 1


bloody_df['verdict_sub_bool'] = bloody_df['verdictSubcategory'].apply(
    boolean_vsub)
control_df['verdict_sub_bool'] = control_df['verdictSubcategory'].apply(
    boolean_vsub)


bins = np.arange(-1.0, 1.01, 0.1)

control_prop = control_df.groupby([pd.cut(
    control_df.text_polarity, bins), 'verdict_sub_bool'])    .agg('count')[['date']].reset_index()
bloody_prop = bloody_df.groupby([pd.cut(
    bloody_df.text_polarity, bins), 'verdict_sub_bool'])    .agg('count')[['date']].reset_index()

x_bloody = bloody_prop[bloody_prop['verdict_sub_bool']
                       == 0].reset_index(drop=True)
y_bloody = bloody_prop[bloody_prop['verdict_sub_bool']
                       == 1].reset_index(drop=True)
x_control = control_prop[control_prop['verdict_sub_bool']
                         == 0].reset_index(drop=True)
y_control = control_prop[control_prop['verdict_sub_bool']
                         == 1].reset_index(drop=True)

props_control = y_control['date']/(y_control['date']+x_control['date'])
props_bloody = y_bloody['date']/(y_bloody['date']+x_bloody['date'])
polarity_bins = x_bloody['text_polarity']


bloody_prop_df = pd.concat([polarity_bins, props_bloody], axis=1).rename(
    columns={"date": "Partial Verdict Proportion"})
control_prop_df = pd.concat([polarity_bins, props_control], axis=1).rename(
    columns={"date": "Partial Verdict Proportion"})

bloody_prop_df = bloody_prop_df.replace(float('NaN'), 0)
control_prop_df = control_prop_df.replace(float('NaN'), 0)


def interval_clean(interval):
    return interval.mid


bloody_prop_df['Offence Polarity'] = bloody_prop_df['text_polarity'].apply(
    interval_clean)
bloody_prop_df['group'] = 'The Bloody Era'
control_prop_df['Offence Polarity'] = control_prop_df['text_polarity'].apply(
    interval_clean)
control_prop_df['group'] = 'Post 1823'

both_prop_df = pd.concat([bloody_prop_df, control_prop_df])


# Below are the distributions of partial verdict proportions given the offence polarity score. To the left is the distribution during the Bloody Code era and to the left is after The Judgment of Death Act 1823.


g = sns.relplot(x="Offence Polarity", y="Partial Verdict Proportion", col="group",
                kind="line", data=both_prop_df, height=10)


# Joinplot from Seaborn allows me to view both a joint distribution and its marginals at once. For example, beyond visualizing how polarity and partial verdicts are distributed individually, I am also able to view their joint distribution.


sns.jointplot(x="Offence Polarity", y="Partial Verdict Proportion",
              data=bloody_prop_df, kind="kde", color='r')


# <b>Distribution after Judgement of Death Act</b>


sns.jointplot(x="Offence Polarity", y="Partial Verdict Proportion",
              data=control_prop_df, kind="kde", color='g')


thefts = ['theft', 'violentTheft']
bloody_df_thefts = bloody_df[bloody_df['offenceCategory'].isin(thefts)]
control_df_thefts = control_df[control_df['offenceCategory'].isin(thefts)]


def amount_stolen(offenceText):
    value = '\d+ [a-z]\. \d+ [a-z]\.|\d+ ?[a-z]\.'
    return re.findall(value, offenceText)


warnings.filterwarnings("ignore")

bloody_df_thefts['value_stolen'] = bloody_df_thefts['offenceText'].apply(
    amount_stolen)
control_df_thefts['value_stolen'] = control_df_thefts['offenceText'].apply(
    amount_stolen)


bloody_df_thefts = bloody_df_thefts[bloody_df_thefts.value_stolen.map(len) > 0]
control_df_thefts = control_df_thefts[control_df_thefts.value_stolen.map(
    len) > 0]


bloody_df_thefts = bloody_df_thefts.reset_index(drop=True)
control_df_thefts = control_df_thefts.reset_index(drop=True)


def value_change(values):
    total = 0
    for i in range(0, len(values)):
        currency = values[i].split(' ')
        if currency[len(currency)-1] == 'l.':
            total += (int(values[i].split(' ')[0])*240)
        if currency[len(currency)-1] == 's.':
            total += (int(values[i].split(' ')[0])*12)
        if currency[len(currency)-1] == 'd.':
            total += (int(values[i].split(' ')[0]))
    return total


bloody_df_thefts['value_pennies'] = bloody_df_thefts['value_stolen'].apply(
    value_change)
control_df_thefts['value_pennies'] = control_df_thefts['value_stolen'].apply(
    value_change)

bloody_df_thefts = bloody_df_thefts[bloody_df_thefts['value_pennies'] < 4000]
control_df_thefts = control_df_thefts[control_df_thefts['value_pennies'] < 4000]


bins = np.arange(min(bloody_df_thefts['value_pennies']), max(
    bloody_df_thefts['value_pennies'])/4, 30)
control_theft_prop = control_df_thefts.groupby([pd.cut(
    control_df_thefts.value_pennies, bins), 'verdict_sub_bool'])    .agg('count')[['date']].reset_index()
bloody_theft_prop = bloody_df_thefts.groupby([pd.cut(
    bloody_df_thefts.value_pennies, bins), 'verdict_sub_bool'])    .agg('count')[['date']].reset_index()

x_bloody = bloody_theft_prop[bloody_theft_prop['verdict_sub_bool'] == 0].reset_index(
    drop=True)
y_bloody = bloody_theft_prop[bloody_theft_prop['verdict_sub_bool'] == 1].reset_index(
    drop=True)
x_control = control_theft_prop[control_theft_prop['verdict_sub_bool'] == 0].reset_index(
    drop=True)
y_control = control_theft_prop[control_theft_prop['verdict_sub_bool'] == 1].reset_index(
    drop=True)

theft_props_control = y_control['date']/(y_control['date']+x_control['date'])
theft_props_bloody = y_bloody['date']/(y_bloody['date']+x_bloody['date'])
theft_polarity_bins = x_bloody['value_pennies']


bloody_theft_prop_df = pd.concat([theft_polarity_bins, theft_props_bloody], axis=1)                        .rename(
    columns={"date": "Proportion of Partial Verdicts"})
control_theft_prop_df = pd.concat([theft_polarity_bins, theft_props_control], axis=1)                            .rename(
    columns={"date": "Proportion of Partial Verdicts"})

bloody_theft_prop_df = bloody_theft_prop_df.replace(float('NaN'), 0)
control_theft_prop_df = control_theft_prop_df.replace(float('NaN'), 0)


def interval_clean(interval):
    return interval.mid


bloody_theft_prop_df['Value Stolen (in Pennies)'] = bloody_theft_prop_df['value_pennies'].apply(
    interval_clean)
bloody_theft_prop_df['group'] = 'The Bloody Era'
control_theft_prop_df['Value Stolen (in Pennies)'] = control_theft_prop_df['value_pennies'].apply(
    interval_clean)
control_theft_prop_df['group'] = 'Post 1823'

both_theft_prop_df = pd.concat([bloody_theft_prop_df, control_theft_prop_df])


g = sns.relplot(x="Value Stolen (in Pennies)", y="Proportion of Partial Verdicts", col="group",
                kind="line", data=both_theft_prop_df, height=10)

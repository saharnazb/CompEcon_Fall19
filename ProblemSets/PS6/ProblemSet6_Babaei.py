# -*- coding: utf-8 -*-
# Problem Set 6
# Saharnaz Babaei
#-----------------------------------------------
#-----------------------------------------------
'''
Research question (API Application):
    What is the relationship between aided student and college outcome measures?


My Sketch to Answer the Question:
    - Import data and convert to a dataframe (Outcome Measures and Student Characteristics)
    - Plot the graph to show the relationship between aided students and the college outcome variables

Variables of Choice:
(Using IPEDS and Scorecard Data)
'completion_rate_4yr' = Graduation rate within 4 years
'completion_rate_6yr' = Graduation rate within 6 years
'completion_rate_8yr' = Graduation rate within 8 years
'lowincome_pct' = Share of aided students who have family incomes between $0 and $30,000 in nominal dollars
'midincome1_pct' = Share of aided students who have family incomes between $30,001 and $48,000 in nominal dollars
'midincome2_pct' = Share of aided students who have family incomes between $48,001 and $75,000 in nominal dollars
'highincome1_pct' = Share of aided students who have family incomes between $75,001 and $110,000 in nominal dollars
'highincome2_pct' = Share of aided students who have family incomes above $110,001 in nominal dollars
'dependent_income_mean' = Average family income of dependent students (in real 2015 dollars)
'independent_income_mean' = Average family income of independent students (in real 2015 dollars)


'''

#import pandas_datareader.data as web
import requests
import json
#import urllib
from json import loads
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib
from scipy.stats import linregress
#-----------------------------------------------
#-----------------------------------------------
#OUTCOME MEASURE
# Endpoint URL's have the format:   /api/v1/college-university/ipeds/outcome-measures/{year}/
# It did not let me to access the data from different years at the same time

url15 = "https://educationdata.urban.org/api/v1/college-university/ipeds/outcome-measures/2015/"
response15 = requests.get(url15)
outcome15 = response15.text
# I want to keep the data from API for off-line use:
#with open("oc15.json", 'w') as f:
#    json.dump(outcome15, f)
oc15 = pd.read_json(r'C:\Users\PhD.Econ\Desktop\PS6\oc15.json', orient = 'records')
#-----------------------------------------------
url16 = "https://educationdata.urban.org/api/v1/college-university/ipeds/outcome-measures/2016/"
response16 = requests.get(url16)
outcome16 = response16.text
#with open("oc16.json", 'w') as f:
#    json.dump(outcome16, f)
oc16 = pd.read_json(r'C:\Users\PhD.Econ\Desktop\PS6\oc16.json', orient = 'records')
#-----------------------------------------------
# Appending two years of data for outcome measures
df_oc = oc15.append(oc16, ignore_index = True)
df_oc.describe()
#-----------------------------------------------
#-----------------------------------------------
# STUDENT CHARACTERISTICS
# Endpoint URL's have the format:    /api/v1/college-university/scorecard/student-characteristics/{year}/aid-applicants/

url_st15 = "https://educationdata.urban.org/api/v1/college-university/scorecard/student-characteristics/2015/aid-applicants/"
response_st15 = requests.get(url_st15)
st15 = response_st15.text
#with open("st15.json", 'w') as f:
#    json.dump(st15, f)
st15 = pd.read_json(r'C:\Users\PhD.Econ\Desktop\PS6\st15.json', orient = 'records')
#-----------------------------------------------
st16 = "https://educationdata.urban.org/api/v1/college-university/scorecard/student-characteristics/2016/aid-applicants/"
st16 = requests.get(st16)
st16 = st16.text
#with open("st16.json", 'w') as f:
#    json.dump(st16, f)
st16 = pd.read_json(r'C:\Users\PhD.Econ\Desktop\PS6\st16.json', orient = 'records')
#-----------------------------------------------
# Appending two years of data
df_st = st15.append(st16, ignore_index = True)
df_st.describe()
#-----------------------------------------------
#-----------------------------------------------
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
df_merge = df_oc.merge(df_st, on=['unitid', 'year', 'fips'])
df = df_merge[['year', 'unitid', 'fips', 'completion_rate_4yr', 'completion_rate_6yr',
               'completion_rate_8yr', 'lowincome_pct', 'midincome1_pct', 'midincome2_pct',
               'highincome1_pct', 'highincome2_pct', 'dependent_income_mean',
               'independent_income_mean']]
'''
Special values
-1—Missing/not reported
-2—Not applicable
-3—Suppressed data
Dropping these values as follows:
'''
data = df[(df >= 0).all(1)]
data[data['year']==2016].describe()
#-----------------------------------------------
#-----------------------------------------------
# Plot data (y='completion_rate_6yr')
plt.figure(figsize=(10,10))
ax = plt.gca()

data[data['year']==2016].plot(x='lowincome_pct', y='completion_rate_6yr', kind='scatter', color='blue',
                              legend=True, label='Low Income Family', ax=ax)
#data[data['year']==2016].plot(x='midincome1_pct', y='completion_rate_6yr', kind='scatter', color='yellow',
                               #legend=True, label='Middle Income Family', ax=ax)
data[data['year']==2016].plot(x='midincome2_pct', y='completion_rate_6yr', kind='scatter', color='red',
                              legend=True, label='Middle Income Family', ax=ax)
#data[data['year']==2016].plot(x='highincome1_pct', y='completion_rate_6yr', kind='scatter', color='purple',
                              #legend=True, label='High Income Family', ax=ax)
data[data['year']==2016].plot(x='highincome2_pct', y='completion_rate_6yr', kind='scatter', color='green',
                              legend=True, label='High Income Family', ax=ax)

x1=data[data['year']==2016].lowincome_pct
y=data[data['year']==2016].completion_rate_6yr
stats = linregress(x1, y)
m = stats.slope
b = stats.intercept
plt.scatter(x1, y)
plt.plot(x1, m * x1 + b, color='blue')

x2=data[data['year']==2016].midincome2_pct
y=data[data['year']==2016].completion_rate_6yr
stats = linregress(x2, y)
m = stats.slope
b = stats.intercept
plt.scatter(x2, y)
plt.plot(x2, m * x2 + b, color='red')

x3=data[data['year']==2016].highincome2_pct
y=data[data['year']==2016].completion_rate_6yr
stats = linregress(x3, y)
m = stats.slope
b = stats.intercept
plt.scatter(x3, y)
plt.plot(x3, m * x3 + b, color='green', )

plt.xlabel("Share of aided students", fontsize=20)
plt.ylabel("Graduation rate within 6 years", fontsize=20)
plt.savefig("comp6.png")

plt.show()
#-----------------------------------------------
#-----------------------------------------------
# Plot data (y='completion_rate_8yr')
plt.figure(figsize=(10,10))
ax = plt.gca()

data[data['year']==2016].plot(x='lowincome_pct', y='completion_rate_8yr', kind='scatter', color='blue',
                              legend=True, label='Low Income Family',ax=ax)
#data[data['year']==2016].plot(x='midincome1_pct', y='completion_rate_8yr', kind='scatter', color='yellow',
                               #legend=True, label='Middle Income Family',ax=ax)
data[data['year']==2016].plot(x='midincome2_pct', y='completion_rate_8yr', kind='scatter', color='red',
                              legend=True, label='Middle Income Family',ax=ax)
#data[data['year']==2016].plot(x='highincome1_pct', y='completion_rate_8yr', kind='scatter', color='purple',
                               #legend=True, label='High Income Family', ax=ax)
data[data['year']==2016].plot(x='highincome2_pct', y='completion_rate_8yr', kind='scatter', color='green',
                              legend=True, label='High Income Family',ax=ax)

x1=data[data['year']==2016].lowincome_pct
y=data[data['year']==2016].completion_rate_8yr
stats = linregress(x1, y)
m = stats.slope
b = stats.intercept
plt.scatter(x1, y)
plt.plot(x1, m * x1 + b, color='blue')

x2=data[data['year']==2016].midincome2_pct
y=data[data['year']==2016].completion_rate_8yr
stats = linregress(x2, y)
m = stats.slope
b = stats.intercept
plt.scatter(x2, y)
plt.plot(x2, m * x2 + b, color='red')

x3=data[data['year']==2016].highincome2_pct
y=data[data['year']==2016].completion_rate_8yr
stats = linregress(x3, y)
m = stats.slope
b = stats.intercept
plt.scatter(x3, y)
plt.plot(x3, m * x3 + b, color='green')

plt.xlabel("Share of aided students", fontsize=20)
plt.ylabel("Graduation rate within 8 years", fontsize=20)
plt.savefig("comp8.png")

plt.show()
#-----------------------------------------------
#-----------------------------------------------
# Plot data (y='completion_rate_6yr')
plt.figure(figsize=(10,10))
ax = plt.gca()

data[data['year']==2016].plot(x='dependent_income_mean', y='completion_rate_6yr', kind='scatter', color='purple',
                              legend=True, label='Dependent students',ax=ax)
data[data['year']==2016].plot(x='independent_income_mean', y='completion_rate_6yr', kind='scatter', color='orange',
                              legend=True, label='Independent students',ax=ax)

x1=data[data['year']==2016].dependent_income_mean
y=data[data['year']==2016].completion_rate_6yr
stats = linregress(x1, y)
m = stats.slope
b = stats.intercept
plt.scatter(x1, y)
plt.plot(x1, m * x1 + b, color='purple')

x2=data[data['year']==2016].independent_income_mean
y=data[data['year']==2016].completion_rate_6yr
stats = linregress(x2, y)
m = stats.slope
b = stats.intercept
plt.scatter(x2, y)
plt.plot(x2, m * x2 + b, color='orange')

plt.xlabel("Average family income", fontsize=20)
plt.ylabel("Graduation rate within 6 years", fontsize=20)
plt.savefig("family.png")

plt.show()

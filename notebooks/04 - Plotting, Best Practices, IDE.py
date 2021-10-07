# Databricks notebook source
# MAGIC %md ## Agenda
# MAGIC What are we going to go through today:
# MAGIC - Plotting
# MAGIC   - How to make prettier plots than in pure matplotlib
# MAGIC - Data Science workflow
# MAGIC   - Pandas profiling
# MAGIC   - QA
# MAGIC   - setting up development environment
# MAGIC - Working with IDE
# MAGIC   - Best practices
# MAGIC - Git

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# COMMAND ----------

# MAGIC %md ## Heatmap

# COMMAND ----------

# Importing the titanic dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# heat_map = sns.heatmap(df_titanic_data, cmap="YlGnBu", mask=df_titanic_data.isnull())

df_titanic_data = pd.read_csv(r'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# COMMAND ----------

df_titanic_data.head()

# COMMAND ----------

# Heatmap is useful for looking at missing values
missing_vals = sns.heatmap(df_titanic_data.isnull(), cbar = False)
display(missing_vals)

# COMMAND ----------

df_titanic_data = df_titanic_data.sort_values(by = ['Survived'])

missing_vals = sns.heatmap(df_titanic_data.isnull(), cbar = False)
display(missing_vals)

# COMMAND ----------

# MAGIC %md ## Correlation matrix

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://www.mathsisfun.com/data/images/correlation-examples.svg'>

# COMMAND ----------

corr = df_titanic_data.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

display(sns.heatmap(corr, center=0, mask=mask, annot=True, cmap="YlGnBu"))

# COMMAND ----------

# MAGIC %md ## Plotting

# COMMAND ----------

data = pd.read_csv('https://raw.githubusercontent.com/FBosler/AdvancedPlotting/master/combined_set.csv')

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Columns explained
# MAGIC 
# MAGIC - Year: The year of measurement (from 2007 to 2018)
# MAGIC - Life Ladder: respondents measure of the value their lives today on a 0 to 10 scale (10 best) based on Cantril ladder
# MAGIC - Log GDP per capita: GDP per capita is in terms of Purchasing Power Parity (PPP) adjusted to constant 2011 international dollars, taken from the World Development Indicators (WDI) released by the World Bank on November 14, 2018
# MAGIC - Social support: Answer to question: “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”
# MAGIC - Healthy life expectancy at birth: Life expectancy at birth is constructed based on data from the World Health Organization (WHO) Global Health Observatory data repository, with data available for 2005, 2010, 2015, and 2016.
# MAGIC - Freedom to make life choices: Answer to question: “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”
# MAGIC - Generosity: Responses to “Have you donated money to a charity in the past month?” compared to GDP per capita
# MAGIC - Perceptions of corruption: Answer to “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?”
# MAGIC - Positive affect: comprises the average frequency of happiness, laughter, and enjoyment on the previous day.
# MAGIC - Negative affect: comprises the average frequency of worry, sadness, and anger on the previous day.
# MAGIC - Confidence in national government: Self-explanatory
# MAGIC - Democratic Quality: how democratic is a country
# MAGIC - Delivery Quality: How well a country delivers on its policies
# MAGIC - Gapminder Life Expectancy: Life expectancy from Gapminder
# MAGIC - Gapminder Population: Population of a country

# COMMAND ----------

# MAGIC %md ## Seaborn

# COMMAND ----------

# We can set the size of the plot
sns.set(rc={'figure.figsize':(8,4.5)})

# COMMAND ----------

# Let's extract a smaller dataset

data_vector = data[data['ISO3'] == 'CZE'][['Year', 'Healthy life expectancy at birth']]
print(data_vector)

# COMMAND ----------

plot = sns.scatterplot(
  x = 'Year',
  y = 'Healthy life expectancy at birth',
  data = data[data['ISO3'] == 'CZE'][['Year', 'Healthy life expectancy at birth']]
)

display(plot)

# COMMAND ----------

# Set the plot size
sns.set(rc={'figure.figsize':(14,10)}, font_scale = 1.2)

# Set the plot style
sns.set_style("white")

neighbors = ['Austria', 'Czech Republic', 'Germany', 'Poland', 'Slovak Republic']

sns.scatterplot(
  x = 'Year',
  y = 'Healthy life expectancy at birth',
  data = data[data['Country name'].isin(neighbors)],
  hue = 'Country name',
  size = 'Gapminder Population'
)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
display()

# COMMAND ----------

# Use default styles
sns.reset_orig()
sns.set_context("paper")
# sns.set(font_scale = 1.5)


neighbors = ['Austria', 'Czech Republic', 'Germany', 'Poland', 'Slovak Republic']

sns.scatterplot(
  x = 'Year',
  y = 'Healthy life expectancy at birth',
  data = data[data['Country name'].isin(neighbors)],
  hue = 'Country name',
  size = 'Gapminder Population'
)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
display()

# COMMAND ----------

sns.scatterplot(
  x='Log GDP per capita',
  y='Life Ladder',
  data=data[data['Year'] == 2018],    
  hue='Continent',
  size='Gapminder Population'
)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
display()

# COMMAND ----------

sns_scatter = sns.scatterplot(
  x='Log GDP per capita',
  y='Life Ladder',
  data=data[data['Year'] == 2018],    
  hue='Continent',
  size='Gapminder Population',
  sizes = (100, 500),
  alpha = 0.75,
  palette = 'muted'
)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
display()

# COMMAND ----------

sns.scatterplot(
  x='Log GDP per capita',
  y='Life Ladder',
  data=data[data['Year'] == 2018],    
  hue='Continent',
  size='Gapminder Population',
  sizes = (100, 500),
  alpha = 0.75,
  palette = 'muted'
)

sns.despine(offset=1, trim=True);

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0.)
display()

# COMMAND ----------

# MAGIC %md ## Plotly

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

fig = px.scatter(
    data_frame=data[data['Year'] == 2018], 
    x="Log GDP per capita", 
    y="Life Ladder", 
    size="Gapminder Population", 
    color="Continent",
    hover_name="Country name",
    size_max=60
)
fig.show()

# COMMAND ----------

# df = px.data.gapminder()
fig = px.scatter(data, x="Log GDP per capita", y="Healthy life expectancy at birth", animation_frame="Year", animation_group="Country name",
           size="Gapminder Population", color="Continent", hover_name="Country name",
           log_x=False, size_max=55, range_x=[6.3,12], range_y=[25,90])

# fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()

# COMMAND ----------

df = px.data.gapminder().query("continent=='Europe'")

fig = px.line(df, x="year", y="lifeExp", color='country')
fig.show()

# COMMAND ----------

# MAGIC %md ## Data scientist workflow

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling

# COMMAND ----------

report = pandas_profiling.ProfileReport(data, title="Gapminder Profiling Report", explorative = True)
# report.to_file("gapminder_profiling_report.html")

# COMMAND ----------

displayHTML(report.to_html())

# COMMAND ----------

# MAGIC %md ## QA

# COMMAND ----------

# The native function to check the result/status of something is assert in Python - best used for rudimentary testing

# Let's imagine this function to sum two non-negative integers
def sum_integers(int1, int2):
  return int1 + int2

# COMMAND ----------

def sum_integers(int1, int2):
  if int1 < 0:
    raise Exception('The value of int1 has to be > 0. The value of int1 was {}.'.format(int1))
  elif int2 < 0:
    raise Exception('The value of int2 has to be > 0. The value of int2 was {}.'.format(int2))
  return int1 + int2

sum_integers(-1,1)

# COMMAND ----------

# Let's see how we can use try, except, else
def check_age(age, age_limit):
  if age > 0 and type(age) is int:
    if age > age_limit:
      return True
    else:
      return False
  else:
    raise Exception()

    
def sell_drain_cleaner(age, age_limit):
  try:
    is_old_enough = check_age(age, age_limit)
  except:
    print('The age value entered is incorrect.')
  else:
#     if is_old_enough:
#       print('We can sell the drain cleaner.')
#     else:
#       print('Sorry, we can\'t sell this to you.')
    
    return is_old_enough
    
sell_drain_cleaner(27, 18)

# COMMAND ----------

# The try statement is often used with I/O
try:
  data = pd.read_csv('https://raw.githubusercontent.com/FBosler/AdvancedPlotting/master/combined_set.csv')
  print('Data loaded succesfully!')
except:
  print('The data can\'t be loaded.')

# COMMAND ----------

# How to test our function works?
def test_sell_drain_cleaner():
  assert sell_drain_cleaner(17,18) == False, 'We shouldn\'t sell drain cleaner to underage customers.'
  assert sell_drain_cleaner(22,18) == True, 'We can sell drain cleaner to adult customers.'
  
test_sell_drain_cleaner()

# COMMAND ----------

# MAGIC %md ## PEP8

# COMMAND ----------

# Writing Modular and Non-Repetitive Code
def add_2(x):
  return x+2

def add_3(x):
  return x+3

# ---
def add(x,y):
  return x+y

add(2,2)

# COMMAND ----------

# PEP8

# Indentation
for x in [1,2,3,4,5]:
    print(x)

# COMMAND ----------

# Line length
new_df = df_titanic_data[df_titanic_data['Sex'] == 'female'].groupby(['Survived', 'Embarked']).agg(['mean', 'count']).rename(columns={'Pclass': 'PassClass'})[['PassengerId', 'PassClass', 'Fare']].dropna(how='all', axis=1)

print(new_df)

# COMMAND ----------

# Blank lines
def sum_integers(int1, int2):
  return int1 + int2


def divide_integers(int1, int2):
  return int1 / int2

# COMMAND ----------

# Whitespaces
def sum_integers(int1, int2):
  return int1 + int2

def divide_integers(int1, int2):
  return int1 / int2

# COMMAND ----------

# Importing
import os
import io
import pathlib
import math
import time
import pandas as pd
import numpy as np

# COMMAND ----------

# Naming conventions

def functionName(VARIABLE, Another_Variable):
  return Result

def function_name(variable, another_variable):
  return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### The necessity of Documentation:

# COMMAND ----------

# Using DocString
def find_minimum(num_list):
  """
  This function is used to find the minimal number from a given list
  It takes a list of numbers as argument
  It returns the minimum number from the list
  """
return np.min(num_list)

# COMMAND ----------

# First step

# Second step

# Third step


# COMMAND ----------

# TODO: Add condition for when val is None

# COMMAND ----------

# DEMO IN NUMPY
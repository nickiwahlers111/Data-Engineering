#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv("books.csv")
df.head(10) 


# In[2]:


#OUTLINE GENEROUSLY PROVIDED BY RYAN FILGAS :)

# A. Filtering 
# Use python and pandas to filter this data by dropping these columns: 
# Edition Statement, Corporate Author, Corporate Contributors, Former owner, 
# Engraver, Issuance type, Shelfmarks

# Do this two ways. First use the DataFrame drop() method. Then do the same with the usecols 
# argument of pandas.read_csv()

#Hint: are you dropping rows or columns? Is there an argument for that in the drop method? 
#What types of values does the usecols argument expect?

#TWO WAYS OF FILTERING:
df2 = df.drop(columns = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner', 
              'Engraver', 'Issuance type', 'Shelfmarks'])
# df2.head()

df3 = pd.read_csv("books.csv", usecols=['Identifier', 'Place of Publication', 'Date of Publication', 'Publisher', 'Title', 'Author', 'Contributors', 'Flickr URL' ])

df3.head()



# In[48]:


#B. Tidying Up the Data
# In the book data, notice that the “Date of Publication” column has many inconsistencies. 
# Update all of the data in this column to be consistent four digit year values. Specifically, 

# X Remove the extra dates in square brackets, wherever present: e.g., 1879 [1878] should be converted to 1879
# X Remove uncertain dates and replace them with NumPy’s NaN: [1897?]
# X Convert date ranges to their “start date”: e.g., 1860-63; 1839, 38-54


def brackets(text):
    if type(text) != str:
        return text 
    else:
        val = re.sub(" \[\d\d\d\d\]", "", str(text))
        val = re.sub("\[","", str(val))
        val = re.sub("\]", "", str(val))
        if "?" in val:
            return np.NaN
        
        val =re.sub("\D\w*", "", str(val))
        return val
        
    
df_dates = df3["Date of Publication"]

res = df_dates.apply(brackets)
# res.head(30)



# X Convert the string nan to NumPy’s NaN value
# -Finally, update the type of the “Date of Publication” column to be numeric (not string, not object)
# -The “Place of Publication” column of this data set is also untidy. Transform all of the values in this 
# -column to be only the name of the city. If the city name is not found in the string, then the name of the country. If neither are present then transform to the string “unknown”.



# In[4]:


# C. Tidying with applymap()
# See this list of USA towns that have universities: uniplaces.txt. This data was originally created 
# for another purpose and contains artifacts of that. For example it is alphabetized by the name of 
# the state where the university is located. The state is listed once and then the universities present 
# in that state are listed on the lines below. Additionally there are extra punctuation marks to designate 
# separation of the town and the university name (), and an artifact number at the end [2].  We would like 
# to change the data to have 3 columns containing the state, city, and university.

# Task:
# Use the applymap() method to apply a custom function to the data. This should transform it into a tidy 
# list of city, town, and university. Details below.

# Task Details
# There are a lot of examples you can find for how to use applymap(). This example from Geeks4geeks 
# uses a lambda function. A lambda function in python is a simple function that can be accomplished in 
# one line. The method applymap() then applies that function to each row of the dataframe. Therefore 
# df.applymap(lambda x: len(str(x))) will operate on the dataframe called df. For each element of each 
# row of df, it applies the lambda function, naming the element x. This lambda function finds the length of x. 

# In python you can pass functions as arguments to another function. In this example below, you can choose #
# to emphasize your text by either shouting it or whispering it, depending on which function you pass to 
# #emphasize().

# >>> def shout(text):
# ... 	return text.upper()
# ...
# >>> def whisper(text):
# ... 	return text.lower()
# ...
# >>> def emphasize(myfunc, s):
# ... 	if s[0:5] == 'Hello':
# ...     	     	return myfunc(s[0:5]) + s[5:]
# ... 	return s
# ...
# >>> emphasize(shout, "Hello World!")
# 'HELLO World!'
# >>> emphasize(whisper, "Hello World!")
# 'hello World!'

# In our applymap() example above, we applied a lambda function to each row of the dataframe. 
# Instead, you can apply your own custom function.

# The method applymap() takes a function as input and applies it to the dataframe it is called on. 
# Write a custom function which handles the uniplaces.txt data and reformats it as 3 columns for state, 
# city, university.

# Hint: first create a dataframe from this data, with the desired columns. Then use applymap to clean out 
# 'the extra artifacts


# In[5]:


# D. Decoding
# Similar to C-Tran, TriMet also produces breadcrumb data for its buses. Here is a sample for one 
# bus on one day of October 2021: link to breadcrumb data

# One column of the TriMet breadcrumb data is called “OCCURRENCES”. Our contact at TriMet explained 
# this field as follows:

# OCCURRENCES – number of times a point appeared in the dataset. This is to clean up some of the data 
# because sometimes when the vehicle is stationary it will replicate multiple instances at the same point. 
# This consolidates those into a single record.

# This encoding of multiple breadcrumbs into a single record helps to save space, but for analysis we 
# typically need to decode it so that all of the records can be analyzed. Often decoding consists of 
# exploding one row out into multiple rows.

# Your job is to decode records with OCCURRENCES > 1 into replicated records in a DataFrame. So for example, 
# a sequence of records like this: 

# 4313660399,03411,B,29OCT2021:08:36:17,29OCT2021:00:00:00,30977,-122.844715,45.503493,0,223428.48,8,12,0.7,1,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30982,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660401,03411,B,29OCT2021:08:36:57,29OCT2021:00:00:00,31017,-122.844858,45.503212,5,223533.47,10,10,1.3,2,Y,TRANS,31OCT2021:06:06:40


# Should be expanded to a sequence of records like this:

# 4313660399,03411,B,29OCT2021:08:36:17,29OCT2021:00:00:00,30977,-122.844715,45.503493,0,223428.48,8,12,0.7,1,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30982,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30987,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30992,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30997,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,31002,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30907,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660400,03411,B,29OCT2021:08:36:22,29OCT2021:00:00:00,30912,-122.8448,45.503335,32,223487.54,8,11,0.7,9,Y,TRANS,31OCT2021:06:06:40
# 4313660401,03411,B,29OCT2021:08:36:57,29OCT2021:00:00:00,31017,-122.844858,45.503212,5,223533.47,10,10,1.3,2,Y,TRANS,31OCT2021:06:06:40
# 4313660401,03411,B,29OCT2021:08:36:57,29OCT2021:00:00:00,31022,-122.844858,45.503212,5,223533.47,10,10,1.3,2,Y,TRANS,31OCT2021:06:06:40
 

# This is because the second breadcrumb in the example (4313660400) has an OCCURRENCES value of 9. Note that 
# for this exercise it is OK to duplicate the 3VEH13660400

# After you have expanded out the multiple rows, be sure to clean up the dataframe if necessary. It should 
# have the same number of columns that you started with, in the same order, and they should all be named the 
# same as when we started.

# Hint: How can you decode a row into multiple rows in pandas? While it may be tempting to try to iterate 
# through the dataframe and append new rows, instead consider table-level pandas methods that you can use. 
# If any DataFrame methods you want to use are not available on a Series, is there an equivalent method 
# for the Series?


# In[6]:


# E. Filling
# The TriMet data, linked above, is missing some values in the VALID_FLAG column. Use the 
# pandas.DataFrame.ffill() method to fill in the missing data. 

# Hint: How can you check for bad data like NaN or duplicates in a DataFrame? How can you 
# find all the unique values in a column? For a column named like VALID_FLAG, what do you 
# think are the expected values?


# In[7]:


# F. Interpolating
# The TriMet breadcrumb data, linked above, is missing some values in the ARRIVE_TIME column. 
# Use the pandas.DataFrame.interpolate() method to fill in the missing time data. The interpolate 
# method fills in NAN values in a pandas DataFrame or Series. There are many different methods of 
# interpolation that you can specify for different use cases. Be sure to use the ‘linear’ interpolation 
# method which fills in the value based on previous values, ignoring the index, and equally spacing the 
# missing values.

# Hint: What is the frequency of the bus datapoints? Do we expect them every minute, every few seconds, 
# etc? Does interpolate achieve this automatically? If not, how can you adjust it to do so?

# Could you have used the interpolate() method for problem E above?


# In[8]:


# G. Melt
# This activity will be filled in for Wednesday.


# In[9]:


# H. Merge
# This activity will be filled in for Wednesday.


# In[10]:


# I. Transformation Visualizations
# This activity will be filled in for Wednesday.


# In[ ]:





# In[ ]:






#Real-time Twitter sentiment Analysis for Brand improvement and Topic Tracking

# Twitter Sentiment Analysis and Interactive Data Visualization using 
# RE(Regular Expressions), Textblob, NLTK(Natural Language Toolkit), and Plotly

# Load data from MySQL to perform exploratory data analysis


#Importing the libraries which will be used for the 
#Processing and inference generation program

from nltk import collocations
import settings
import sys
import mysql.connector 
import pandas as pd
import time
import itertools
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#%matplotlib inline
import plotly.express as px
import datetime
from IPython.display import clear_output

import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
py.init_notebook_mode()

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

print(sys.executable)

# Filter constants for states in US
#Set up all state names and their abbreviations as constants for further abbreviation-name transformation.
STATES = ['Alabama','AL', 'Alaska', 'AK', 'American Samoa', 'AS','Arizona', 'AZ', 'Arkansas', 'AR','California', 'CA', 'Colorado', 'CO',
'Connecticut', 'CT', 'Delaware', 'DE', 'District of Columbia', 'DC','Federated States of Micronesia', 'FM', 'Florida', 'FL',
'Georgia','GA','Guam','GU','Hawaii', 'HI', 'Idaho', 'ID','Illinois','IL','Indiana','IN','Iowa', 'IA','Kansas','KS','Kentucky','KY',
'Lousiana','LA', 'Maine', 'ME', 'Marshall Islands','MH','Maryland','MD','Massachusetts','MA','Michigan','MI','Minnesota','MN',
'Mississippi','MS','Missouri','MO','Montana','MT','Nebraska','NE','Nevada','NV','New Hampshire','NH','New Jersey','NJ','New Mexico','NM',
'New York','NY','North Carolina','NC','North Dakota','ND','Northern Mariana Islands','MP','Ohio','OH','Oklahoma','OK','Oregon','OR',
'Palau','PW','Pennsylvania','PA','Puerto Rico','PR','Rhode Island','RI','South Carolina','SC','South Dakota','SD','Tennessee','TN',
'Texas','TX','Utah','UT','Vermont','VT','Virgin Islands','VI','Virginia','VA','Washington','WA','West Virginia','WV','Wisconsin','WI',
'Wyoming','WY']

STATE_DICT = dict(itertools.zip_longest(*[iter(STATES)]*2,fillvalue=""))
INV_STATE_DICT = dict((v,k) for k,v in STATE_DICT.items())


# Complex plot to show the latest twitter data within 20 minutes with automatic updation

while True:
    clear_output()
    db_connection = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "password",
        database = "TwitterDB",
        charset = 'utf8'
    )

    #load data from MySQL
    timenow = (datetime.datetime.utcnow() - datetime.timedelta(hours =0,minutes = 20)).strftime('%Y-%m-%d %H:%M:%S')
    #used to get the time 20 minutes prior to the current time
    #converting the date and time object to string representation using strf function

    query = "SELECT id_str, text, created_at, polarity, user_location FROM \
            {} WHERE created_at >= '{}'".format(settings.TABLE_NAME, timenow)
    #This is the SQL query which will be executed to extract the relevant information from the tweets which have been posted not more than 20 minutes back
    #This condition is specified using the where condition in the above MySQL query
    # We extract the id of the account, time of creation, polarity and user_location from the database to plot our required figures

    df = pd.read_sql(query, con = db_connection)
    #this extracts the data to the pandas dataframe 

    # df = pd.read_csv("sample_data.csv") 
    # for testing purposes 


    #UTC for date time at default
    #converting the entry from string to datetime data type using the following command
    df['created_at'] = pd.to_datetime(df['created_at'])

    #making the big figure which will have the 4 subplots

    fig = make_subplots(rows =2, cols =2, column_widths = [1, 0.4], row_heights = [0.6, 0.4],
        specs = [[{"type":"scatter","rowspan":2},{"type":"chloropleth"}],
        [ None, {"type":"bar"}]])


    
    
    #Plotting the Line Chart which represents the sentiments - positive, negative and neutral towards the selected topic/brand(specified in the settings.py file)

    #Cleaning and transforming the data to enable time series 
    # Convert the entire time serises into groups of 2 seconds, and count the number of sentiment for each kind of polarities 
    # (e.g. -1, 0, and 1) in each time-interval group.
    # Apply unstack-stack technology to make sure all categories in each group are displayed even if one of categories doesn't 
    # have any value. As we only display real-time 
    # tweets posted in last 30 minutes, groups of 2-second interval could best display on the screen in practice. 
    
    
    result = df.groupby([pd.Grouper(key = 'created_at',freq = '2s'), 'polarity']).count().unstack(fill_value = 0).stack().reset_index()

    # After that, rename the columns to allow them self explanatory.
    result = result.rename(columns = {"id_str":"Num of {} mentions".format(settings.TRACK_WORDS[0]),"created_at":"Time in UTC"})

    #Record the time series with 2-second interval for further index usage.

    time_series = result["Time in UTC"][result['polarity']==0].reset_index(drop = True)

    #Add three Lines of negatives, neutrals, and positives in the first subplot using add_trace and go.Scatter . 
    #In addition, row and col represent the place of this subplot in the big figure.
    #1.Adding the Neutral line plot
    fig.add_trace(go.Scatter(x = time_series, y = result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==0].reset_index(drop = True), name = "Nuetral", opacity= 0.8), row = 1, col = 1)
    #2.Adding the Positive line plot
    fig.add_trace(go.Scatter(x = time_series, y = result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==1].reset_index(drop = True),name = "Positive", opacity = 0.8), row = 1, col = 1)
    #3.Adding the Negative line plot
    fig.add_trace(go.Scatter(x = time_series, y = result["Num of '{}' mentions".format(settings.TRACK_WORDS[0])][result['polarity']==-1].reset_index(drop = True),name = "Negative", opacity = 0.8), row = 1, col = 1)




    #Plotting the Bar Chart
    #This part will use the NLTK(Natural Language Toolkit) to track the hottest words or the most frequency tokens in the tweets
    #In order to achieve the above functionality, we join all tweet texts, remove the URLs, clean the 'RT' and '&'(a.k.a '&amp;') symbols
    # and convert all characters into lowercases

    #data preprocessing
    content = ' '.join(df["text"]) #joins all the text of the tweets as a single string
    content = re.sub(r"http\S+","", content) #removing URLs
    content = content.replace('RT ', ' ').replace('&amp;','and') # replace the and words with the word and
    content = re.sub('[^A-Za-z0-9]+',' ',content) #python removes anything that is not a letter or a number
    content = content.lower() # makes the entire string lowercase

    #Punkt Sentence Tokenizer is used to divide a text into a list of sentences by using an unsupervised algorithm.
    #Then tokenize the entire text from all tweets, use Stop Words to remove commonly used words, 
    #and extract 10 most common words in the Frequency Distribution of all words.
    
    tokenized_word = word_tokenize(content) # gets the words from the long string, stores in tokenized_word as a list
    stop_words = set(stopwords.words("english")) #getting the stop words in english language, so that they can be removed from the word list
    filtered_sent = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    fdist = FreqDist(filtered_sent) #obtaining the frequency distribution
    fd = pd.DataFrame(fdist.most_common(10), columns = ["Word", "Frequency"]).drop([0]).reindex()
    # getting a pandas dataframe which shows the 10 most common words with columns : word and frequency

    #Plotting the bar chart on the main figure, in the place row =2, column = 2
    #Add the Bar Chart for the topic frequency distribution using add_trace and go.Bar . 
    # We use rgb(xx,xx,xx) or rgba(xx,xx,xx,x) to change the color of certain elements in the figure.
    fig.add_trace(go.Bar(x = fd["Word"], y= fd["Frequency"], name = "Freq Dist"), row =2 , col =2)
    fig.update_traces(marker_color = 'rgb(59, 89, 152)', marker_line_color = 'rgb(8,48,107)',marker_line_width=0.5, opacity = 0.7, row = 2, col =2)



    #Geographic Segmentation Recognition with Text Processing
    #Plotted as a geo distribution

    #Data pre-processing to extract location information:
    #To explore users’ geographic distributions, 
    # we need to identify their locations through their user profile rather than the locations attached with tweets, since less than 1% people will attach their tweet locations. 
    # However, according to locations in users’ profile, they may include one or more of counties, cities, states, countries, or planets. 
    # Thus, filtering these data into US state-level locations is the core of geographic segmentation recognization.
    
    #Extracting the state information from their locations by iterating the list of state names and the list of user locations.
    is_in_US = []
    geo = df[['user_location']]
    df = df.fillna(" ")
    for x in df['user_location']:
        check = False
        for s in STATES:
            if s in x:
                is_in_US.append(STATE_DICT[s] if s in STATE_DICT else s)
                check = True
                break
            if not check:
                is_in_US.append(None)
    geo_dist = pd.DataFrame(is_in_US, columns =['State']).dropna().reset_index()
    
    #Count the number of tweets posted in each state of US, 
    # and use logarithmic number to avoid the extreme values (e.g. 500+ in California and 3 in North Dakota) for better visualization.
    geo_dist = geo_dist.groupby('State').count().rename(columns = {"index":"Number"}).sort_values(by = ['Number'],ascending= False).reset_index()
    geo_dist["Log Num"] = geo_dist["Number"].apply(lambda x : math.log(x,2))
    
    #Adding explanatory text information for the hover text on later dashboard.
    geo_dist['Full State Name'] = geo_dist['State'].apply(lambda x: INV_STATE_DICT[x])
    geo_dist['text'] = geo_dist['Full State Name'] + '<br>' + 'Num: ' + geo_dist['Number'].astype(str)

    #Then insert the Map in the top right, and set locations and numbers per location.
    fig.add_trace(go.Choropleth(locations = geo_dist['State'], z = geo_dist['Log Num'].astype(float), locationmode = 'USA-states', colorscale = "Blues",text = geo_dist['text'], showscale = False, geo = 'geo' ), row =1 , col =2)

    #Add title in the layout of our figure, 
    # reduce the geo-scope of our map, 
    # turn the template theme into the dark,
    # add annotations for the layout using go.layout.Annotation.
    fig.update_layout(title_text = "Real-time tracking '{}' mentions on Twitter {} UTC".format(settings.TRACK_WORDS[0],datetime.date.utcnow().strftime('%m-%d %H:%M')), geo = dict(scope = 'usa'),template = "plotly_dark", margin = dict(r = 20, t = 50, b = 50, l = 20),annotations = [go.layout.Annotation(text = "Source: Twitter",showarrow = False,xref = "paper", yref = "paper", x =0, y =0)], showlegend = False, xaxis_rangeslider_visible = True)

    #Last, display all subplots in the single figure.
    fig.show()


    time.sleep(60)







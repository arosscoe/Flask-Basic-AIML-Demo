import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tweepy
import csv
import os
import pandas as pd

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import seaborn as sns
import re
import spacy
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
from statistics import mean
import re
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler





def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            file = request.files['file']
            if not os.path.isdir('flask_csv'):
                os.mkdir('flask_csv')
            filepath = os.path.join('flask_csv', 'User_upload.csv')
            file.save(filepath)

            tweets = pd.read_csv(filepath)

            text_df = tweets.drop(['Unnamed: 0', 'Date Created', 'Favourite Count',
            'Number of Likes', 'Retweeet Count', 'Reply Count', 'Retweeted Tweet',
            'Language'], axis=1)

            def clean_text(df, text_field):
                df[text_field] = df[text_field].str.lower()
                df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
                return df

            clean_tweets = clean_text(text_df, 'Tweet')

            nlp = en_core_web_sm.load()
            tokenizer = RegexpTokenizer(r'\w+')
            lemmatizer = WordNetLemmatizer()
            stop = set(stopwords.words('english'))
            punctuation = list(string.punctuation) #already taken care of with the cleaning function.
            stop.update(punctuation)
            w_tokenizer = WhitespaceTokenizer()

            
            def furnished(text):
                final_text = []
                for i in w_tokenizer.tokenize(text):
            #     for i in text.split():
                    if i.lower() not in stop:
                        word = lemmatizer.lemmatize(i)
                        final_text.append(word.lower())
                return " ".join(final_text)

            clean_tweets.Tweet = clean_tweets['Tweet'].apply(furnished)

            economy_related_words = '''agriculture infrastructure capitalism trading service sector technology  economical supply 
                            industrialism efficiency frugality retrenchment downsizing   credit debit value 
                            economize   save  economically
                            economies sluggish rise   rising spending conserve trend 
                            low-management  decline   industry impact poor  
                                profession    surplus   fall
                            declining  accelerating interest sectors balance stability productivity increase rates
                                pushing expanding stabilize  rate industrial borrowing struggling
                            deficit predicted    increasing  data
                            economizer analysts investment market-based economy   debt free enterprise
                            medium  exchange metric savepoint scarcity capital bank company stockholder fund business  
                            asset treasury tourism incomes contraction employment jobs upturn deflation  macroeconomics
                            bankruptcies exporters hyperinflation dollar entrepreneurship upswing marketplace commerce devaluation 
                            quicksave deindustrialization stockmarket reflation downspin dollarization withholder bankroll venture capital
                            mutual fund plan economy mortgage lender unemployment rate credit crunch central bank financial institution
                            bank rate custom duties mass-production black-market developing-countries developing economic-growth gdp trade barter 
                            distribution downturn economist'''
        
            social_related_words = '''sociable, gregarious societal friendly society socialization political  sociality 
                            interpersonal  ethnic socially party welfare public community socialist societies development
                                network humans socialism collective personal corporation social constructivism
                            relations volition citizenship brute   attitude rights socio 
                            socioeconomic ethics civic communal marital  sociale socialized communities     
                            policy   unions        
                            institutions values     governmental   organizations jamboree 
                            festivity    fairness  support  care  
                            sides   activism     unsocial psychosocial 
                            socializing psychological distributional  demographic  participation reunion 
                            partygoer partyism festive power network gala housewarming celebration counterparty   social-war
                            particularist interactional ideational asocial'''
    
            culture_related_words  = ''' ethnicity heritage modernity spirituality marxismmaterial culture 
                            ethos nationality humanism romanticism civilisation traditionalism genetics
                            kinship heredity marriage   indigenous  archeology  acculturate  
                        ontogenesis viniculture modern clothes     rooted 
                        cicero societies history roots influence geography historical folk origins 
                        phenomenon teleology ancient aspects perspective liberalism nowadays community style unique prevalent describes 
                            today  origin   modernity beliefs  genre barbarian ethnic 
                        colonization cultural universal organization western-civilization structuralism  culture 
                        heathen pagan transculturation culture peasant classicist nativism anarchy ungrown philosophic cult  
                        consciousness islamist bro-culture evolve cultic diaspora aftergrowth native cultural-relativism  
                        mongolian cosmopolitan epistemology lifestyles diversity chauvinism westernization materialism vernacular 
                        homogeneity otherness holism tusculanae disputationes primitivism superficiality hedonism discourse
                        puritanism modernism intellectualism  exclusiveness elitism  colonialism  
                        pentecostalism paganism nationwide expansion rural  auxesis kimono 
                        culturize alethophobia nettlebed japanification  dongyi clannishness insularity hybridity
                        westernisation foreignness worldview exclusionism enculturation ethnocentrism  confucianist vulgarization
                        shintoism  westernism denominationalism    deracination
                            eurocentrism  cosmologies  emotiveness bohemianism territorialism
                        philosophical-doctrine ethnic minority social-darwinism  theory cultural evolution belief systemfolk music 
                        traditional art house karl-marx   theorymedia  
                        film-theory art history museum studies cultural artifact'''

            health_related_words = '''disease obesity world health organization medicine nutrition well-being exercise welfare wellness health care public health 
                        nursing stress safety hygiene research social healthy condition aids epidemiology healthiness wellbeing
                        care illness medical dieteducation infectious disease environmental healthcare physical fitness hospitals 
                        health care provider doctors healthy community design insurance sanitation human body patient mental health
                        medicare agriculture health science fitnesshealth policy  weight loss physical therapy psychology pharmacy
                        metabolic organism human lifestyle status unhealthy upbeat vaccination sleep condom alcohol smoking water family
                        eudaimonia eudaemonia air house prevention genetics public families poor needs treatment communicable disease 
                        study protection malaria development food priority management healthful mental provide department administration
                        programs help assistance funding environment improving emergency need program affected schools private mental illness 
                        treat diseases preparedness perinatal fertility sickness veterinary sanitary pharmacists behavioral midwives
                        gerontology infertility hospitalization midwifery cholesterol childcare pediatrician pediatrics medicaid asthma 
                        pensions sicknesses push-up physical education body-mass-index eat well gymnastic apparatus tune up good morning 
                        bathing low blood-pressure heart attack health club ride-bike you feel good eczema urticaria dermatitis sunburn overwork 
                        manufacturing medical sociology need exercise run'''
                            
            lp = en_core_web_sm.load()
            tokenizer = RegexpTokenizer(r'\w+')
            lemmatizer = WordNetLemmatizer()
            stop = set(stopwords.words('english'))
            punctuation = list(string.punctuation)
            stop.update(punctuation)
            w_tokenizer = WhitespaceTokenizer()
        
            # clean the set of words
                    
            def furnished(text):
                final_text = []
                for i in text.split():
                    if i.lower() not in stop:
                        word = lemmatizer.lemmatize(i)
                        final_text.append(word.lower())
                return " ".join(final_text)
        
            economy = furnished(economy_related_words)
            social = furnished(social_related_words)
            culture = furnished(culture_related_words)
            health = furnished(health_related_words)

            # delete duplicates
            string1 = economy
            words1 = string1.split()
            economy = " ".join(sorted(set(words1), key=words1.index))

            string2 = social
            words2 = string2.split()
            social = " ".join(sorted(set(words2), key=words2.index))

            string3 = culture
            words3 = string3.split()
            culture = " ".join(sorted(set(words3), key=words3.index))

            string4 = health
            words4 = string4.split()
            health = " ".join(sorted(set(words4), key=words4.index))

            #Vectorizing the sets of words, then standardizing them.

            def get_vectors(*strs):
                text = [t for t in strs]
                vectorizer = TfidfVectorizer(text)
                vectorizer.fit(text)
                return vectorizer.transform(text).toarray()
        
            ## Vectorizing the tweets
            tv=TfidfVectorizer()
            tfidf_tweets =tv.fit_transform(clean_tweets['Tweet'])

            #Jaccard similarity is good for cases where duplication does not matter, 


            def jaccard_similarity(query, document):
                intersection = set(query).intersection(set(document))
                union = set(query).union(set(document))
                return len(intersection)/len(union)


            def get_scores(group,tweets):
                scores = []
                for tweet in tweets:
                    s = jaccard_similarity(group, tweet)
                    scores.append(s)
                return scores
        
            e_scores = get_scores(economy, clean_tweets['Tweet'].to_list())
            s_scores = get_scores(social, clean_tweets['Tweet'].to_list())
            c_scores = get_scores(culture, clean_tweets['Tweet'].to_list())
            h_scores = get_scores(health, clean_tweets['Tweet'].to_list())

            data  = {'names':clean_tweets['User'].to_list(), 'economic_score':e_scores,
                'social_score': s_scores, 'culture_score':c_scores, 'health_scores':h_scores}
            scores_df = pd.DataFrame(data)

            def get_clusters(l1, l2, l3, l4):
                econ = []
                socio = []
                cul = []
                heal = []
                for i, j, k, l in zip(l1, l2, l3, l4):
                    m = max(i, j, k, l)
                    if m == i:
                        econ.append(1)
                    else:
                        econ.append(0)
                    if m == j:
                        socio.append(1)
                    else:
                        socio.append(0)        
                    if m == k:
                        cul.append(1)
                    else:
                        cul.append(0)  
                    if m == l:
                        heal.append(1)
                    else:
                        heal.append(0)   
                
                return econ, socio, cul, heal
        
            l1 = scores_df.economic_score.to_list()
            l2 = scores_df.social_score.to_list()
            l3 = scores_df.culture_score.to_list()
            l4 = scores_df.health_scores.to_list()

            econ, socio, cul, heal = get_clusters(l1, l2, l3, l4)

            data = {'names':clean_tweets['User'].to_list(), 'economic':econ, 'social':socio, 'culture':cul, 'health': heal}
            cluster_df = pd.DataFrame(data)

            a =  cluster_df[['economic', 'social', 'culture', 'health']].sum(axis = 1) > 1
            c = cluster_df[['economic', 'social', 'culture', 'health']].sum(axis = 1)
            b = cluster_df.copy()
            cluster_df.loc[(a), ['economic','social', 'culture', 'health']] = 1/c

            pivot_clusters = cluster_df.groupby(['names']).sum()
            pivot_clusters['economic'] = pivot_clusters['economic'].astype(int)
            pivot_clusters['social'] = pivot_clusters['social'].astype(int)
            pivot_clusters['culture'] = pivot_clusters['culture'].astype(int)
            pivot_clusters['health'] = pivot_clusters['health'].astype(int)
            pivot_clusters['total'] = pivot_clusters['health'] + pivot_clusters['culture'] + pivot_clusters['social'] +  pivot_clusters['economic']
            pivot_clusters.loc["Total"] = pivot_clusters.sum()  #add a totals row

            #pivot_clusters = pivot_clusters.tail().to_html()
            fig = plt.figure(figsize =(10, 7)) 
            a = pivot_clusters.drop(['total'], axis = 1)
            plt.pie(a.loc['Total'], labels = a.columns)
            plt.title('A pie chart showing the volumes of tweets under different categories.')
            imagepath1 = os.path.join('static', 'piechart' + '.png')
            plt.savefig(imagepath1)

            # Users with the most tweets in dataset

            d = pivot_clusters.sort_values(by = 'total', ascending  = False)
            e = d.head(12)
            e.drop(e.head(2).index, inplace=True)

            plt.figure(figsize=(12,10))
            sns.barplot(x = e.index, y = e.total)
            plt.title('Top tweets of users based on volume')
            plt.xticks(rotation=45)
            plt.xlabel('screen names')
            plt.ylabel('total tweets')
            imagepath2 = os.path.join('static', 'mostTweets' + '.png')
            plt.savefig(imagepath2)

            # Users with most economy tweets
            d = pivot_clusters.sort_values(by = 'economic', ascending  = False)
            e = d.head(11)
            e.drop(e.head(1).index, inplace=True)

            plt.figure(figsize=(12,10))
            sns.barplot(x = e.index, y = e.economic)
            plt.title('Top tweets from users based on volume of ECONOMIC Tweets')
            plt.xticks(rotation=45)
            plt.xlabel('screen names')
            plt.ylabel('economy tweets')
            imagepath3 = os.path.join('static', 'mostEconomy' + '.png')
            plt.savefig(imagepath3)

            # Users with most social tweets
            d = pivot_clusters.sort_values(by = 'social', ascending  = False)
            e = d.head(12)
            e.drop(e.head(2).index, inplace=True)


            plt.figure(figsize=(12,10))
            sns.barplot(x = e.index, y = e.social)
            plt.title('Top tweets from users based on volume of SOCIAL Tweets')
            plt.xticks(rotation=45)
            plt.xlabel('screen names')
            plt.ylabel('social tweets')
            imagepath4 = os.path.join('static', 'mostSocial' + '.png')
            plt.savefig(imagepath4)
            
            # Users with most culture tweets
            d = pivot_clusters.sort_values(by = 'culture', ascending  = False)
            e = d.head(11)
            e.drop(e.head(1).index, inplace=True)

            plt.figure(figsize=(12,10))
            sns.barplot(x = e.index, y = e.culture)
            plt.title('Top tweets from users based on volume of Culture Tweets')
            plt.xticks(rotation=45)
            plt.xlabel('screen names')
            plt.ylabel('culture tweets')
            imagepath5 = os.path.join('static', 'mostCulture' + '.png')
            plt.savefig(imagepath5)

            # Users with most health tweets
            d = pivot_clusters.sort_values(by = 'health', ascending  = False)
            e = d.head(12)
            e.drop(e.head(2).index, inplace=True)

            plt.figure(figsize=(12,10))
            sns.barplot(x = e.index, y = e.health)
            plt.title('Top tweets from users based on volume of HEALTH Tweets')
            plt.xticks(rotation=45)
            plt.xlabel('screen names')
            plt.ylabel('health tweets')
            imagepath6 = os.path.join('static', 'mostHealth' + '.png')
            plt.savefig(imagepath6)

            pivot_clusters.drop(pivot_clusters.tail(1).index,inplace=True)
            # Target variable
            sns.distplot(pivot_clusters.economic , fit=norm);

            # Get the fitted parameters used by the function
            (mu, sigma) = norm.fit(pivot_clusters.economic)
            print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

            #Plot the distribution
            plt.title('Ecocnomy tweets distribution plot')
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                        loc='best')
            plt.ylabel('Frequency')
            plt.title('y distribution')

            #Get also the QQ-plot
            fig = plt.figure()
            res = stats.probplot(pivot_clusters.economic, plot=plt)
            imagepath7 = os.path.join('static', 'EconomicDist' + '.png')
            plt.savefig(imagepath7)

            # Target variable
            sns.distplot(pivot_clusters.social , fit=norm);

            # Get the fitted parameters used by the function
            (mu, sigma) = norm.fit(pivot_clusters.social)
            print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

            #Now plot the distribution
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                        loc='best')
            plt.title('Social tweets distribution plot')
            plt.ylabel('Frequency')
            plt.title('y distribution')

            #Get also the QQ-plot
            fig = plt.figure()
            res = stats.probplot(pivot_clusters.social, plot=plt)
            imagepath8 = os.path.join('static', 'SocialDist' + '.png')
            plt.savefig(imagepath8)

            # Target variable
            sns.distplot(pivot_clusters.culture , fit=norm);

            # Get the fitted parameters used by the function
            (mu, sigma) = norm.fit(pivot_clusters.culture)
            print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

            #Now plot the distribution
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                        loc='best')
            plt.title('Culture tweets distribution plot')
            plt.ylabel('Frequency')
            plt.title('y distribution')

            #Get also the QQ-plot
            fig = plt.figure()
            res = stats.probplot(pivot_clusters.culture, plot=plt)
            imagepath9 = os.path.join('static', 'CultureDist' + '.png')
            plt.savefig(imagepath9)

            # Target variable
            sns.distplot(pivot_clusters.health , fit=norm);

            # Get the fitted parameters used by the function
            (mu, sigma) = norm.fit(pivot_clusters.health)
            print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

            #Now plot the distribution
            plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                        loc='best')
            plt.title('Health tweets distribution plot')
            plt.ylabel('Frequency')
            plt.title('y distribution')

            #Get also the QQ-plot
            fig = plt.figure()
            res = stats.probplot(pivot_clusters.health, plot=plt)
            imagepath10 = os.path.join('static', 'HealthDist' + '.png')
            plt.savefig(imagepath10)


            seg = pivot_clusters.copy()
            # seg = seg.drop
            pca = PCA()
            pca.fit(seg)
            pca = PCA(n_components = 2)
            pca.fit(seg)
            scores = pca.transform(seg)
            n = 4
            kmeans_pca = KMeans(n_clusters = n, init = 'k-means++', random_state = 0)
            kmeans_pca.fit(scores)
            c = pd.concat([seg.reset_index(drop = True), pd.DataFrame(scores)], axis = 1)
            c.columns.values[-2:] = ['component1', 'component2']
            c['segment_kmeans_pca'] = kmeans_pca.labels_
            plt.figure(figsize = (10,8))
            sns.scatterplot(x = c['component1'], y = c['component2'], hue = c['segment_kmeans_pca'], palette = ['g', 'r', 'b', 'y'])
            plt.title('Clusters by PCA')
            imagepath11 = os.path.join('static', 'ClusterByPCA' + '.png')
            plt.savefig(imagepath11)

            tweets = pd.read_csv(filepath)
            tweets['scores'] = tweets['Number of Likes'] + tweets["Retweeet Count"] + tweets['Reply Count']
            top_tweets = tweets.sort_values(by = 'scores', ascending = False)
            top_tweets['Date Created'] = pd.to_datetime(top_tweets['Date Created'])

            # Add columns with year, month, and weekday name
            top_tweets['Year'] = pd.DatetimeIndex(top_tweets['Date Created']).year
            top_tweets['Month'] = pd.DatetimeIndex(top_tweets['Date Created']).month
            top_tweets['Weekday'] = pd.DatetimeIndex(top_tweets['Date Created']).weekday

            # week day analysis
            weeks_df = top_tweets.groupby(['Weekday']).count() 
            weeks_df = weeks_df[['Tweet']]
            weeks_df.reset_index(inplace=True)
            dict_map = {0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'}
            weeks_df['Weekday'] = weeks_df['Weekday'].replace(dict_map)

            #sorter = ['0', '1', '2', '3', '4', '5', '6']
            #sorterIndex = dict(zip(sorter,range(len(sorter))))

            weeks_df['Day_id'] = weeks_df.index
            #weeks_df['Day_id'] = weeks_df['Day_id'].map(sorterIndex)
            weeks_df.sort_values('Day_id', inplace=True)

            # now lets see some of the busy days of the week.
            plt.figure(figsize=(12,10))
            sns.lineplot(x = weeks_df.Weekday, y = weeks_df.Tweet)
            plt.title('A lineplot plot showing tweets distribution across the weeks')
            plt.xticks(rotation=45)
            plt.xlabel('Day of the week')
            plt.ylabel('Number of Tweets')
            imagepath12 = os.path.join('static', 'WeekdayAnalysis' + '.png')
            plt.savefig(imagepath12, dpi=300)
            
            return render_template('returnHome.html')
            #return 'The file name of the uploaded file is: {}'.format(file.filename)
                
        return render_template('upload.html')

    @app.route('/graphs')
    def graphs():

        return render_template('image.html', image = '/static/piechart.png', image2 = '/static/mostTweets.png',
                               image3 = '/static/mostEconomy.png', image4 = '/static/mostSocial.png', image5 = '/static/mostCulture.png',
                               image6 = '/static/mostHealth.png', image7 = '/static/EconomicDist.png', image8 = '/static/SocialDist.png',
                               image9 = '/static/CultureDist.png', image10 = '/static/HealthDist.png', 
                               image11 = '/static/ClusterByPCA.png', image12 = '/static/WeekdayAnalysis.png')
                
    return app
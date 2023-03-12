#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install newspaper3k


# ### Summary Extraction using Newspaper3k

# In[2]:


from newspaper import Article
from newspaper import Config
import nltk
import pandas as pd
#The user agent has to be obtained from the https://www.whatismybrowser.com/detect/what-is-my-user-agent for running on your system!
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
config = Config()
config.browser_user_agent = user_agent


# In[3]:


url ="https://colombiacheck.com/index.php/chequeos/cifras-de-covid-19-en-vacunados-de-israel-son-usadas-para-desinformar-sobre-las-vacunas"


# In[4]:


news = Article(url,lang='es', config=config) 
news.download()
news.parse()
nltk.download('punkt')
news.nlp()
# Extract summary
print(news.summary)


# In[5]:


# from google.colab import drive 
# drive.mount('/content/drive')


# In[11]:


#Reading the training data
data = pd.read_csv('train.csv')


# In[12]:


#Creating summary for all the URLs
def fil(url):
    try:
        news = Article(url, config=config) 
        news.download()
        news.parse()
        news.nlp()
        return news.summary
    except:
        return ''
data['summary'] = data.apply(lambda x: fil(x['Fact-checked Article']),axis=1)


# In[13]:


#Saving the data as a csv
data.to_csv('results_config_autolangdet.csv', index= False)


# In[5]:


# !pip install -U sentence-transformers -q


# In[3]:


import pandas as pd
import numpy as np
np.random.seed(2022)


# In[4]:


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')


# In[4]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[6]:


# data = pd.read_csv(r"C:\Users\rmanoger\Desktop\CSE 472\Project 2\results_config_autolangdet.csv")
data = pd.read_csv("results_config_autolangdet.csv")


# In[7]:


data.loc[data["summary"].isna(),'summary']=' '


# In[8]:


data.head()


# In[9]:


# import spacy
# import string
# nlp = spacy.load("en_core_web_sm")
# stop_words = nlp.Defaults.stop_words
# print(stop_words)


# In[10]:


# punctuations = string.punctuation
# print(punctuations)


# In[11]:


# def spacy_tokenizer(sentence):
#     doc = nlp(sentence)
#     mytokens = [ word.lemma_.lower().strip() for word in doc ]
#     mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
#     sentence = " ".join(mytokens)
#     return sentence


# In[13]:


# data['tokenize_Claim'] = data['Claim'].apply(spacy_tokenizer)


# In[14]:


# data['tokenize_summary'] = data['summary'].apply(spacy_tokenizer)


# In[12]:


# data.head()


# In[15]:


# data['embeddings_Claim'] = data['tokenize_Claim'].apply(model.encode)


# In[16]:


# data['embeddings_summary'] = data['tokenize_summary'].apply(model.encode)


# In[112]:


#Mapping the redundant source names together 
Source_mapping ={'whataspp':'social media','website':'websites','social networks':'social media','lord sumption':'person',
'Social Networking Media Users':'social media', 'twitter':'social media','donald trump':'person','joe biden':'person',
'pereson':'person','perseon':'person','narendra modi':'person','news website':'news media','mutliple sources':'multiple sources',
'multiplie sources':'multiple sources', 'multipe sources':'multiple sources', 'Multiple sources':'multiple sources',
'multple sources':'multiple sources', 'multiple persons':'person', 'multiple people':'person',
'government':'government department','government party':'government department','government department':'others','political party':'others',
'government source':'government department','facbook':'social media','anthony fauci':'person', 'bernie sanders':'person', 'beverley turner':'person', 'boris johnson':'person', 'carlos bolsonaro':'person', 'eduardo pazuello':'person', 'emmanuel macron':'person', 'flavio bolsonaro':'person', 'flÃivio bolsonaro':'person', 'Harry Roque':'person', 'ione belarra':'person', 'jair bolsonaro':'person', 'michael yeedon':'person', 'tayyip erdogan':'person', 'sophia dorinskaya':'person', 'rodrigo duterte':'person', 
'nick donnelly':'person', 'social media ':'social media','youtube':'social media','reddit':'social media','instagram':'social media','meme':'social media',
'email':'social media','tiktok':'social media','telegram':'social media','whatsapp':'social media','facebook':'social media','Several sources':'multiple sources', 
'Jair Bolsonaro':'person','joão doria':'person','andrés manuel lópez obrador':'person','michał dworczyk':'person','michael "mike" defensor':'person','steven hotze':'person','flávio bolsonaro':'person','joão doria':'person', 'Jair Bolsonaro':'person','mike pence':'person',
'video':'others','photo':'others','study':'others','meme':'others','unknown':'others', 'No data':'others','audio recording':'others','scam':'others','viral image':'others','fake image':'others', 'fake document':'others','various authors':'others','health department':'others','daily expose':'others','telegraph':'others','doctors for life':'others'
}


# In[113]:


data['Source'] = data['Source'].replace(Source_mapping)


# In[114]:


data['Source'].value_counts()


# In[21]:


#Top 10 contributing to 65 percent of the data..So using them and binning other countries together
data['Country (mentioned)'].value_counts()[:9]*100/len(data)


# In[22]:


#Using only the top 10 countries as is and binning all the other countries together as "Others"
Country_mapping = {}
for country in data['Country (mentioned)'].unique():
    if country not in data['Country (mentioned)'].value_counts()[:10].index:
        Country_mapping[country] = 'Others'

Country_mapping


# In[23]:


data['Country (mentioned)'] = data['Country (mentioned)'].replace(Country_mapping)


# In[24]:


data['Country (mentioned)'].unique()


# In[25]:


data.columns


# In[26]:


data.Claim[19]


# In[27]:


data.summary[19]


# ### Using cosine similarity of spaCy

# In[28]:


import spacy
get_ipython().system('python -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")


# In[29]:


doc1 = nlp(u'Hospitals in Oklahoma are overwhelmed with cases of people suffering ivermectin overdose after taking the drug as COVID-19 cure.')
doc2 = nlp(u'Online articles and social media posts claim that overdose cases from people using anti-parasitic drug ivermectin against COVID-19 are overwhelming hospitals in Oklahoma, citing remarks by a doctor in the US state.\nAlso Read: No, Bill Gates Did Not Call For Depopulation Through Forced Vaccination"Gunshot Victims Left Waiting as Horse Dewormer Overdoses Overwhelm Oklahoma Hospitals, Doctor Says," said a September 3, 2021 Rolling Stone magazine headline.\n"What we can confirm is that we have seen a handful of ivermectin patients in our emergency rooms, to include INTEGRIS Grove Hospital.\nScott Schaffer, director of the Oklahoma Center for Poison and Drug Information, told AFP that calls related to ivermectin remain relatively low.\n"In Mississippi it was reported that 70 percent of their calls were ivermectin related, but that was misinterpretation," he said.')
print(doc1.similarity(doc2))  


# In[59]:


data_cos_sim = data.copy()


# In[62]:


data_cos_sim['cosine_sim_scores'] = data_cos_sim.apply(lambda x: nlp(x['Claim']).similarity(nlp(x['summary'])),axis=1)


# ### One hot Encoding

# In[64]:


# Using make_column_transformer to One-Hot Encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd


transformer = make_column_transformer(
    (OneHotEncoder(), ['Country (mentioned)','Source']),
    remainder='passthrough')

transformed_cos_sim = transformer.fit_transform(data_cos_sim)
transformed_df_cos_sim = pd.DataFrame(
    transformed_cos_sim, 
    columns=transformer.get_feature_names()
)


# In[116]:


X.columns


# In[66]:


X = transformed_df_cos_sim[['onehotencoder__x0_Brazil', 'onehotencoder__x0_China',
       'onehotencoder__x0_France', 'onehotencoder__x0_India',
       'onehotencoder__x0_Indonesia', 'onehotencoder__x0_Italy',
       'onehotencoder__x0_Others', 'onehotencoder__x0_Portugal',
       'onehotencoder__x0_Spain', 'onehotencoder__x0_United Kingdom',
       'onehotencoder__x0_United States', 'onehotencoder__x1_multiple sources',
       'onehotencoder__x1_news media', 'onehotencoder__x1_others',
       'onehotencoder__x1_person', 'onehotencoder__x1_social media',
       'onehotencoder__x1_websites','cosine_sim_scores']]
y = transformed_df_cos_sim['Label'].values
y=y.astype('int')


# ### Train Test Split

# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)


# ### Model Fitting

# In[68]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 6).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)


# In[71]:


from sklearn import metrics
metrics.accuracy_score(y_test,dtree_predictions)


# ### Reading the test file

# In[40]:


from newspaper import Article
from newspaper import Config
import nltk
import pandas as pd
#The user agent has to be obtained from the https://www.whatismybrowser.com/detect/what-is-my-user-agent for running on your system!
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
config = Config()
config.browser_user_agent = user_agent


# In[41]:


# test = pd.read_csv('/content/drive/MyDrive/Social Media Mining Project 2/test.csv')
test = pd.read_csv("test.csv")


# In[42]:


test


# In[43]:


test['Fact-checked Article'][1]


# In[44]:


url = "https://checamos.afp.com/cristiano-ronaldo-nao-anunciou-plano-de-transformar-hoteis-em-hospitais-para-pacientes-do-novo"
news = Article(url, config=config) 
news.download()
news.parse()
nltk.download('punkt')
news.nlp()
# Extract summary
print(news.summary)


# In[45]:


def fil(url):
    try:
        news = Article(url, config=config) 
        news.download()
        news.parse()
        news.nlp()
        return news.summary
    except:
        return ''
test['summary'] = test.apply(lambda x: fil(x['Fact-checked Article']),axis=1)


# In[46]:


test[test['summary']!='']


# In[49]:


test['Source'] = test['Source'].replace(Source_mapping)
test['Country (mentioned)'] = test['Country (mentioned)'].replace(Country_mapping)


# In[50]:


test['Country (mentioned)'].unique()


# In[51]:


test['Source'].unique()


# In[77]:


test['cosine_sim_scores'] = test.apply(lambda x: nlp(x['Claim']).similarity(nlp(x['summary'])),axis=1)


# In[79]:


# Using make_column_transformer to One-Hot Encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd


transformer = make_column_transformer(
    (OneHotEncoder(sparse=False), ['Country (mentioned)','Source']),
    remainder='passthrough')

transformed_test_cos_sim = transformer.fit_transform(test)
transformed_df_test_cos_sim = pd.DataFrame(
    transformed_test_cos_sim, 
    columns=transformer.get_feature_names()
)


# In[80]:


transformed_df_test_cos_sim.columns


# In[82]:


X = transformed_df_test_cos_sim[['onehotencoder__x0_Brazil', 'onehotencoder__x0_China',
       'onehotencoder__x0_France', 'onehotencoder__x0_India',
       'onehotencoder__x0_Indonesia', 'onehotencoder__x0_Italy',
       'onehotencoder__x0_Others', 'onehotencoder__x0_Portugal',
       'onehotencoder__x0_Spain', 'onehotencoder__x0_United Kingdom',
       'onehotencoder__x0_United States', 'onehotencoder__x1_multiple sources',
       'onehotencoder__x1_news media', 'onehotencoder__x1_others',
       'onehotencoder__x1_person', 'onehotencoder__x1_social media',
       'onehotencoder__x1_websites','cosine_sim_scores']]


# In[83]:


dtree_predictions = dtree_model.predict(X)


# In[86]:


dtree_predictions


# In[87]:


submisssion__ = pd.DataFrame({'Id':range(1,len(X)+1),'Category':dtree_predictions})


# In[88]:


submisssion__.to_csv('Submission2_CosSim.csv',index=False)


# ### Approach 2: Using CrossEncoder since it captures the semantic similarity

# In[30]:


from sentence_transformers import CrossEncoder

# Load the pre-trained model
model = CrossEncoder('cross-encoder/stsb-roberta-base')


# In[32]:


sentence_pairs = []
for sentence1, sentence2 in zip(data['summary'], data['Claim']):
    sentence_pairs.append([sentence1, sentence2])
    
data['SBERT CrossEncoder_score'] = model.predict(sentence_pairs, show_progress_bar=True)


# In[93]:


# Using make_column_transformer to One-Hot Encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd


transformer = make_column_transformer(
    (OneHotEncoder(), ['Country (mentioned)','Source']),
    remainder='passthrough')

transformed = transformer.fit_transform(data)
transformed_df = pd.DataFrame(
    transformed, 
    columns=transformer.get_feature_names()
)


# In[94]:


transformed_df.columns


# In[95]:


X = transformed_df[['onehotencoder__x0_Brazil', 'onehotencoder__x0_China',
       'onehotencoder__x0_France', 'onehotencoder__x0_India',
       'onehotencoder__x0_Indonesia', 'onehotencoder__x0_Italy',
       'onehotencoder__x0_Others', 'onehotencoder__x0_Portugal',
       'onehotencoder__x0_Spain', 'onehotencoder__x0_United Kingdom',
       'onehotencoder__x0_United States', 'onehotencoder__x1_multiple sources',
       'onehotencoder__x1_news media', 'onehotencoder__x1_others',
       'onehotencoder__x1_person', 'onehotencoder__x1_social media',
       'onehotencoder__x1_websites','SBERT CrossEncoder_score']]
y = transformed_df['Label'].values
y=y.astype('int')


# In[96]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)


# In[97]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 6).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)


# In[98]:


from sklearn import metrics
metrics.accuracy_score(y_test,dtree_predictions)


# In[47]:


sentence_pairs_test = []
for sentence1, sentence2 in zip(test['summary'], test['Claim']):
    sentence_pairs_test.append([sentence1, sentence2])
    
test['SBERT CrossEncoder_score'] = model.predict(sentence_pairs_test, show_progress_bar=True)


# In[48]:


test


# In[101]:


# Using make_column_transformer to One-Hot Encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pandas as pd


transformer = make_column_transformer(
    (OneHotEncoder(sparse=False), ['Country (mentioned)','Source']),
    remainder='passthrough')

transformed = transformer.fit_transform(test)
transformed_df_test = pd.DataFrame(
    transformed, 
    columns=transformer.get_feature_names()
)


# In[102]:


transformed_df_test.columns


# In[103]:


X = transformed_df_test[['onehotencoder__x0_Brazil', 'onehotencoder__x0_China',
       'onehotencoder__x0_France', 'onehotencoder__x0_India',
       'onehotencoder__x0_Indonesia', 'onehotencoder__x0_Italy',
       'onehotencoder__x0_Others', 'onehotencoder__x0_Portugal',
       'onehotencoder__x0_Spain', 'onehotencoder__x0_United Kingdom',
       'onehotencoder__x0_United States', 'onehotencoder__x1_multiple sources',
       'onehotencoder__x1_news media', 'onehotencoder__x1_others',
       'onehotencoder__x1_person', 'onehotencoder__x1_social media',
       'onehotencoder__x1_websites','SBERT CrossEncoder_score']]


# In[104]:


label_predictions = dtree_model.predict(X)


# In[105]:


label_predictions


# In[106]:


submisssion = pd.DataFrame({'Id':range(1,len(X)+1),'Category':label_predictions})


# In[107]:


submisssion.to_csv('Submission1_CrossEncoder.csv',index=False)


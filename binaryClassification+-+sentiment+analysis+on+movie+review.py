
# coding: utf-8

# In[79]:


#https://github.com/SrinidhiRaghavan/AI-Sentiment-Analysis-on-IMDB-Dataset/blob/master/driver_3.py


# In[80]:


import pandas as pd


# In[81]:


imdbData = pd.read_csv("C://Users/583175/.jupyter/imdb_tr.csv", header=0, encoding = 'ISO-8859-1')


# In[82]:


reviews = imdbData['text']


# In[84]:


reviews


# In[128]:


reviews[1]


# In[139]:


import re


# In[145]:


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")


# In[146]:


NO_SPACE = ""


# In[165]:


preProcessedText = REPLACE_NO_SPACE.sub(NO_SPACE, "Hello, My name is John. I'm working in google.")


# In[166]:


print(preProcessedText)


# In[85]:


polarity = imdbData['polarity']


# In[86]:


polarity


# In[87]:


import sklearn


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


features = reviews
labels = polarity


# In[90]:


train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.10,
                                                          random_state=42)


# In[92]:


from sklearn.feature_extraction.text import CountVectorizer


# In[102]:


vectorizer = CountVectorizer()
vectorizerTr = vectorizer.fit(train)
vectorizerTe = vectorizer.fit(test)


# In[103]:


trainMatrix = vectorizerTr.transform(train)
testMatrix = vectorizerTe.transform(test)


# In[104]:


trainMatrix


# In[105]:


testMatrix


# In[106]:


print(vectorizer.get_feature_names)
#print(trainMatrix.toarray())


# In[107]:


from sklearn.linear_model import SGDClassifier


# In[108]:


clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=20)


# In[109]:


clf.fit(trainMatrix, train_labels)


# In[112]:


train_labels


# In[113]:


test_labels


# In[116]:


testOutput = clf.predict(testMatrix)


# In[123]:


testOutput


# In[119]:


from sklearn.metrics import accuracy_score


# In[120]:


accuracy = accuracy_score(testOutput, test_labels)


# In[121]:


accuracy


# In[167]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[168]:


confMatrix = confusion_matrix(testOutput, test_labels)


# In[172]:


print(confMatrix)


# In[170]:


clfReport = classification_report(testOutput, test_labels)


# In[173]:


print(clfReport)


# In[124]:


negReview = "The movie was really boring and I hate it"
posReview = "The movie was great and I loved it. Everyone should watch it"
bothReviews = ["The movie was really boring and I hate it", 
               "The movie was great and I loved it. Everyone should watch it",
              "The second part of the movie was not interesting as it was in the first part."]

posReview = [posReview]
negReview = [negReview]


# In[125]:


posReviewMatrix = vectorizer.transform(posReview)
negReviewMatrix = vectorizer.transform(negReview)
bothReviewMatrix = vectorizer.transform(bothReviews)


# In[126]:


sentimentOutput = clf.predict(bothReviewMatrix)


# In[127]:


print(sentimentOutput)


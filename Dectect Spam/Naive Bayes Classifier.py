import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_table('E:/materials/ML\MLND/Dectect Spam/smsspamcollection/SMSSpamCollection',sep='\t',header=None,names=['label','sms_message'])
df.head()
df.shape

df['label']=df.label.map({'ham':0,'spam':1})
df.head()

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))    
print(sans_punctuation_documents)

preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

frequency_list = []

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)    
print(frequency_list)


count_vector = CountVectorizer()
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
doc_array
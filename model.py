# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:02:34 2021

@author: Dhakshin Krishna J & Team
"""

import pandas as pd
import numpy as np
# import regex as re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional 
plt.style.use('seaborn')




#settings varibles
#setting function
def setting(mode):
    if mode=='0':
        #Advanced mode
        modifier=1
        tagTable=1
        tokenArray=1
    elif mode == '1':   
        #Viewer mode
        modifier=0
        tagTable=0
        tokenArray=1
    elif mode=='2':
        #Modifier mode
        modifier=1
        tagTable=0
        tokenArray=1
    else:
        #Default mode
        modifier=0
        tagTable=1
        tokenArray=0
    return modifier,tagTable,tokenArray
             
        
    
        

#combining two csv files to create the dataframe
TheDataList=[]
listOfCsv=["datasets/ner_datasetreference.csv","datasets/added.csv"]
for x in listOfCsv:
    TheDataList.append(pd.read_csv(x,encoding='unicode_escape',skipinitialspace=True,skip_blank_lines=True))
    
data=pd.concat(TheDataList)
data.head(-20)
    
    
data=data.fillna(method="ffill")
data.head(-1)
words=list(set(data['Word'].values))
words.append("ENDPAD")
#print(words)


num_words=len(words)
print("Total number of words",num_words)


tags = list(set(data["Tag"].values))
num_tags = len(tags)
print("List of tags: " + ', '.join([tag for tag in tags]))
print(f"Total Number of tags {num_tags}")
  
    
 #creating class for the model to access
class Get_sentence(object):
  def __init__(self,data):
    self.n_sent=1
    self.data=data
    agg_func=lambda s:[(w,t,p) for w,t,p in zip(s['Word'].tolist(),s['Tag'].tolist(),s['POS'].tolist())]
    self.grouped=self.data.groupby('Sentence #').apply(agg_func)
    self.sentences=[s for s in self.grouped]
     
        
getter=Get_sentence(data)
sentence=getter.sentences
 #len(getter.data)
 # print(sentence[47959])
 #len(sentence)


      
    
#     #RE EVALUTAION USING POS TAGS
# regTimeEnd=r'^[NV].*'
# regTimeContn=r'^[JRTID].*'
# regNumEnd=r'^[NJ].*'
# regNumContn=r'^[TID].*'
# regLocEnd=r'^[N].*'
# regLocContn=r'^[TID].*'
# regDisEnd=r'^[JRN].*'
# regDisContn=r'^[TID].*'
   

# def checkFor(s,idx,tag,regEnd,regContn,repeat):
#     if (idx>0 and idx+1<len(s)):
#         prev=s[idx-1]
#         nxt=s[idx+1]
#         if prev[1]=="O":
#             if re.match(regEnd,prev[2]):
#                 s[idx-1]=tuple([prev[0],tag,prev[2]])
              
#             elif (re.match(regContn,prev[2]) and repeat==0):
#                 s[idx-1]=tuple([prev[0],tag,prev[2]])
#                 s=checkFor(s,idx-1,tag,regEnd,regContn,1)
#         if nxt[1]=="O":
#             if re.match(regEnd,nxt[2]):
#                 s[idx+1]=tuple([nxt[0],tag,nxt[2]])
              
#             elif re.match(regContn,prev[2]):
#                 s[idx+1]=tuple([nxt[0],tag,nxt[2]])
#                 s=checkFor(s,idx+1,tag,regEnd,regContn,1)
#     return s
# def check_tim(s):
#     for idx,p in enumerate(s):
#         if(p[1]=="B-tim"):
#             s=checkFor(s,idx,"B-tim",regTimeEnd,regTimeContn,0)
#         if(p[1]=="I-tim"):
#             s=checkFor(s,idx,"I-tim",regTimeEnd,regTimeContn,0)            
#     return s    
# def check_num(s):
#     for idx,p in enumerate(s):
#         if(p[1]=="Num"):
#             s=checkFor(s,idx,"Num",regNumEnd,regNumContn,0)            
#     return s
# def check_loc(s):
#     for idx,p in enumerate(s):
#         if(p[1]=="B-geo"):
#             s=checkFor(s,idx,"B-geo",regLocEnd,regLocContn,0)
#         if(p[1]=="I-geo"):
#             s=checkFor(s,idx,"I-geo",regLocEnd,regLocContn,0)
#         if(p[1]=="I-gpe"):
#             s=checkFor(s,idx,"I-gpe",regLocEnd,regLocContn,0)
#         if(p[1]=="B-gpe"):
#             s=checkFor(s,idx,"B-gpe",regLocEnd,regLocContn,0)            
#     return s
# def check_dis(s):
#     for idx,p in enumerate(s):
#         if(p[1]=="Dis"):
#             s=checkFor(s,idx,"Dis",regDisEnd,regDisContn,0)
#         if(p[1]=="Dis-impact"):            
#             s=checkFor(s,idx,"Dis-impact",regDisEnd,regDisContn,0)            
#     return s
# def posRules(s):
#     s=check_tim(s)
#     s=check_num(s)
#     s=check_loc(s)
#     s=check_dis(s)
#     return s
for s in sentence:
  
    len_s=len(s)
   
   #initialize id for words and tags
word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)}
print(tag_idx)
# word_idx["India"]
# tag_idx["Dis"]
# tag_idx
    
    
     #plot figure for the frequency of length of sentences
plt.figure(figsize=(14,7))
plt.hist([len(s) for s in sentence],bins = 50)
plt.xlabel("Length of Sentences")
plt.show()
    
    
    
     #plot the figure for frequency of tags
plt.figure(figsize=(14, 7))
plt.xlabel("Frequency of tags")
data.Tag[data.Tag != 'O'].value_counts().plot.barh();    
    
# word_idx.keys()
   
max_len=max([len(s) for s in sentence])
# max_len
X=[[word_idx[w[0]] for w in s] for s in sentence]
X=pad_sequences(maxlen=max_len,sequences=X,padding='post',value=num_words)
y=[[tag_idx[w[1]]for w in s]for s in sentence]
y=pad_sequences(maxlen=max_len,sequences=y,padding='post',value=tag_idx['O'])

    
y=[to_categorical(i,num_classes=num_tags) for i in y]
    
    
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)
    
    
    
    
#np.shape(X_train)
#np.shape(X_test)
#np.shape(y_test)
#np.shape(y_train)
    
    
input_word=Input(shape=(max_len,))
   
#BI-lstm
model=Embedding(input_dim=num_words+1,output_dim=max_len,input_length=max_len)(input_word)
model=SpatialDropout1D(0.1)(model)
model=Bidirectional(LSTM(units=52,return_sequences=True,recurrent_dropout=0.1))(model)
out=TimeDistributed(Dense(num_tags,activation='softmax'))(model)
model=Model(input_word,out)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
 
  
#LSTM-----MODEL IGNORED AS BI-LSTM IS MORE EFFECTIVE
    
# model1=Embedding(input_dim=num_words+1,output_dim=max_len,input_length=max_len)(input_word)
# model1=SpatialDropout1D(0.1)(model1)
# lstm1=LSTM(units=64,return_sequences=True)(model1)
# lstm2=LSTM(units=64,return_sequences=True)(lstm1)
# model1=Bidirectional(LSTM(units=52,return_sequences=True,recurrent_dropout=0.1))(model1)
# out1=TimeDistributed(Dense(num_tags,activation='relu'))(lstm2)
# model1=Model(input_word,out1)
  
# model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model1.summary()
 
  
# ERROR ANALYSED:ONLY APPLICABLE IN JUPYTER NOTEBOOKS
# plot_model(model,show_shapes=True)
    
    
    
model.fit(X_train,np.array(y_train),verbose=1,epochs=1,validation_split=0.2)
  
   
print("------------------------------Model Training complete------------")
 
model.evaluate(X_test,np.array(y_test))
   
print("-------------------------------Model Testing complete------------")
    
model.save("savedModel");    
    
print("-------------------------------Model Stored for future------------")
    
    
#RANODMLTY PICK A SENTENCE AND TEST THE OUTCOME
    

x=input("Want to test a random sentence from the dataset?y/n:")
if x=='y' or x=='Y':
    rand_sent=np.random.randint(0,X_test.shape[0])
    p=model.predict(np.array([X_test[rand_sent]]))
    p=np.argmax(p,axis=-1)

    y_true=np.argmax(np.array(y_test),axis=-1)[rand_sent]
    print("{:20}{:20}\t{}\n".format("Word","Truth","Pred"))
    print("-"*55)
    strl=" "
    for (w,t,pred)in zip(X_test[rand_sent],y_true,p[0]):
        
        if(words[w-1] != "ENDPAD"):
            if not(tags[t]=="O" and tags[pred]=="O"):
                print("{:20}{:20}\t{}".format(words[w-1],tags[t],tags[pred]))
        if(words[w-1] != "ENDPAD"):
            strl=strl+words[w-1]+" "    
    print("\n\n"+strl);    
           


            
            
        

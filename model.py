# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:02:34 2021

@author: Dhakshin Krishna J & Team
"""

import pandas as pd
import numpy as np
import nltk
import regex as re
import matplotlib.pyplot as plt
import tensorflow
from csv import DictWriter
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import word_tokenize
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


      
    
    #RE EVALUTAION USING POS TAGS
regTimeEnd=r'^[NV].*'
regTimeContn=r'^[JRTID].*'
regNumEnd=r'^[NJ].*'
regNumContn=r'^[TID].*'
regLocEnd=r'^[N].*'
regLocContn=r'^[TID].*'
regDisEnd=r'^[JRN].*'
regDisContn=r'^[TID].*'
   

def checkFor(s,idx,tag,regEnd,regContn,repeat):
    if (idx>0 and idx+1<len(s)):
        prev=s[idx-1]
        nxt=s[idx+1]
        if prev[1]=="O":
            if re.match(regEnd,prev[2]):
                s[idx-1]=tuple([prev[0],tag,prev[2]])
              
            elif (re.match(regContn,prev[2]) and repeat==0):
                s[idx-1]=tuple([prev[0],tag,prev[2]])
                s=checkFor(s,idx-1,tag,regEnd,regContn,1)
        if nxt[1]=="O":
            if re.match(regEnd,nxt[2]):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
              
            elif re.match(regContn,prev[2]):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
                s=checkFor(s,idx+1,tag,regEnd,regContn,1)
    return s
def check_tim(s):
    for idx,p in enumerate(s):
        if(p[1]=="B-tim"):
            s=checkFor(s,idx,"B-tim",regTimeEnd,regTimeContn,0)
        if(p[1]=="I-tim"):
            s=checkFor(s,idx,"I-tim",regTimeEnd,regTimeContn,0)            
    return s    
def check_num(s):
    for idx,p in enumerate(s):
        if(p[1]=="Num"):
            s=checkFor(s,idx,"Num",regNumEnd,regNumContn,0)            
    return s
def check_loc(s):
    for idx,p in enumerate(s):
        if(p[1]=="B-geo"):
            s=checkFor(s,idx,"B-geo",regLocEnd,regLocContn,0)
        if(p[1]=="I-geo"):
            s=checkFor(s,idx,"I-geo",regLocEnd,regLocContn,0)
        if(p[1]=="I-gpe"):
            s=checkFor(s,idx,"I-gpe",regLocEnd,regLocContn,0)
        if(p[1]=="B-gpe"):
            s=checkFor(s,idx,"B-gpe",regLocEnd,regLocContn,0)            
    return s
def check_dis(s):
    for idx,p in enumerate(s):
        if(p[1]=="Dis"):
            s=checkFor(s,idx,"Dis",regDisEnd,regDisContn,0)
        if(p[1]=="Dis-impact"):            
            s=checkFor(s,idx,"Dis-impact",regDisEnd,regDisContn,0)            
    return s
def posRules(s):
    s=check_tim(s)
    s=check_num(s)
    s=check_loc(s)
    s=check_dis(s)
    return s
for s in sentence:
  
    len_s=len(s)
    s=posRules(s)
  
   
   #initialize id for words and tags
word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)}

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
           

#function to evaluate the custom input
# model=tensorflow.keras.models.load_model("savedModel")

# def validateInput(strCustom,sett):
#     modifier,tagTable,tokenArray=setting(sett)
#     fields=['Sentence #','Word','POS','Tag']
#     outStr=[]
#     #strCustom="this is a dummy text"
#     A=word_tokenize(strCustom)
#     posA=nltk.pos_tag(A)
#     A,pos_A=zip(*posA)
#     #print(pos_A)
#     #print(A)
#     A_test=[]
#     if tokenArray==1:
#         print(A)
#     checkForNew=0
#     for w in A:
#         outStr.append(w)
#         if w in words:
#             A_test.append(word_idx[w])
#         else:
#             checkForNew=1
#             A_test.append(word_idx["this"])
            
            
            
                        
#     for w in range(104-len(A)):
#         A_test.append(word_idx["ENDPAD"])
        
        
#     p=model.predict(np.array([A_test]))
#     p=np.argmax(p,axis=-1)
    
#     print("{:20}\t{}\n".format("Word","Pred"))
#     print("-"*55)
    
#     for (w,pred)in zip(A_test,p[0]):
    
#         if(words[w-1] != "ENDPAD"):
#             if(tagTable==0):
#                 if(tags[pred]!="O"):
#                     print("{:20}\t{}".format(outStr[w-1],tags[pred]))
#             else:
#                 print("{:20}\t{}".format(outStr[w-1],tags[pred]))
    
    
#     print("\n\n"+strCustom);
    
#     if checkForNew==1:
#         if modifier==1:
#             r=input("\n\nHappy with the tags? if not type ------    x    ------- to help us improve the model:\n")
#             if r=='x':
#                 len_sen=len(sentence)
#                 print("tag-id reference:\n")
#                 [print(key,':',value) for key, value in tag_idx.items()]
#                 y=len_sen+1
                
#                 print(strCustom+"\n in this sentence\n")
#                 counter=0
#                 with open('added.csv','a',newline='') as f_object :
#                     dictWriter_object=DictWriter(f_object,fieldnames=fields)
#                     for x in A:
#                         q=input("Enter tag for this word:  -->"+x+"\n")
#                         newstr=f'{y}'
#                         newDict={fields[0]:"Sentence: "+newstr,fields[1]: x,fields[2]:pos_A[counter],fields[3]:q}
#                         counter+=1
#                         dictWriter_object.writerow(newDict)
#                     f_object.close()
#             else:
#                 print("Thank you have a nice day!")
                
#         counter=0
    
    
    
    
# #TAKE A USER GIVEN INPUT AND predict thee output
# x=input("Want to test a CUSTOM INPUT y/n:")
# if x=='y' or x=='Y': 
    
#     #There is a massive earthquake in nepal, which has caused a lot of damage to the surrounding around the epicenter
#     ask=input("CHOOSE YOUR INPUT METHOD:\n\n1.Input in CLI (max 100 WORDS..)\n2.Input the address of a text file containing input:\n\n")
#     sett=input("Enter 0=> for running the program in Advanced mode\n 1=> for running the program in Viewer mode\n 2=> for running the program in Modifier mode\n ANY=> for running the program in Default mode\n")
#     if ask=='1':
#         strInput=input("Enter the message:\n")
#         validateInput(strInput,sett)
         
#     elif ask=='2':
#         fileName=input("enter the file name:\n")
#         fileName+=".txt"
#         with open(fileName,"r") as file:
#             lines=(line.rstrip() for line in file)
#             dataCustom=list(line for line in lines if line)
#         for each in dataCustom:
#             validateInput(each,sett)
            
            
            
        

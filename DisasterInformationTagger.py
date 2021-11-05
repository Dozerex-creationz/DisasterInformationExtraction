# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:08:44 2021

@author: Dhakshin Krishna J
"""

import pandas as pd
import numpy as np
import nltk
import regex as re
import matplotlib.pyplot as plt
import tensorflow
from csv import DictWriter
from nltk.tokenize import word_tokenize
#function to evaluate the custom input


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

    
     #plot figure for the frequency of length of sentences
plt.figure(figsize=(14,7))
plt.hist([len(s) for s in sentence],bins = 50)
plt.xlabel("Length of Sentences")
plt.show()
    
    
      #plot the figure for frequency of tags
plt.figure(figsize=(14, 7))
plt.xlabel("Frequency of tags")
data.Tag[data.Tag != 'O'].value_counts().plot.barh();


 #initialize id for words and tags
word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)}
  
model=tensorflow.keras.models.load_model("savedModel")

def validateInput(strCustom,sett):
    modifier,tagTable,tokenArray=setting(sett)
    fields=['Sentence #','Word','POS','Tag']
    outStr=[]
    #strCustom="this is a dummy text"
    A=word_tokenize(strCustom)
    posA=nltk.pos_tag(A)
    A,pos_A=zip(*posA)
    #print(pos_A)
    #print(A)
    A_test=[]
    if tokenArray==1:
        print(A)
    checkForNew=0
    for w in A:
        outStr.append(w)
        if w in words:
            A_test.append(word_idx[w])
        else:
            checkForNew=1
            A_test.append(word_idx["this"])
            
            
            
                        
    for w in range(104-len(A)):
        A_test.append(word_idx["ENDPAD"])
        
        
    p=model.predict(np.array([A_test]))
    p=np.argmax(p,axis=-1)
    
    print("{:20}\t{}\n".format("Word","Pred"))
    print("-"*55)
    
    for (w,pred)in zip(A_test,p[0]):
    
        if(words[w-1] != "ENDPAD"):
            if(tagTable==0):
                if(tags[pred]!="O"):
                    print("{:20}\t{}".format(outStr[w-1],tags[pred]))
            else:
                print("{:20}\t{}".format(outStr[w-1],tags[pred]))
    
    
    print("\n\n"+strCustom);
    
    if checkForNew==1:
        if modifier==1:
            r=input("\n\nHappy with the tags? if not type ------    x    ------- to help us improve the model:\n")
            if r=='x':
                len_sen=len(sentence)
                print("tag-id reference:\n")
                [print(key,':',value) for key, value in tag_idx.items()]
                y=len_sen+1
                
                print(strCustom+"\n in this sentence\n")
                counter=0
                with open('added.csv','a',newline='') as f_object :
                    dictWriter_object=DictWriter(f_object,fieldnames=fields)
                    for x in A:
                        q=input("Enter tag for this word:  -->"+x+"\n")
                        newstr=f'{y}'
                        newDict={fields[0]:"Sentence: "+newstr,fields[1]: x,fields[2]:pos_A[counter],fields[3]:q}
                        counter+=1
                        dictWriter_object.writerow(newDict)
                    f_object.close()
            else:
                print("Thank you have a nice day!")
                
        counter=0
    
    
    
    
#TAKE A USER GIVEN INPUT AND predict thee output
x=input("Want to test a CUSTOM INPUT y/n:")
if x=='y' or x=='Y': 
    
    #There is a massive earthquake in nepal, which has caused a lot of damage to the surrounding around the epicenter
    ask=input("CHOOSE YOUR INPUT METHOD:\n\n1.Input in CLI (max 100 WORDS..)\n2.Input the address of a text file containing input:\n\n")
    sett=input("Enter 0=> for running the program in Advanced mode\n 1=> for running the program in Viewer mode\n 2=> for running the program in Modifier mode\n ANY=> for running the program in Default mode\n")
    if ask=='1':
        strInput=input("Enter the message:\n")
        validateInput(strInput,sett)
         
    elif ask=='2':
        fileName=input("enter the file name:\n")
        fileName+=".txt"
        with open(fileName,"r") as file:
            lines=(line.rstrip() for line in file)
            dataCustom=list(line for line in lines if line)
        for each in dataCustom:
            validateInput(each,sett)

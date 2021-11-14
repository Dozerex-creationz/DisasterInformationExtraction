# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:08:44 2021

@author: Dhakshin Krishna J
"""

import pandas as pd
import numpy as np
import nltk
import regex as re
# import matplotlib.pyplot as plt
import tensorflow
from csv import DictWriter
from nltk.tokenize import word_tokenize
#function to evaluate the custom input


#settings varibles
#setting function

disCom=["flood","quake","cyclone","typhoon","hurricane","storm","disaster","emergency","volcano","eruption","bomb","blast","landfall","landslide","tsunami","massacre"]
disImp=["evacuation","epicenter","killed","attacked","destruct","destroy","damage","debris","wreck","havoc","death","casualit","martyr"]
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
        tokenArray=0
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

#default value of settings is set to mode =3 i.e not 0,1,2
sett=3
#combining two csv files to create the dataframe
TheDataList=[]
listOfCsv=["datasets/ner_datasetreference.csv","datasets/added.csv"]
for x in listOfCsv:
    TheDataList.append(pd.read_csv(x,encoding='unicode_escape',skipinitialspace=True,skip_blank_lines=True))
    
data=pd.concat(TheDataList)
# data.head(-20)
    
    
data=data.fillna(method="ffill")
data.head(-1)
words=list(set(data['Word'].values))
words.append("ENDPAD")
#print(words)


num_words=len(words)
# print("Total number of words",num_words)


tags = list(set(data["Tag"].values))
tags.sort()
num_tags = len(tags)
# print("List of tags: " + ', '.join([tag for tag in tags]))
# print(f"Total Number of tags {num_tags}")
  
    
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

    
#      #plot figure for the frequency of length of sentences
# plt.figure(figsize=(14,7))
# plt.hist([len(s) for s in sentence],bins = 50)
# plt.xlabel("Length of Sentences")
# plt.show()
    
    
#       #plot the figure for frequency of tags
# plt.figure(figsize=(14, 7))
# plt.xlabel("Frequency of tags")
# data.Tag[data.Tag != 'O'].value_counts().plot.barh();


    #RE EVALUTAION USING POS TAGS
regTimeEnd=r'^[NV].*'
regTimeContn=r'^[TID].*'
regNumEnd=r'^[NJ].*'
regNumContn=r'^[TID].*'
regLocEnd=r'^[NJ].*'
regLocContn=r'^[TID].*'
regDisEnd=r'^[JRN].*'
regDisContn=r'^[TID].*'
regNoun=r'^N'
regVerb=r'^V'   
fileResult=[]
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
            elif (re.match(regContn,prev[2]) and repeat==1):
                ntn=s[idx+1]
                s[idx+1]=tuple(ntn[0],"O",ntn[2])
        if nxt[1]=="O":
            if re.match(regEnd,nxt[2]):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
            elif (re.match(regContn,nxt[2]) and repeat==0):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
                s=checkFor(s,idx+1,tag,regEnd,regContn,1)
            elif (re.match(regContn,nxt[2]) and repeat==1):
                ntn=s[idx-1]
                s[idx-1]=tuple(ntn[0],"O",ntn[2])    
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
def checkDict(s):
    for id,p in enumerate(s):
        if(p[1]!='Dis' or p[1]!='Dis-impact'):
            for k in disCom:
                if p[0].find(k)!=-1 and re.match(regNoun,p[2]):
                    s[id]=tuple([p[0],"Dis",p[2]])
                elif p[0].find(k)!=-1:
                    s[id]=tuple([p[0],"Dis-impact",p[2]])                
            for r in disImp:
                if p[0].find(r)!=-1:
                    s[id]=tuple([p[0],"Dis-impact",p[2]])        
                  
    return s 
def posRules(s):
    s=checkDict(s)
    s=check_tim(s)
    s=check_num(s)
    s=check_loc(s)
    s=check_dis(s)
    return s


 #initialize id for words and tags
word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)}
model=tensorflow.keras.models.load_model("savedModel")



def validateInput(strCustom,sett,method):
    checkForNew=0
    modifier,tagTable,tokenArray=setting(sett)
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
    line=[]
    c=0
    for (w,pred) in zip(A_test,p[0]):
        if words[w-1]!="ENDPAD":
            line.append(tuple([outStr[c],tags[pred],pos_A[c]]))
            c+=1
    print("hiiiiiiiiiii")
    print(p[0])
    line=posRules(line)
    print(line)
    print("{:20}\t{}\n".format("Word","Pred"))
    print("-"*55)
    temp=[]
    for w in line:
        
        if(tagTable==0):
            if(w[1]!="O"):
                print("{:20}\t{}".format(w[0],w[1]))
                temp.append((w[0],w[1]))
        else:
            print("{:20}\t{}".format(w[0],w[1]))           
            temp.append((w[0],w[1]))
    
    print("\n\n"+strCustom);
    if method=="text":
        checkNew(checkForNew,strCustom,A,pos_A)
        return temp
    else:
        return temp
        
    
def checkNew(checkForNew,strCustom,A,pos_A):
    modifier,tagTable,tokenArray=setting(sett)
    fields=['Sentence #','Word','POS','Tag']
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
def taggerProgram(ask,sett): 
    
    #There is a massive earthquake in nepal, which has caused a lot of damage to the surrounding around the epicenter
    if ask=='1':
        strInput=input("Enter the sentence:")
        result = validateInput(strInput,sett,"text")
        return result,strInput 
    elif ask=='2':
        strInput=input("Enter the fileName:")
        strInput+=".txt"
        with open(strInput,"r",errors="ignore") as file:
            lines=(line.rstrip() for line in file)
            dataCustom=list(line for line in lines if line)
        for idx,each in enumerate(dataCustom):
            fileResult.append((each,validateInput(each,sett,"file"),idx))
        return fileResult
            


sett=input("Which mode do you wanna run the program\n\n1.Admin Mode\n2.Viewer Mode\n3.Modifier Mode.\n")
ask=input("For text input press 1\nfor file input enter 2\n")
   
taggerProgram(ask,sett)   
#end   

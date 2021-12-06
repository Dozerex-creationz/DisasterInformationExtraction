# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:08:44 2021

@author: Dhakshin Krishna J
"""

#There is a massive earthquake in Nepal, which has caused a lot of damage to the surroundings around the epicenter

import pandas as pd
import numpy as np
import nltk
import regex as re
import tensorflow
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
from nltk.tokenize import word_tokenize
#function to evaluate the custom input


#settings varibles
#setting function

disCom=["flood","quake","cyclone","typhoon","hurricane","storm","disaster","emergency","volcano","depression","rainfall","eruption","bomb","blast","landfall","landslide","tsunami","massacre"]
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



    #RE EVALUTAION of O TAGS USING POS TAGS
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
                s[idx+1]=tuple([ntn[0],"O",ntn[2]])
        if nxt[1]=="O":
            if re.match(regEnd,nxt[2]):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
            elif (re.match(regContn,nxt[2]) and repeat==0):
                s[idx+1]=tuple([nxt[0],tag,nxt[2]])
                s=checkFor(s,idx+1,tag,regEnd,regContn,1)
            elif (re.match(regContn,nxt[2]) and repeat==1):
                ntn=s[idx-1]
                s[idx-1]=tuple([ntn[0],"O",ntn[2]])    
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
    s=check_dis(s)
    s=check_tim(s)
    s=check_num(s)
    s=check_loc(s)
    return s


 #initialize id for words and tags
word_idx = {w : i + 1 for i ,w in enumerate(words)}
tag_idx =  {t : i for i ,t in enumerate(tags)} 
model=tensorflow.keras.models.load_model("savedModel")


#validating input, fitting it accrdn to the shape of the prdiction model and prediction of tags
def validateInput(strCustom,sett,method):
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
            A_test.append(word_idx["this"])
            
            
            
                        
    for w in range(104-len(A)):
        A_test.append(word_idx["ENDPAD"])
        
    #predicting the tags    
    p=model.predict(np.array([A_test]))
    p=np.argmax(p,axis=-1)
    line=[]
    c=0
    
    #printing the tags
    for (w,pred) in zip(A_test,p[0]):
        if words[w-1]!="ENDPAD":
            line.append(tuple([outStr[c],tags[pred],pos_A[c]]))
            c+=1
    line=posRules(line)
    temp=[]
    
    #packing output for delivery
    for w in line:
        
        if(tagTable==0):
            if(w[1]!="O"):
                
                temp.append((w[0],w[1]))
        else:
        
            temp.append((w[0],w[1]))
    

    return temp
        
    
        
    
#TAKE A USER GIVEN INPUT AND predict thee output
def taggerProgram(ask,sett,strInput): 
    
    #There is a massive earthquake in Nepal, which has caused a lot of damage to the surroundings around the epicenter
    if ask=='1':
        result = validateInput(strInput,sett,"text")
        return result,strInput 
    elif ask=='2':
        with open(strInput,"r",errors="ignore") as file:
            lines=(line.rstrip() for line in file)
            dataCustom=list(line for line in lines if line)
        for idx,each in enumerate(dataCustom):
            fileResult.append((each,validateInput(each,sett,"file"),idx))
        return fileResult
            



#web application            
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
@app.route('/')
def index():
    return render_template('index.html')


#route for file based extraction
@app.route('/file',methods=['POST'])
def file():
    if(request.files['fileInput']):
        f = request.files['fileInput']
        result=secure_filename(f.filename)
        f.save(result)
        rest=taggerProgram('2','1',result)
        finRes=[]
    
        for p in rest:
            newResD,newResDI,newResSP,newResN,newResL,newResT,newResO=[],[],[],[],[],[],[]
            r=p[1]
            for x in r:
                if x[1]=="Dis":
                    newResD.append(x)
                elif x[1]=="Dis-impact":
                    newResDI.append(x)
                elif x[1]=="Num":
                    newResN.append(x)
                elif x[1]=="B-tim" or x[1]=="I-tim":
                    newResT.append(x)
                elif x[1]=="B-geo" or x[1]=="I-geo" or x[1]=="B-gpe" or x[1]=="I-gpe":
                    newResL.append(x)
                elif x[1]=="B-eve" or x[1]=="I-eve" or x[1]=="I-per" or x[1]=="B-per" or x[1]=="I-org" or x[1]=="B-org" or x[1]=="I-nat" or x[1]=="B-nat":
                    newResSP.append(x)
                else:
                    newResO.append(x)
            finRes.append(tuple([p[0],[newResD,newResDI,newResT,newResN,newResL,newResSP,newResO],p[2]]))   
        return render_template('fileReport.html',fullFile=finRes)
    else:
        return redirect(url_for('index'))
    
#route for text based extraction    
@app.route('/text',methods=['POST'])
def text():
    if(request.form["textInput"]):
        response=request.form
        result=response["textInput"]
        rest,sen=taggerProgram('1','1',result)
        newResD,newResDI,newResSP,newResN,newResL,newResT,newResO=[],[],[],[],[],[],[]
        for x in rest:
            if x[1]=="Dis":
                newResD.append(x)
            elif x[1]=="Dis-impact":
                newResDI.append(x)
            elif x[1]=="Num":
                newResN.append(x)
            elif x[1]=="B-tim" or x[1]=="I-tim":
                newResT.append(x)
            elif x[1]=="B-geo" or x[1]=="I-geo" or x[1]=="B-gpe" or x[1]=="I-gpe":
                newResL.append(x)
            elif x[1]=="B-eve" or x[1]=="I-eve" or x[1]=="I-per" or x[1]=="B-per" or x[1]=="I-org" or x[1]=="B-org" or x[1]=="I-nat" or x[1]=="B-nat":
                newResSP.append(x)
            else:
                newResO.append(x)
        res=[newResD,newResDI,newResT,newResN,newResL,newResSP,newResO]         
        return render_template('textReport.html',result=res,sentence=sen)
    else:
        return redirect(url_for('index'))
    
    
#main web app.run()
if __name__ == '__main__':
   app.run()
   debug=True
   
   
   
   
#end   

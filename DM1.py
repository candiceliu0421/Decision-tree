#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:07:07 2017

@author: candice
"""

import pandas as pd
import numpy as np
from sklearn import tree
import pydotplus
#from sklearn.preprocessing import add_dummy_feature




def initial(df):    
    data = pd.read_csv("character-deaths.csv")
    df = pd.DataFrame(data)
    df = df.drop('Death Year', 1)
    df = df.drop('Death Chapter', 1)

    df['Book of Death'] = np.where(np.isnan(df['Book of Death']), 0, 1)
    df['Book Intro Chapter'] = np.where(np.isnan(df['Book Intro Chapter']), 0, 1)
    dummyAllegiances(df)
    

    
def dummyAllegiances(df):    
    for name in np.unique(df['Allegiances']):
	    df['dummy'+name] = df['Allegiances'].map(lambda x: 1 if x == name else 0)

    df = df.drop('Allegiances', 1)
    Classification(df,clf = None)
    
def Classification(df,clf):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(df.values[0:round(len(df)*0.75),2:].tolist(), df.values[0:round(len(df)*0.75),1].tolist())
	#First column is name and the second is dead or alive.
    
    graph_data = tree.export_graphviz(clf, out_file=None,
                         filled=True, rounded=True,
                         special_characters=True,
						 max_depth=3,
						 feature_names= df.columns.values[2:].tolist())
    #Load graph as defined by data in DOT format.	
    graph = pydotplus.graph_from_dot_data(graph_data)  
    graph.write_pdf("tree.pdf")
    Test(df,clf)

#Use ConfusionMatrix to calculate the answer    
def Test(df,clf):
    predictData = clf.predict(df.values[round(len(df)*0.75)+1:,2:]).tolist()
    ConfusionMatrix ={'TP':0,'FN':0, 'FP':0,'TN':0}

    for index, predict in enumerate(predictData):
        if predict == df.values[index + round(len(df)*0.75),1]:
            if predict==1:
                ConfusionMatrix['TP']+=1
            else:		   
                ConfusionMatrix['TN']+=1
        else:
            if predict==1:
                ConfusionMatrix['FP']+=1
            else:
                ConfusionMatrix['FN']+=1
    analysis(ConfusionMatrix)
    
def analysis(ConfusionMatrix):
	print("Accuracy: "+str(round((ConfusionMatrix['TP']+ConfusionMatrix['TN'])/sum(ConfusionMatrix.values()),4)))
	print("Precision: "+str(round(ConfusionMatrix['TP']/(ConfusionMatrix['TP']+ConfusionMatrix['FP']),4)))
	print("Recall: "+str(round(ConfusionMatrix['TP']/(ConfusionMatrix['TP']+ConfusionMatrix['FN']),4)))
    
if __name__ == '__main__':
    	initial(df = None)

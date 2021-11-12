#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:22:19 2021

@author: mohamednacereddinetoros
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import statsmodels.robust as rb
import matplotlib.pyplot as plt
import plotnine as p9

import seaborn as sns

from statistics import mode

import random

donneeA = pd.read_csv('DonnÇesFournisseur1_v0r2.csv')
donneeB = pd.read_csv('DonnÇesFournisseur2_v0r2.csv')
donneeC = pd.read_csv('DonnÇesFournisseur3_v0r2.csv')
donneesUsine =  pd.read_csv('DonnÇesUsine_v0r2.csv')

" Idée consolidé les données du 3 fournisseur en un seul donneABC "

donneeABC = pd.concat([donneeA, donneeB, donneeC], axis = 0)

donneeABC.rename(columns={'Dates': 'dates'}, inplace=True)


donneeABC_triee= donneeABC.sort_values(by = 'dates')
donneeABC_triee.index = donneeABC.index

donneesUsine_triee = donneesUsine.sort_values(by = 'Dates')
donneesUsine_triee.index = donneesUsine.index


donneesUsine_triee.drop('Dates', inplace=True, axis=1)

donneeABC_G = pd.concat([donneeABC_triee.reset_index(drop=True), donneesUsine_triee.reset_index(drop=True)], axis= 1)

# convert the 'Date' column to datetime format
#donneeABC_G['dates']= donneeABC_G.to_datetime(donneeABC_G['dates'], format='%d%b%Y')


statsABC= donneeABC_G.describe()
dimensionsABC = donneeABC_G.shape
nomsvariablesABC = pd.DataFrame(donneeABC_G.columns)
statsABC = donneeABC.describe()
dimensionsABC = donneeABC.shape
nomsvariablesABC = pd.DataFrame(donneeABC_G.columns)


obj_donneeABC = donneeABC_G.select_dtypes(include=['object']).copy()
num_donneeABC = donneeABC_G.select_dtypes(exclude=['object']).copy()

"Mesure des indices de qualité pour Fournisseur 1"

"Calcul du degré de complitude pour l'ensemble du jeu données global"

NR_ABC = dimensionsABC[0]

Nnan_ABC = donneeABC_G.isnull().sum()

DegCompletudeABC = 1 - (Nnan_ABC/NR_ABC)


"=========================Stratégie de nettoyage des donnees =============================="


"Décision de supprimer le champs Niveau Bassin avec un indice de complétude de 0.83 ou qui sont hors context du domain d'affaire"

donneeABC_G_clean = donneeABC_G.drop('Niveau bassin',1)
donneeABC_G_clean = donneeABC_G.drop("Types d'arrêts",1)
donneeABC_G_clean = donneeABC_G.drop("Usine",1)

names_cols = donneeABC_G_clean.columns
NR = donneeABC_G_clean.shape[0]


"Strategie 02 enlever les tuples qui contienent des valeurs manquantes (nan)  (ici avec indice de 0.98 et 0,99 tolérable)"
donneeABC_G_clean = donneeABC_G_clean.dropna()

Nombre_total_cumulé_billots = donneeABC_G_clean['Épinettes (# de billots)']+donneeABC_G_clean['Sapins (# de billots)']+donneeABC_G_clean['Pins (# de billots)']+donneeABC_G_clean['Autres résineux (# de billots)']+donneeABC_G_clean['Peupliers (# de billots)']+donneeABC_G_clean['Érables (# de billots)']+donneeABC_G_clean['Bouleaux (# de billots)']

"Calcul du degré de cohérence pour le nombre des billots"
diff=abs(Nombre_total_cumulé_billots-donneeABC_G_clean["Nombre total de billots"])
NNC=sum(i != 0 for i in diff)
DegCoherence=(NR-NNC)/NR
diff=pd.DataFrame(diff)
diff.index = donneeABC_G_clean.index
diff.columns = ["Différence"]


"Enlever les tuples incohérents"
diff=pd.DataFrame(diff)
donneeABC_G_clean = donneeABC_G_clean[(diff["Différence"] == 0)]

#ax_production_box = donneeABC_G_clean['Production (pmp)'].plot.box()
#ax_production_box.set_ylabel('Production (pmp)')

#ax_production_hist=donneeABC_G_clean["Production (pmp)"].plot.hist(density=False, bins = 10, color = 'blue', edgecolor = 'black')
#ax_production_hist.set_xlabel("Production (pmp)")

Q1 = donneeABC_G_clean["Production (pmp)"].quantile(0.25)
Q3 = donneeABC_G_clean["Production (pmp)"].quantile(0.75)
IQR = Q3 - Q1

"Enlever les valeurs aberaantes (outliers)"

donneeABC_G_clean = donneeABC_G_clean[(donneeABC_G_clean["Production (pmp)"] > (Q1 - 1.5 * IQR)) & (donneeABC_G_clean["Production (pmp)"] < (Q3 + 1.5 * IQR))]

#ax_production_hist=donneeABC_G_clean["Production (pmp)"].plot.hist(density=False, bins = 10, color = 'blue', edgecolor = 'black')
#ax_production_hist.set_xlabel("Production (pmp)")

#ax_production = donneeABC_G_clean['Production (pmp)'].plot.box()
#ax_production.set_ylabel('Production (pmp)')



#ax_cout = donneeABC_G_clean['Coût ($/pmp)'].plot.box()
#ax_cout.set_ylabel('Coût ($/pmp)')



#ax_nbre_billots = donneeABC_G_clean['Nombre total de billots'].plot.box()
#ax_nbre_billots.set_ylabel('Nombre total de billots')


Q01 = donneeABC_G_clean["Nombre total de billots"].quantile(0.25)
Q03 = donneeABC_G_clean["Nombre total de billots"].quantile(0.75)
IQR1 = Q03 - Q01


#ax_nbre_billots = donneeABC_G_clean['Nombre total de billots'].plot.box()
#ax_nbre_billots.set_ylabel('Nombre total de billots')


"Enlever les valeurs aberaantes (outliers)"
donneeABC_G_clean = donneeABC_G_clean[(donneeABC_G_clean["Nombre total de billots"] > (Q01 - 1.5 * IQR1)) & (donneeABC_G_clean["Nombre total de billots"] < (Q03 + 1.5 * IQR1))]


#ax_nbre_billots = donneeABC_G_clean['Nombre total de billots'].plot.box()
#ax_nbre_billots.set_ylabel('Nombre total de billots')



"========================Étude de coolération ====================================="

"Q1: ................ "

donneeABC_G_clean['dates'] = donneeABC_G_clean['dates'] = pd.to_datetime(donneeABC_G_clean['dates'])

X=pd.get_dummies(donneeABC_G_clean)

MatriceR = X.corr()

#sns.heatmap(MatriceR, annot=False, center=0, xticklabels=True, yticklabels=True, square=True, cmap="RdYlGn_r")

#plt.figure(1)
#plt.plot(donneeABC_G_clean["Pins (# de billots)"], donneeABC_G_clean["Production (pmp)"],'o')
#plt.ylabel("Production (pmp)")
#plt.xlabel("Pins (# de billots)")
#plt.show()



#ax = donneeABC_G_clean.plot.hexbin(x="Pins (# de billots)", y="Production (pmp)", gridsize =8,sharex = False)
#ax.xlabel("Pins (# de billots)")
#ax.ylabel("Production (pmp)")
#ax.show()


#ax = sns.kdeplot(donneeABC_G_clean["Pins (# de billots)"], donneeABC_G_clean["Production (pmp)"], ax=ax)
#ax.set_xlabel("Pins (# de billots)")
#ax.set_ylabel("Production (pmp)")


#sns.kdeplot(donneeABC_G_clean["Pins (# de billots)"], donneeABC_G_clean["Production (pmp)"])



#graph = p9.ggplot(data= donneeABC_G_clean,mapping=p9.aes(x="Diamètre moyen des billots (po)", y="Production (pmp)",color ='Coût ($/pmp)'))

#print (graph + p9.geom_point())

#plt.figure(2)
#plt.plot(donneeABC_G_clean["Production (pmp)"], donneeABC_G_clean["Fournisseur"=='B'], 'o')
#plt.xlabel("Production (pmp)")
#plt.ylabel("Diamètre moyen des billots (po)")
#plt.show()



"============================== Monte Carlo Process =============================="




"Question 2 : Analyse de sensibilité (Monte Carlo)"

"Paramètres de la simulation"


couts = donneeABC_G_clean['Coût ($/pmp)'].values



productions = donneeABC_G_clean['Production (pmp)'].values

fournisseurs = donneeABC_G_clean['Fournisseur'] 


coutvarA = couts[fournisseurs=='A']

prodvarA = productions[fournisseurs=='A']



coutvarB = couts[fournisseurs=='B']

prodvarB = productions[fournisseurs=='B']


coutvarC = couts[fournisseurs=='C']

prodvarC = productions[fournisseurs=='C']



CoutPredA = []
ProdPredA = []

CoutPredB = []
ProdPredB = []


CoutPredC = []
ProdPredC = []


profitPred = []



L=10000
FracA = 0.3
FracB = 0.4
FracC = 0.3
PrixPMP= 1.0


for i in range(L):
    
    
    CoutA = random.choice(coutvarA)    
    ProdA = random.choice(prodvarA)
    
    CoutB = random.choice(coutvarB)
    
    ProdB = random.choice(prodvarB)
    
    
    CoutC = random.choice(coutvarC)
    
    ProdC = random.choice(prodvarC)
    
    
    Profit = FracA*ProdA*(PrixPMP - CoutA) + FracB*ProdA*(PrixPMP - CoutB) + FracC*ProdC*(PrixPMP - CoutC)
    
    CoutPredA.append(CoutA)
    ProdPredA.append(ProdA)
    CoutPredB.append(CoutB)
    ProdPredB.append(ProdB)
    CoutPredC.append(CoutC)
    ProdPredC.append(ProdC)
    profitPred.append(Profit)
    
    
    
    

CoutPredA=pd.DataFrame(CoutPredA)
ProdPredA=pd.DataFrame(ProdPredA)



CoutPredB=pd.DataFrame(CoutPredB)
ProdPredB=pd.DataFrame(ProdPredB)



CoutPredC=pd.DataFrame(CoutPredC)
ProdPredC=pd.DataFrame(ProdPredC)

profitPred = pd.DataFrame(profitPred)


ax = profitPred.plot.hist(density=False, bins = 10, color = 'blue', edgecolor = 'black')
ax.set_xlabel("Profits ($/jour) ")
StatsProfit=profitPred.describe()


intervalles_Profit = pd.cut(profitPred[0],10)

ProfitDist = intervalles_Profit.value_counts()/(profitPred.shape[0])

ProfitDist=ProfitDist.sort_index()


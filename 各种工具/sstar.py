import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
star = dd.read_csv('E://DLcode//dr5data//dr5_stellar.csv',delimiter='|')
star['snrsum'] = star.iloc[:,10:15].sum(axis=1)
star.compute()
print('done')
A = star[star['subclass'].str.contains(r'^A')]
A = A.compute()
A = A.sort_values(by=['snrsum'], axis=0, ascending=False)
A.to_csv('A_STAR.csv')

del A
print('doneA')

F = star[star['subclass'].str.contains(r'^F')]
F = F.compute()
F = F.sort_values(by=['snrsum'], axis=0, ascending=False)
F.to_csv('F_STAR.csv')
del F
print('doneF')
G = star[star['subclass'].str.contains(r'^G')]
G = G.compute()
G = G.sort_values(by=['snrsum'], axis=0, ascending=False)
G.to_csv('G_STAR.csv')
del G
print('doneG')
K = star[star['subclass'].str.contains(r'^K')]
K = K.compute()
K = K.sort_values(by=['snrsum'], axis=0, ascending=False)
K.to_csv('K_STAR.csv')
del K
print('doneK')

print('all done')

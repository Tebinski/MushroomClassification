"""
Alone in the woods (Kaggle) :
https://www.kaggle.com/akshay22071995/alone-in-the-woods-using-theil-s-u-for-survival
"""
#%%

import math
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    " compute the theils U coeficient of TWO LIST"
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    #p_x = [n/total_occurrences for n in x_counter.values()]
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

#-------------------------------------------------------

#%% Load files from kaggle
exec(open("download_kaggle_files.py").read())
df = pd.read_csv(r'./data/mushrooms.csv')

#%% md
""" Para usar Theils, tenemos que definir nuestra variable objetivo, 
y ver si existe relacion con las otras variables.

En nuestro caso la var obj es "class, que nos dice si la seta es venenosa " poisonous", 
o comestible (edible)"""

#%%
theilu = pd.DataFrame(index=['class'],columns=df.columns)
columns = df.columns
for j in range(0,len(columns)):
    u = theil_u(df['class'].tolist(),df[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)

#%% Representation
tsorted = theilu.sort_values(by='class', axis=1, ascending=False, na_position='last')
plt.figure(figsize=(20,3))
sns.heatmap(tsorted,annot=True,fmt='.2f')
plt.show()

#%% md
print(
    f" Theils U nos dice que podriamos determinar al {tsorted.iloc[0,1]:.03f} "
    f" si una seta es venenosa usando solos valores de *{tsorted.columns[1]}*.")

#%% Ahora tendriamos que ver qué olores nos indican venenoso o no

print(df.odor.value_counts())

plt.figure()
sns.set(rc={'figure.figsize':(15,8)})
ax=sns.countplot(x='odor',hue='class',data=df)
# añadimos esta parte para gestionar los nan
for p in ax.patches:
    patch_height = p.get_height()
    if np.isnan(patch_height):
        patch_height = 0
    ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.05, patch_height+10))
plt.show()

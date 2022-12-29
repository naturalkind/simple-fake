import numpy as np
import pandas as pd

#df = pd.read_csv("key_median.csv", encoding='utf8')

df = pd.read_csv("key_count.csv", encoding='utf8')
print (df)

#cols = df.columns.tolist()[1:]

#df_np = df.to_numpy()[:,1:]

#print (df_np.shape, len(cols))

#G = []
#for u in range(df_np.shape[0]):
#    S = np.sort(df_np[u])
#    AS = np.argsort(df_np[u])
#    #print (S[::-1], AS[::-1])
#    new_cols = []
#    for k in AS:
#        new_cols.append(cols[k])
#        print (k, cols[k])
#    G.append(S[::-1])

#AAA1 = np.array(G)
#print (AAA1.shape)    

#dataset1 = pd.DataFrame(data=AAA1[0:,0:], index=[i for i in range(AAA1.shape[0])], columns=new_cols)
#dataset1[new_cols] = dataset1[new_cols].replace(['0', 0], np.nan)
##dataset1.to_csv(f'key_count.csv')
#dataset1.to_csv(f'key_median.csv')
#print (dataset1)


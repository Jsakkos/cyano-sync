import pandas as pd
from glob import glob

f = glob(r'.\Data\*.fcs')
L = list()
for file in f:
    L.append(file.split('\\')[-1])
    
df = pd.DataFrame(L,columns=['Filename'])
df.to_excel('files.xlsx')
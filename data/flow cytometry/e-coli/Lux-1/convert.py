import pandas as pd
import os
cwd = os.getcwd()
print(cwd)

dataframe = pd.read_excel('LuxR updated.xlsx',sheet_name='LuxR')
sample_list = list()
for sample in dataframe.iterrows():
    print(sample)
    sample_list.append([sample[1]['Flow numbering']+ '.fcs',sample[1]['Sample name'].split()[1][:-1],sample[1]['Sample name'].split()[0][:-1],sample[1].Family])
    
import csv

with open("out.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(sample_list)
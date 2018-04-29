import sys
import os
import pandas as pd
from writeapriorifile import WriteAprioriFile
#print(sys.path)
os.chdir("..\\..\\..\\02450Toolbox_Python\\Scripts")
#print(os.getcwd())
from similarity import binarize2

data = pd.read_csv('..\\..\\Projects\\Project1\\Projekt3\\data.csv',header="infer")
X = data.as_matrix()
attributeNames = list(data)
data, newnames
 = binarize2(X,columnnames=attributeNames)
print(data) #ser godt ud

# Gem binæriseret data som en pik med et ordentligt navn
WriteAprioriFile(data,filename="FatBinarizedIndians.txt")

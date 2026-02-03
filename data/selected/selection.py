import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

input_file = "project/data/processed"
output_file = "project/data/selected"

df = pd.read_csv(
    f"{input_file}/processed.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

def scenario(df, scenario = 1):
    if scenario == 1:
        cols = ['log10(Sliding velocity (m/s))','Humidity','Ball hardness (GPa)','Load (N)','Temperature','Sp2/Sp3','DLC groupe','Film hardness (GPa)','H','Friction coefficient','log10(Wear rate)']
    elif scenario == 2:
        cols = ['log10(Sliding velocity (m/s))','Humidity','Ball hardness (GPa)','Load (N)','Temperature','Sp2/Sp3','DLC groupe','Film hardness (GPa)','Doped','H','Friction coefficient','log10(Wear rate)']
    elif scenario == 3:
        cols = ['log10(Sliding velocity (m/s))','Humidity','Ball hardness (GPa)','Load (N)','Temperature','Sp2/Sp3','DLC groupe','Film hardness (GPa)','H','O','Ti','Si','W','Mo','N','Cr','Ta','Ag','Ar','Al','Fe','Nb','Cu','F','S','B','Friction coefficient','log10(Wear rate)']
    else:
        raise ValueError(f'scenario {scenario} does not exist, choose 1 or 2')
    return df[cols]

df1 = scenario(df, scenario = 1)
df1["DLC groupe"] = df1["DLC groupe"].replace({
    "a-C": 0,
    "ta-C": 1,
    "a-C:H": 0,
    "ta-C:H": 1
})
df1["H"] = df1["H"].replace({True: 1,False: 0})
filename1 = f"{output_file}/Scenario1.csv"
df1.to_csv(filename1, sep=";", decimal=",", index=False, encoding="utf-8")

df2 = scenario(df, scenario = 2)
df2["DLC groupe"] = df2["DLC groupe"].replace({
    "a-C": 0,
    "ta-C": 1,
    "a-C:H": 0,
    "ta-C:H": 1
})
df2["H"] = df2["H"].replace({True: 1,False: 0})
df2["Doped"] = df2["Doped"].replace({True: 1,False: 0})
filename2 = f"{output_file}/Scenario2.csv"
df2.to_csv(filename2, sep=";", decimal=",", index=False, encoding="utf-8")

df3 = scenario(df, scenario = 3)
df3["DLC groupe"] = df["DLC groupe"].replace({
    "a-C": 0,
    "ta-C": 1,
    "a-C:H": 0,
    "ta-C:H": 1
})
for el in ['H','O','Ti','Si','W','Mo','N','Cr','Ta','Ag','Ar','Al','Fe','Nb','Cu','F','S','B']:
    df3[el] = df3[el].replace({True: 1, False: 0})

filename3 = f"{output_file}/Scenario3.csv"
df3.to_csv(filename3, sep=";", decimal=",", index=False, encoding="utf-8")
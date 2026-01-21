import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# Chargement des données depuis le fichier CSV avec les paramètres régionaux français
df1 = pd.read_csv(
    "data/DLC_database_from_pdf_1_init.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

df2 = pd.read_csv(
    "data/DLC_database_from_pdf_2_init.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

mapping_df = pd.read_csv(
    "data/dlc_types_mapping.csv",   
    sep=";",
    decimal=",",
    encoding="utf-8"
)

# remplacer tous les #DIV/0! et #VALEUR! par du vide dans le dataframe
df1.replace('#DIV/0!', np.nan, inplace=True)
df1.replace('#VALEUR!', np.nan, inplace=True)


#ajouter Ne à df1
df1.insert(df1.columns.get_loc('B') + 1, 'Ne', np.nan)


#ajouter C_content et H_content à df1
df1.insert(df1.columns.get_loc('DLC type') + 1, 'H_content', df1['H'])
df1.insert(df1.columns.get_loc('DLC type') + 1, 'C_content', df1['C'])


# transformer df1 pour ne garder que la présence ou non des éléments et les vides en NA
elements = ['O','Ti','Si','H','W','Mo','N','Cr','Ta','Ag','Ar','Al','Fe','Nb','Cu','F','S','B','Ne']

df1[elements] = (
    df1[elements]
        .apply(pd.to_numeric, errors='coerce')  # vides -> NaN
        .where(lambda x: x.isna(), lambda x: x > 0)
)

df1[elements] = df1[elements].astype("boolean")


#Trasformer maping pour ne mettre que des booléens
mapping_df[elements + ['Doped']] = (
    mapping_df[elements + ['Doped']]
        .apply(pd.to_numeric, errors='coerce')   # vides -> NaN
        .where(lambda x: x.isna(), lambda x: x > 0)
)

mapping_df[elements + ['Doped']] = mapping_df[elements + ['Doped']].astype("boolean")


# Détection du dopage
elements.remove('H')

has_data = df1[elements].notna().any(axis=1)     # au moins une info renseignée
has_dopant = (df1[elements] > 0).any(axis=1)     # au moins un dopant présent

df1.insert(
    df1.columns.get_loc('H_content') + 1,
    'Doped',
    has_dopant.where(has_data)                    # garde NaN si aucune donnée
)

df1['Doped'] = df1['Doped'].astype('boolean')


#ajouter les informations de mapping à df1
df1.insert(df1.columns.get_loc('DLC type') + 1, 'DLC groupe',np.nan)

elements.append('H')
df = df1.merge(
    mapping_df[['DLC type'] + elements + ['Doped', 'DLC groupe']],
    on='DLC type',
    how='left',
    suffixes=('', '_map'))

for el in elements + ['Doped', 'DLC groupe']:
    df[el] = df[el].fillna(df[f'{el}_map'])

df.drop(columns=[f'{el}_map' for el in elements + ['Doped', 'DLC groupe']], inplace=True)

elements.remove('H')
for el in elements :
    df.loc[(~df['Doped']) & df[el].isna(), el] = False

#supprimer les données incohérentes (pour l'instant)
mask_incoherent = (
    df['DLC groupe'].notna() &
    df['H_content'].notna() &
    (
        ((df['DLC groupe'] == 'ta-C')   & df['H'])    |
        ((df['DLC groupe'] == 'a-C:H')  & (~df['H'])) |
        ((df['DLC groupe'] == 'a-C')    & df['H'])    |
        ((df['DLC groupe'] == 'ta-C:H') & (~df['H']))
    )
)
df = df.loc[~mask_incoherent]


# compléter DLC groupe avec les valeurs de Hardness et H_content si vide

Hardness_mask_aC = (df['Film hardness (GPa)'] < 28) & ((df['H']==False) | df['H'].isna()) & df['DLC groupe'].isna()
Hardness_mask_taC = (df['Film hardness (GPa)'] >= 28) & ((df['H']==False) | df['H'].isna()) & df['DLC groupe'].isna()
Hardness_mask_aCH = (df['Film hardness (GPa)'] < 30) & (df['H']) & df['DLC groupe'].isna()
Hardness_mask_taCH = (df['Film hardness (GPa)'] >= 30) & (df['H']) & df['DLC groupe'].isna()

df.loc[Hardness_mask_aC, 'DLC groupe'] = 'a-C'
df.loc[Hardness_mask_taC, 'DLC groupe'] = 'ta-C'
df.loc[Hardness_mask_aCH, 'DLC groupe'] = 'a-C:H'
df.loc[Hardness_mask_taCH, 'DLC groupe'] = 'ta-C:H'


Elasticity_mask_aC = (df['Film elastic modulus (GPa)'] < 190) & ((~df['H']) | df['H'].isna()) & df['DLC groupe'].isna()
Elasticity_mask_taC = (df['Film elastic modulus (GPa)'] > 210) & ((~df['H']) | df['H'].isna()) & df['DLC groupe'].isna()
Elasticity_mask_aCH = (df['Film elastic modulus (GPa)'] < 175) & (df['H']) & df['DLC groupe'].isna()
Elasticity_mask_taCH = (df['Film elastic modulus (GPa)'] > 210) & (df['H']) & df['DLC groupe'].isna()

df.loc[Elasticity_mask_aC, 'DLC groupe'] = 'a-C'
df.loc[Elasticity_mask_taC, 'DLC groupe'] = 'ta-C'
df.loc[Elasticity_mask_aCH, 'DLC groupe'] = 'a-C:H'
df.loc[Elasticity_mask_taCH, 'DLC groupe'] = 'ta-C:H'


#compléter H_content du groupe avec la valeure médiane des H_content du groupe non nuls lorsque H_content vide et H == 1
H_mediane_aCH = df.loc[((df['DLC groupe']=='a-C:H') & df['H']), 'H_content'].median()
mediane_taCH = df.loc[((df['DLC groupe']=='ta-C:H') & df['H']), 'H_content'].median()

mask_aCH = df['H_content'].isna() & (df['H']) & df['DLC groupe']=='a-C:H'
df.loc[mask_aCH, 'H_content'] = H_mediane_aCH

mask_taCH = df['H_content'].isna() & (df['H']) & df['DLC groupe']=='ta-C:H'
df.loc[mask_aCH, 'H_content'] = mediane_taCH

mask2 = df['H_content'].isna() & (~df['H'])
df.loc[mask2, 'H_content'] = 0


#suppress irrelevant columns
columns_to_suppress = ['C', 'Process type','Current (A)']
df.drop(columns=columns_to_suppress, inplace=True)


# Filtrage des données selon les critères spécifiés
df_filtered = df[
    (
        ((df['Sliding velocity (m/s)'] <= 2) | df['Sliding velocity (m/s)'].isna()) &
        ((df['Friction coefficient'] <= 1) | df['Friction coefficient'].isna()) &
        (((df['Wear rate'] <= 1e-4) & (df['Wear rate'] >= 1e-11)) | df['Wear rate'].isna()) &
        ((df['Sp2/Sp3'] <= 2) | df['Sp2/Sp3'].isna())
    )
].copy()

print(len(df_filtered))

# 10^3 pout affichage Hertz pressure
for col in ['Hertz pressure']:
    df_filtered[col] = pd.to_numeric(
    df_filtered[col].str.replace(',', '.'),  # convertir "," en "."
    errors='coerce'  # mettra NaN pour tout ce qui n'est pas convertible
)
    df_filtered['Hertz pressure (Gpa)'] = df_filtered['Hertz pressure'] * 1e3
    df_filtered.drop(columns='Hertz pressure', inplace=True)


# log transformation des données pour certaines colonnes
for col in ['Sliding velocity (m/s)', 'Wear rate', 'Rq (nm)', 'Wear Volume V', 'Sliding distance (m)']:
    new_col = f'log10({col})'
    df_filtered[new_col] = np.log10(df_filtered[col].replace(0, np.nan))
    df_filtered.drop(columns=col, inplace=True)



# Sauvegarde du DataFrame modifié dans un nouveau fichier CSV
df_filtered.to_csv(
    "data/cleaned_dataset.csv",
    sep=";",
    decimal=",",
    encoding="utf-8",
    index=False
)
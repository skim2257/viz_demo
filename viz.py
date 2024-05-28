import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

# vanilla pair plot
df = pd.read_excel("data/RADCURE-DA-CLINICAL-2.xlsx")
print(df.shape)
print(df.describe().to_csv("data/describe.csv"))

fig = sns.pairplot(df, 
                   diag_kind='kde',
                   height=5)

fig.savefig("pairplots/test.png")

# pair plot with variable as hue
vars = ["HPV", "Smoking Status", "Sex"]
for var in vars:
    fig = sns.pairplot(df, 
                       diag_kind='kde',
                       hue=var,
                       height=5)

    fig.savefig(f"pairplots/{var}.png")

def prep_for_pie(df, label):
    # df[value] = pd.to_numeric(df[value])

    data = df.groupby(label).size().sort_values(ascending=False)

    labels = data.index.tolist()
    values = data.values.tolist()
    
    return labels, values

# categorical plots
vars = ["Sex", "ECOG PS", "Ds Site", "Smoking Status", "T", "N", "M", "Stage", "HPV", "Tx Modality", "Chemo", "ContrastEnhanced"]
df_counts = pd.DataFrame(columns=["counts"])
for var in vars:
    fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=150)
    sns.set(style="white")
    labels, values = prep_for_pie(df, var)

    # save counts to dataframe
    df_counts = pd.concat([df_counts, 
                           pd.DataFrame(index=["", var], data=["", ""], columns=["counts"]), 
                           pd.DataFrame(index=labels, data=values, columns=["counts"])])

    # only write % if big enough
    def autopct(pct):
        return ('%1.1f%%' % pct) if pct > 3.5 else ''

    ax[0].pie(values, 
              labels=labels, 
              autopct=autopct, 
              startangle=90,
              counterclock=False,
              colors=plt.cm.Set2.colors) # 90 = 12 o'clock, 0 = 3 o'clock, 180 = 9 o'clock
    ax[0].set_title(f"{var} Distribution")

    sns.violinplot(x=var, y="Age", data=df, ax=ax[1])
    ax[1].set_title(f"{var} vs Age")

    fig.savefig(f"multiplots/{var}_multiplots.png")

df_counts.to_csv("data/counts.csv")


# km curves
df['days_fu'] = pd.to_datetime(df['Last FU']) - pd.to_datetime(df['RT Start'])
df['days_Status'] = pd.to_datetime(df['Date of Death']) - pd.to_datetime(df['RT Start'])
df['days_Local'] = pd.to_datetime(df['Date Local']) - pd.to_datetime(df['RT Start'])
df['days_Regional'] = pd.to_datetime(df['Date Regional']) - pd.to_datetime(df['RT Start'])
df['days_Distant'] = pd.to_datetime(df['Date Distant']) - pd.to_datetime(df['RT Start'])

endpoints = ["Status", "Local", "Regional", "Distant"]
tnms = {"T": ['T0', 'T1', 'T1a', 'T1b', 'T2', 'T3', 'T4', 'T4a', 'T4b'], 
        "N": ['N0', 'N1', 'N2', 'N2a', 'N2b', 'N2c', 'N3'], 
        "Stage": [0, 'I', 'II', 'III', 'IVA', 'IVB']}

df_new = pd.DataFrame(index=df.index)
df_new["T"] = df["T"]
df_new["N"] = df["N"]
df_new["Stage"] = df["Stage"]
for endpoint in endpoints:
    fig, ax = plt.subplots(1, 3, figsize=(16,4), dpi=150)
    for n, tnm in enumerate(tnms):
        for value in tnms[tnm]:
            df_sub = df[df[tnm] == value]
            kmf = KaplanMeierFitter()
            T = df_sub[f"days_{endpoint}"].dt.days
            T.fillna(df['days_fu'].dt.days, inplace=True)
            
            if endpoint == "Status":
                E = df_sub[endpoint].map({"Alive": 0, "Dead": 1})
            else:
                E = df_sub[endpoint].notnull().astype(int)
                        
            kmf.fit(T, event_observed=E, label=value)
            kmf.plot(ci_show=False, ax=ax[n])
        df_new[endpoint]           = E
        df_new[f"{endpoint}_days"] = T
        if endpoint == "Status":
            ax[n].set_title(f"Overall Survival vs {tnm}")
        else:
            ax[n].set_title(f"{endpoint} Spread vs {tnm}")
        fig.savefig(f"km_curves/km_curves_{endpoint} vs tnms.png")

df_new.to_csv("data/km_data.csv")

# Advanced-Python
### Advance Python Assignment [Major]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns    

df=pd.read_csv("C:/Users/Patricia/Desktop/Dataset/honeyproduction 1998-2021.csv")
df.head()

df.shape

df.tail()

 df.dtypes

df.isnull().sum()

### 1. How has honey production yield changed from 1998 to 2021?

data_chan_by_year=df.groupby('year')['totalprod'].mean().round()
print(data_chan_by_year)


plt.figure(figsize=(9,5))
plt.plot(data_chan_by_year.index,data_chan_by_year.values,linewidth=0.7,marker="o",label="total production lb")
plt.xlabel("year")
plt.ylabel("Honey production yield(pounds)")
plt.xticks(data_chan_by_year.index,rotation=90)
plt.title("Honey production yield changed from 1998 to 2021")
plt.legend()
plt.show()

### 2. Over time, what are the major production trends across the states?

df['State'].nunique()

sns.set(rc={"figure.figsize":(14,7)})
sns.set_style('darkgrid')
sns.set_context('poster',font_scale=0.5,rc={'grid.linewidth':0.7})
sns.set_palette('coolwarm')
sns.lineplot(x='year',y='totalprod',data=df,hue='State')
plt.xlabel("Year",size=13)
plt.ylabel("Production trends across the state",size=14)
plt.legend(bbox_to_anchor=(1,1))
plt.xticks(df['year'],rotation=90)
plt.title("Production trens across the states over time",color='g',size=20)
plt.show()

### 3. Does the data show any trends in terms of the number of honey producing colonies and yield per colony before 2006, which was when concern over Colony Collapse Disorder spread nationwide?

data_before_2006=df[df['year']<2006]
data_before_2006

plt.figure(figsize=(10,8))
plt.plot(data_before_2006['year'],data_before_2006['numcol'],linewidth=0.6,marker="o",label="No. of honey producing colony before 2006")
plt.plot(data_before_2006['year'],data_before_2006['yieldpercol'],linewidth=0.6,marker="^",label="Yield per colony before 2006")
plt.xlabel("Year",size=10)
plt.ylabel("number of honey producing colonies and yield per colony before 2006",size=10)
plt.legend(bbox_to_anchor=(1,1))
plt.xticks(data_before_2006['year'],rotation=90)
plt.title("Trends in terms of the number of honey producing colonies and yield per colony before 2006",color='blue',size=12)
plt.show()

### 4. Are there any patterns that can be observed between total honey production and value of production every year?

hon_prod=df.groupby('year')[['totalprod','prodvalue']].sum()
hon_prod

plt.figure(figsize=(14,10))
plt.plot(hon_prod.index,hon_prod['totalprod'],linewidth=0.6,marker="o",label="Total production year wise")
plt.plot(hon_prod.index,hon_prod['prodvalue'],linewidth=0.6,marker="^",label="Production value year wise")
plt.xlabel("Year",size=18)
plt.ylabel("Total Honey Production v/s Production Value",size=12)
plt.legend()
plt.xticks(hon_prod.index,rotation=90)
plt.title("Patterns between total honey production and value of production every year",color='red',size=12)
plt.show()

### 5. How has the value of production, which in some sense could be tied to demand, changed every year?

df['consumption']=df['totalprod']-df['stocks']

df_tied=df.groupby('year')[['prodvalue','consumption']].mean()
df_tied

sns.set(rc={"figure.figsize":(12,4)})
sns.set_style('darkgrid')
sns.set_context('poster',font_scale=0.5,rc={'grid.linewidth':0.7})
sns.set_palette('tab10')
sns.lineplot(data=df_tied, x='year',y='prodvalue',marker="o",label='prodvalue')
sns.lineplot(data=df_tied,x='year',y='consumption',marker="^",label='consumption')
plt.xlabel("Year",size=14)
plt.ylabel("Values for prodvalue and consumption",size=15)
plt.xticks(df['year'],rotation=90)
plt.legend()
plt.title("Production Value tied up with demand year wise",color='maroon',size=20)
plt.show()

### 6. Constructs the related plots using Seaborn and Matplot apply customization and derive insights from the visualization.

plt.figure(figsize=(20,20))
sns.barplot(x="totalprod",y="State",data=df.sort_values("totalprod",ascending=False),label="Total Production",color="b",ci=None)
sns.barplot(x="stocks",y="State",data=df.sort_values("totalprod",ascending=False),label="Stocks",color="r",ci=None)
plt.legend(ncol=2,loc="lower right",frameon=True)
plt.show()


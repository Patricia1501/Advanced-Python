# Advanced-Python

### Advance Python 
### Honey Production in USA(1998-2021)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns    
```


```python
df=pd.read_csv("C:/Users/Patricia/Desktop/Dataset/honeyproduction 1998-2021.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>numcol</th>
      <th>yieldpercol</th>
      <th>totalprod</th>
      <th>stocks</th>
      <th>priceperlb</th>
      <th>prodvalue</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>16000.0</td>
      <td>71</td>
      <td>1136000.0</td>
      <td>159000.0</td>
      <td>0.72</td>
      <td>818000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arizona</td>
      <td>55000.0</td>
      <td>60</td>
      <td>3300000.0</td>
      <td>1485000.0</td>
      <td>0.64</td>
      <td>2112000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arkansas</td>
      <td>53000.0</td>
      <td>65</td>
      <td>3445000.0</td>
      <td>1688000.0</td>
      <td>0.59</td>
      <td>2033000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>California</td>
      <td>450000.0</td>
      <td>83</td>
      <td>37350000.0</td>
      <td>12326000.0</td>
      <td>0.62</td>
      <td>23157000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Colorado</td>
      <td>27000.0</td>
      <td>72</td>
      <td>1944000.0</td>
      <td>1594000.0</td>
      <td>0.70</td>
      <td>1361000.0</td>
      <td>1998</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (985, 8)




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>numcol</th>
      <th>yieldpercol</th>
      <th>totalprod</th>
      <th>stocks</th>
      <th>priceperlb</th>
      <th>prodvalue</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>980</th>
      <td>Virginia</td>
      <td>6000.0</td>
      <td>40</td>
      <td>240000.0</td>
      <td>79000.0</td>
      <td>8.23</td>
      <td>1975000.0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>981</th>
      <td>Washington</td>
      <td>96000.0</td>
      <td>32</td>
      <td>3072000.0</td>
      <td>1206000.0</td>
      <td>2.52</td>
      <td>7741000.0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>982</th>
      <td>West Virginia</td>
      <td>6000.0</td>
      <td>43</td>
      <td>258000.0</td>
      <td>136000.0</td>
      <td>4.80</td>
      <td>1238000.0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>983</th>
      <td>Wisconsin</td>
      <td>42000.0</td>
      <td>47</td>
      <td>1974000.0</td>
      <td>750000.0</td>
      <td>2.81</td>
      <td>5547000.0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>984</th>
      <td>Wyoming</td>
      <td>38000.0</td>
      <td>58</td>
      <td>2204000.0</td>
      <td>242000.0</td>
      <td>2.07</td>
      <td>4562000.0</td>
      <td>2021</td>
    </tr>
  </tbody>
</table>
</div>




```python
 df.dtypes
```




    State           object
    numcol         float64
    yieldpercol      int64
    totalprod      float64
    stocks         float64
    priceperlb     float64
    prodvalue      float64
    year             int64
    dtype: object




```python
df.isnull().sum()
```




    State          0
    numcol         0
    yieldpercol    0
    totalprod      0
    stocks         0
    priceperlb     0
    prodvalue      0
    year           0
    dtype: int64



### 1. How has honey production yield changed from 1998 to 2021?


```python
data_chan_by_year=df.groupby('year')['totalprod'].mean().round()
print(data_chan_by_year)

```

    year
    1998    5105093.0
    1999    4706674.0
    2000    5106000.0
    2001    4221545.0
    2002    3892386.0
    2003    4122091.0
    2004    4456805.0
    2005    4243146.0
    2006    3761902.0
    2007    3600512.0
    2008    3974927.0
    2009    3626700.0
    2010    4382350.0
    2011    3680025.0
    2012    3522675.0
    2013    3800103.0
    2014    4421650.0
    2015    3884400.0
    2016    4008925.0
    2017    3654125.0
    2018    3773725.0
    2019    3887600.0
    2020    3655475.0
    2021    3127925.0
    Name: totalprod, dtype: float64
    


```python
plt.figure(figsize=(9,5))
plt.plot(data_chan_by_year.index,data_chan_by_year.values,linewidth=0.7,marker="o",label="total production lb")
plt.xlabel("year")
plt.ylabel("Honey production yield(pounds)")
plt.xticks(data_chan_by_year.index,rotation=90)
plt.title("Honey production yield changed from 1998 to 2021")
plt.legend()
plt.show()
```


    
![png](output_9_0.png)
    


### 2. Over time, what are the major production trends across the states?


```python
df['State'].nunique()
```




    44




```python
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
```


    
![png](output_12_0.png)
    


### 3. Does the data show any trends in terms of the number of honey producing colonies and yield per colony before 2006, which was when concern over Colony Collapse Disorder spread nationwide?


```python
data_before_2006=df[df['year']<2006]
data_before_2006
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>numcol</th>
      <th>yieldpercol</th>
      <th>totalprod</th>
      <th>stocks</th>
      <th>priceperlb</th>
      <th>prodvalue</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>16000.0</td>
      <td>71</td>
      <td>1136000.0</td>
      <td>159000.0</td>
      <td>0.72</td>
      <td>818000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arizona</td>
      <td>55000.0</td>
      <td>60</td>
      <td>3300000.0</td>
      <td>1485000.0</td>
      <td>0.64</td>
      <td>2112000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arkansas</td>
      <td>53000.0</td>
      <td>65</td>
      <td>3445000.0</td>
      <td>1688000.0</td>
      <td>0.59</td>
      <td>2033000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>California</td>
      <td>450000.0</td>
      <td>83</td>
      <td>37350000.0</td>
      <td>12326000.0</td>
      <td>0.62</td>
      <td>23157000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Colorado</td>
      <td>27000.0</td>
      <td>72</td>
      <td>1944000.0</td>
      <td>1594000.0</td>
      <td>0.70</td>
      <td>1361000.0</td>
      <td>1998</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Virginia</td>
      <td>8000.0</td>
      <td>37</td>
      <td>296000.0</td>
      <td>59000.0</td>
      <td>2.20</td>
      <td>651000.0</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>339</th>
      <td>Washington</td>
      <td>51000.0</td>
      <td>55</td>
      <td>2805000.0</td>
      <td>1935000.0</td>
      <td>1.01</td>
      <td>2833000.0</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>340</th>
      <td>West Virginia</td>
      <td>8000.0</td>
      <td>51</td>
      <td>408000.0</td>
      <td>102000.0</td>
      <td>1.29</td>
      <td>526000.0</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Wisconsin</td>
      <td>64000.0</td>
      <td>83</td>
      <td>5312000.0</td>
      <td>2922000.0</td>
      <td>1.14</td>
      <td>6056000.0</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Wyoming</td>
      <td>40000.0</td>
      <td>56</td>
      <td>2240000.0</td>
      <td>291000.0</td>
      <td>0.89</td>
      <td>1994000.0</td>
      <td>2005</td>
    </tr>
  </tbody>
</table>
<p>343 rows × 8 columns</p>
</div>




```python
plt.figure(figsize=(10,8))
plt.plot(data_before_2006['year'],data_before_2006['numcol'],linewidth=0.6,marker="o",label="No. of honey producing colony before 2006")
plt.plot(data_before_2006['year'],data_before_2006['yieldpercol'],linewidth=0.6,marker="^",label="Yield per colony before 2006")
plt.xlabel("Year",size=10)
plt.ylabel("number of honey producing colonies and yield per colony before 2006",size=10)
plt.legend(bbox_to_anchor=(1,1))
plt.xticks(data_before_2006['year'],rotation=90)
plt.title("Trends in terms of the number of honey producing colonies and yield per colony before 2006",color='blue',size=12)
plt.show()
```


    
![png](output_15_0.png)
    


### 4. Are there any patterns that can be observed between total honey production and value of production every year?


```python
hon_prod=df.groupby('year')[['totalprod','prodvalue']].sum()
hon_prod
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>totalprod</th>
      <th>prodvalue</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1998</th>
      <td>219519000.0</td>
      <td>146091000.0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>202387000.0</td>
      <td>123657000.0</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>219558000.0</td>
      <td>131568000.0</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>185748000.0</td>
      <td>132282000.0</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>171265000.0</td>
      <td>227302000.0</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>181372000.0</td>
      <td>252079000.0</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>182729000.0</td>
      <td>197307000.0</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>173969000.0</td>
      <td>160793000.0</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>154238000.0</td>
      <td>157924000.0</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>147621000.0</td>
      <td>161356000.0</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>162972000.0</td>
      <td>229992000.0</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>145068000.0</td>
      <td>213920000.0</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>175294000.0</td>
      <td>278370000.0</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>147201000.0</td>
      <td>258688000.0</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>140907000.0</td>
      <td>280725000.0</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>148204000.0</td>
      <td>315118000.0</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>176866000.0</td>
      <td>384483000.0</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>155376000.0</td>
      <td>322505000.0</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>160357000.0</td>
      <td>325557000.0</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>146165000.0</td>
      <td>317502000.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>150949000.0</td>
      <td>324883000.0</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>155504000.0</td>
      <td>297120000.0</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>146219000.0</td>
      <td>297177000.0</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>125117000.0</td>
      <td>314413000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(14,10))
plt.plot(hon_prod.index,hon_prod['totalprod'],linewidth=0.6,marker="o",label="Total production year wise")
plt.plot(hon_prod.index,hon_prod['prodvalue'],linewidth=0.6,marker="^",label="Production value year wise")
plt.xlabel("Year",size=18)
plt.ylabel("Total Honey Production v/s Production Value",size=12)
plt.legend()
plt.xticks(hon_prod.index,rotation=90)
plt.title("Patterns between total honey production and value of production every year",color='red',size=12)
plt.show()
```


    
![png](output_18_0.png)
    


### 5. How has the value of production, which in some sense could be tied to demand, changed every year?


```python
df['consumption']=df['totalprod']-df['stocks']
```


```python
df_tied=df.groupby('year')[['prodvalue','consumption']].mean()
df_tied
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prodvalue</th>
      <th>consumption</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1998</th>
      <td>3.397465e+06</td>
      <td>3.231488e+06</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>2.875744e+06</td>
      <td>2.883651e+06</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>3.059721e+06</td>
      <td>3.130279e+06</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>3.006409e+06</td>
      <td>2.749636e+06</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>5.165955e+06</td>
      <td>3.002000e+06</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>5.729068e+06</td>
      <td>3.198932e+06</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>4.812366e+06</td>
      <td>2.969463e+06</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>3.921780e+06</td>
      <td>2.726390e+06</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>3.851805e+06</td>
      <td>2.292756e+06</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>3.935512e+06</td>
      <td>2.322341e+06</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>5.609561e+06</td>
      <td>2.731122e+06</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>5.348000e+06</td>
      <td>2.693650e+06</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>6.959250e+06</td>
      <td>3.262425e+06</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>6.467200e+06</td>
      <td>2.766275e+06</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>7.018125e+06</td>
      <td>2.731125e+06</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>8.079949e+06</td>
      <td>2.826410e+06</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>9.612075e+06</td>
      <td>3.396900e+06</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>8.062625e+06</td>
      <td>2.838600e+06</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>8.138925e+06</td>
      <td>2.988000e+06</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>7.937550e+06</td>
      <td>2.894300e+06</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>8.122075e+06</td>
      <td>3.054300e+06</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>7.428000e+06</td>
      <td>2.870825e+06</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>7.429425e+06</td>
      <td>2.670175e+06</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>7.860325e+06</td>
      <td>2.548250e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


    
![png](output_22_0.png)
    


### 6. Constructs the related plots using Seaborn and Matplot apply customization and derive insights from the visualization.


```python
plt.figure(figsize=(20,20))
sns.barplot(x="totalprod",y="State",data=df.sort_values("totalprod",ascending=False),label="Total Production",color="b",ci=None)
sns.barplot(x="stocks",y="State",data=df.sort_values("totalprod",ascending=False),label="Stocks",color="r",ci=None)
plt.legend(ncol=2,loc="lower right",frameon=True)
plt.show()
```


    
![png](output_24_0.png)
    



```python

```

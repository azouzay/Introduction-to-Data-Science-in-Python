
# Intro parts (three parts)-  Introduction to Pandas then module Project 

# ## Part 1- working with the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on
# https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table [All Time Olympic Games Medals 
# 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.



import pandas as pd
import numpy as np

# loading csv file 
df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

# renaming columns

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')

df.head() # looking at first rows of the dataframe

df.iloc[90] # using iloc to check specific row i.e iloc[0] gives medals for Afganistan

#country that won the most gold medals in summer games
MaxSummerGold= df['Gold'].idxmax() 


#country that had the biggest difference between their summer and winter gold medal counts
MaxDifferenceGold=(df['Gold']-df['Gold.1']).abs().idxmax() 

# country that has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count? 

df['RatioGold']=0
df['RatioGold'][(df['Gold']>0) & (df['Gold.1']>0)] = ((df['Gold']-df['Gold.1'])/df['Gold.2']).abs()         
MaxRatioGold= df['RatioGold'].idxmax()





# creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.

df['Points']=df['Gold.2']*3+ df['Silver.2']*2+ df['Bronze.2']



# ## Part 2- working with census data from the [United States Census Bureau](http://www.census.gov).
#Counties are political and geographic subdivisions of states in the United States.
#This dataset contains population data for counties and states in the US from 2010 to 2015.
#[See this document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2015/co-est2015-alldata.pdf)for a description of the variable names.
# 
#loading data

census_df = pd.read_csv('census.csv')
census_df.head()

# state with most counties in it
State_big= census_df.groupby('STNAME').CTYNAME.nunique().argmax()



# **Only looking at the three most populous counties for each state**
# what are the three most populous states (in order of highest population to lowest population)
df_1= census_df[(census_df['SUMLEV']==50)].groupby('STNAME').CENSUS2010POP.nlargest(3)
ThreeStates=df_1.groupby(level=0, group_keys=False).sum().nlargest(3)
list(ThreeStates.index.values) 

# Which county has had the largest absolute change in population within the period 2010-2015?
df1 = census_df[(census_df['SUMLEV']==50)].filter(['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015'], axis=1)
census_df['result']= (df1.apply( max, axis=1 )-df1.apply( min, axis=1 )) 
LargestState= str(census_df.loc[census_df['result'].argmax(),'CTYNAME'])

# Creatig a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
census_df['query']=0
census_df['query'][(census_df['REGION'].isin([1,2])) & (census_df['CTYNAME'].str.startswith('Washington')) &(census_df['POPESTIMATE2015'] > census_df['POPESTIMATE2014'])]=1
census_df[['CTYNAME','STNAME']][(census_df['query']==1)]





### Part 3- More Pandas

#The idea is to join data from three files to create dataframe of top 15 countries in term of journal publication 
#([Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology] http://www.scimagojr.com/countryrank.php?category=2102)
#from the file `scimagojr-3.xlsx`) which ranks countries based on their journal contributions in the aforementioned area.
# In addition to `scimagojr-3.xlsx`, data will be loaded also from the file `Energy Indicators.xls`, which is a list of indicators of
#[energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations]
#(http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013
# and GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from
#[World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD)

import pandas as pd
import numpy as np

#Load the energy data from the file `Energy Indicators.xls` and exclude footer, header and unneccessary columns
# change columns labels to ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'] and convert `Energy Supply` to gigajoules
# convert missing data as `np.NaN` values

energyxls = pd.ExcelFile("Energy Indicators.xls") 
energy = pd.read_excel(energyxls,  usecols = [2,3,4,5],skiprows=17, skipfooter=38)
energy.columns =['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x: x * 1000000)
energy[['Energy Supply','Energy Supply per Capita','% Renewable']] = energy[['Energy Supply','Energy Supply per Capita','% Renewable']].apply(pd.to_numeric, errors='coerce')
          
# Rename the following countries "South Korea", "United States of America", "United Kingdom" and "Hong Kong"
energy['Country']=energy['Country'].replace('China, Hong Kong Special Administrative Region3','Hong Kong')
energy['Country']=energy['Country'].replace('China, Macao Special Administrative Region4','Macao')
energy['Country']=energy['Country'].replace(['United States of America20','Republic of Korea'],['United States','South Korea'])
energy['Country']=energy['Country'].replace('United Kingdom of Great Britain and Northern Ireland19','United Kingdom')

# Clean countries names and remove numbers and parenthesis
energy['Country'] = energy['Country'].str.replace("^([a-zA-Z]+(?:\s+[a-zA-Z]+)*).*", r"\1")

# Load the GDP data from the file `world_bank.csv` and rename the following countries: "South Korea", "Iran", "Hong Kong"
GDP = pd.read_csv('world_bank.csv', skiprows=4)
GDP['Country Name']=GDP['Country Name'].replace('Korea, Rep.','South Korea')
GDP['Country Name']=GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
GDP['Country Name']=GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')

# Load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology] from the file `scimagojr-3.xlsx`,
ScimEn = pd.read_excel('scimagojr-3.xlsx')

# Joining the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names).
Merge= pd.merge((pd.merge(ScimEn, energy, how='left', on='Country')),GDP, how='left', left_on='Country', right_on ='Country Name', left_index=True)

#Keep only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).      
Top15=Merge.loc[Merge['Rank'].isin(list(range(1,16)))]
Top15.set_index('Country', inplace=True)
Top15_selection=Top15[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]


# Before reducing the merge to Top 15, how many entries were lost?
Merge_in= pd.merge((pd.merge(ScimEn, energy, how='inner', on='Country')),GDP, how='inner', left_on='Country', right_on ='Country Name', left_index=True)
Merge_out= pd.merge((pd.merge(ScimEn, energy, how='outer', on='Country')),GDP, how='outer', left_on='Country', right_on ='Country Name', left_index=True)

LostOut= np.asscalar(np.int16(len(Merge_out.index) -  len(Merge_in.index))) 

## Getting some values from Top15 dataframe

# average GDP sorted in descending order.
avgGDP = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1).sort_values(ascending =False)

# GDP change over the 10 year span for the country with the 6th largest average GDP
SixthGDP= avgGDP.index[5]
UK_range=Top15.ix[SixthGDP][['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
UK_change=UK_range['2015']-UK_range['2006']

# Mean of `Energy Supply per Capita`
meanESC= Top15['Energy Supply per Capita'].mean()

# Country with the maximum % Renewable and their percentage
MaxRenewable= (Top15['% Renewable'].idxmax(),Top15['% Renewable'].max())
 
# Maximum value of ratio of Self-Citations to Total Citations and corresponding country 
Top15['ratio_citation']= Top15['Self-citations']/Top15['Citations']
MaxratioCitations= (Top15['ratio_citation'].idxmax(),Top15['ratio_citation'].max())

# The third most populous country according to an estimate using Energy Supply and Energy Supply per capita
Top15['Pop_estimate']= Top15['Energy Supply']/Top15['Energy Supply per Capita'].sort_values()
ThirdEstimate= Top15['Pop_estimate'].index[2]

# Correlation between the number of citable documents per capita and the energy supply per capita Pearson's correlation).
Top15['Pop_estimate']= Top15['Energy Supply']/Top15['Energy Supply per Capita']
Top15['Citable docs per person']=Top15['Citable documents']/Top15['Pop_estimate']
CorrCitable= Top15['Citable docs per person'].corr(Top15['Energy Supply per Capita'])

# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15
# and a 0 if the country's % Renewable value is below the median sorted in ascending order of rank.*
HighRenew= (Top15['% Renewable']) >= Top15['% Renewable'].median()
HighRenew=HighRenew.astype(int)

# Group the Countries by Continent, then create a dateframe that displays the sample size and the sum, mean, and std deviation for the estimated population of each country.
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}

Continents= pd.DataFrame(ContinentDict, index= ['Asia', 'Australia', 'Europe', 'North America', 'South America'])
Continents_values= pd.DataFrame(index=['Asia', 'Australia', 'Europe', 'North America', 'South America'], columns=['size', 'sum', 'mean', 'std'])

# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
PopEst =Top15['Pop_estimate'].apply('{:,}'.format)
 


# Module Project - Hypothesis Testing

# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# The following data files are available for this project:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

# get list of_university towns
    
university_towns = pd.read_csv('university_towns.txt', sep=";", names=['RegionName'])
university_towns.insert(0, 'State', university_towns['RegionName'].str.extract('(.*)\[edit\]', expand=False).ffill())
university_towns['RegionName']= university_towns['RegionName'].str.replace(r' \(.+$', '')
university_towns = university_towns[~university_towns['RegionName'].str.contains('\[edit\]')].reset_index(drop=True)
    
# Load GDP data
GDP2000 = pd.read_excel('gdplev.xls',  usecols = [4,6],skiprows=219, names=['Quarter', 'GDP billion $'])

# Find recession start
GDP2000['GDP Difference']=GDP2000['GDP billion $'].diff()
x_prev = GDP2000['GDP Difference'].shift(1)
x_next = GDP2000['GDP Difference'].shift(-1)
start = pd.DataFrame(np.flatnonzero((x_prev > 0) & (GDP2000['GDP Difference'] < 0) & (x_next < 0)))
recession_start= GDP2000['Quarter'][start[0].iloc[0]]

#Find recession end
start2 = pd.DataFrame(np.flatnonzero((x_prev > 0) & (GDP2000['GDP Difference'] < 0) & (x_next < 0)))[0].iloc[0].astype(np.int32)                      
x= GDP2000['GDP Difference'].loc[start2:] 
x_next2 = x.shift(-2)
x_next1 = x.shift(-1) 
end = pd.DataFrame(np.flatnonzero((x< 0) & (x_next1 > 0) & (x_next2 > 0) ))+start+2
recession_end =  GDP2000['Quarter'][end[0].iloc[0]]

# Find recession bottom
starting= GDP2000[GDP2000['Quarter']==recession_start].index.values.astype(int)[0]
ending = GDP2000[GDP2000['Quarter']==recession_end].index.values.astype(int)[0] 
recession= GDP2000.loc[starting:ending]
recession_bottom= recession['Quarter'][recession['GDP billion $']==recession['GDP billion $'].min()].iloc[0]

## convert housing data to quarters
housing_data = pd.read_csv('City_Zhvi_AllHomes.csv',usecols = [1,2,* range(51,251)])
housing_data['State'].replace(states,inplace=True)
housing_data.set_index(['State', 'RegionName'], inplace=True)
housing_data.sort_index(inplace=True)
housing_data=housing_data.groupby(np.arange(len(housing_data.columns))//3, axis=1).mean()
    
Quarters= GDP2000['Quarter']
Quarters[66]= '2016q3'
housing_data.columns = Quarters

## runing a ttest comparing the university town values to the non-university towns values, 
# returning whether the alternative hypothesis (that the two groups are the same) is true or not as well as the p-value of the confidence. 
# Return the tuple (different, p, better) where different=True if the t-test is True at a p<0.01 (we reject the null hypothesis), or different=False if  otherwise (we cannot reject the null hypothesis).
index_rec_start=housing_data.columns.get_loc(recession_start)
rec_before=housing_data.columns[index_rec_start-1]
housing_data['PriceRatio'] = housing_data[rec_before].div(housing_data[recession_bottom])
university_list= university_towns.to_records(index=False).tolist()
hdf_uni=housing_data.loc[university_list]
hdf_non_uni=housing_data.loc[-housing_data.index.isin(university_list)]
(stat,p)=ttest_ind(hdf_uni.dropna()['PriceRatio'], hdf_non_uni.dropna()['PriceRatio']) 
    
different = p <0.01
    
if hdf_non_uni['PriceRatio'].mean() < hdf_uni['PriceRatio'].mean():
    better = "non-university town"
else:
    better = "university town"
    
Result_ttest= (different,p,better)

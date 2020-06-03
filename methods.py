import numpy as np
import pandas as pd


# read global mobility dataset
mobil_df= pd.read_csv('Global_Mobility_Report.csv', dtype=str) # use str dtype to avoid warning messeage

'''This project primarly focuses on mobil_df dataset'''

# read coronavirus cases dataset
cases_df= pd.read_csv('time_series_covid19_confirmed_global.csv')


# In[2]:


# for mobil_df

# convert numeric cols to float64
flt_cols=['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']


# In[3]:


# convert 
mobil_df[flt_cols]=mobil_df[flt_cols].astype(float)

# ## Methodology:
# 
# ### Data Preprocessing:

# In[30]:

# countries in cases_df
cases_countries= list(set(cases_df['Country/Region']))

#countries in mobil_df
mobil_countries= list(set(mobil_df['country_region']))

def match_countries(bigger_lst, smaller_lst):
    """
    DESCRIPTION: returns elements in smaller list which did not occur in bigger list
        
    INPUT: bigger_lst - (list) list that contains country names
           smaller_lst - (list) list that contains fewer country names 
        
    OUTPUT: countries_not_found - (list) list of countries which were not found in bigger_lst
    """
    
    countries_not_found=[]

    # print countries in mobil_df which have no match in cases_df
    for country in smaller_lst:
        if not(country in bigger_lst):
            
            countries_not_found.append(country)
            
    # if missing countries were found 
    if countries_not_found!=[]: 
        
        # print them
        print('The following countries were not found in cases_df:')
        print(*countries_not_found)
    
    # sort to have the same order always
    return sorted(countries_not_found)


countries_not_found=match_countries(cases_countries, mobil_countries)

def remove_sub_regions(df=mobil_df):
    """
    DESCRIPTION: this function removes all region-wise rows and returns only rows with values for the whole nation.
        
    INPUT: df - (pandas.DataFrame) mobility dataset
        
    OUTPUT: mobil_df_new - (pandas.DataFrame) mobility dataset with only rows for countries as a whole
    """
    # Eliminate all rows based on sub region

    # find rows with nan subregions
    nan_region= (mobil_df.sub_region_1.isna() & mobil_df.sub_region_2.isna())

    # select those rows
    mobil_df_new=mobil_df[nan_region]
    
    # Ensure every nation has the same number of rows
    group_country_no_reg= mobil_df_new.groupby(by='country_region') 
   
    # check if elements all equivalent
    if (group_country_no_reg.size()== group_country_no_reg.size()[0]).all():
        return mobil_df_new
    
    else:
        print("Something went wrong. Please check the input paramater")
        return -1
    


# In[31]:


# call function and reset index (for simplification)
mobil_df_new=remove_sub_regions().reset_index(drop=True)

mobil_df_new.shape


# In[32]:


# add new column for continent
import pycountry_convert as pc


# use technique explained [here](https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry)

# In[33]:


# check if all countries in the dataset have a code

for country in set(mobil_df_new.country_region):
    try:
        # if error it does not have the code
        country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        
    except:
        # print country name
        print(country)
        


# In[34]:


def get_continent(country):
    
    """
    DESCRIPTION: return continent code given country name.
        
    INPUT: country - (str) country name
        
    OUTPUT: continent_code - (str) continent code for country
    """
    
    try:
        #get country code
        country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        # get continent code
        continent_code = pc.country_alpha2_to_continent_code(country_code)
    
        
    except:
        
        # given that there are only two countries that cause an exception
        # it's easy to deal with them manually
        
        if country=='The Bahamas':
            return 'NA'
        
        
        elif country=='Myanmar (Burma)':
            return 'AS'
        
        # Redundant case
        else: 
            print('Something Went Wrong')
            return -1
    
    return continent_code


# In[35]:


# Check if it works on all countries

continent_lst=[]
countries = list(set(mobil_df_new.country_region))

for country in countries:
    
    continent= get_continent(country)
    continent_lst.append(continent)

countries[:10]   


# In[36]:


continent_lst[:10]


# In[37]:


# apply function to all rows in df
continents=mobil_df_new.country_region.apply(get_continent)

# add continents column to mobil_df_new
mobil_df_new['continent']= continents


# In[38]:


mobil_df_new.head()


# In[39]:


# rearrange columns and get rid of sub_region_1 and sub_reg

# get list of columns
cols=list(mobil_df_new.columns)

# replace sub_region_1 column with 'continent'
cols[2]=cols[-1]

# remove sub_region_2
cols.remove('sub_region_2')

# get rid of last element 'continent'
cols=cols[:-1]


# In[40]:


mobil_df_new= mobil_df_new[cols]

mobil_df_new.head()


# In[41]:


# Now to concatenate cases_df with mobil_df_new
# Some chnages must be made on cases_df first

countries_not_found


# In[42]:


#Ad hoc method to match countries not found in both datasets

# Puerto Rico not present in cases_df

"""
Names of missing countries in cases_df

Burma
Taiwan*
Korea, South
US
Cabo Verde
Bahamas
Cote d'Ivoire

*Provinces:

Reunion
Hong Kong
Aruba
"""

names_in_cases= ['Aruba',
                 'Cabo Verde',
                 "Cote d'Ivoire",
                 'Hong Kong',
                 'Burma',
                 'Puerto Rico',
                 'Reunion',
                 'Korea, South',
                 'Taiwan*',
                 'Bahamas',
                 'US'
                ]


# In[43]:


def change_names(countries_not_found=countries_not_found, names_in_cases=names_in_cases, cases_df=cases_df):
    """
    DESCRIPTION: return cases_df with country names that match with mobil_df
        
    INPUT: countries_not_found - (list) country names which were not found in cases_df
           names_in_cases - (list) country names that match countries_not_found but with different spelling
           
           NOTE: one country is missing from cases_df: Puerto Rico
        
    OUTPUT: new_country_ser - (pandas.Series) cases_df['Country/Region'] with modified country names
    """
    # get country/region column
    countries_col= cases_df['Country/Region'].values
    
    # get province column
    province_col=cases_df['Province/State'].values
    
    new_country_ser=cases_df['Country/Region'].copy()
    
    # loop through every country in names_in_cases and countries_not_found
    for cases_name, mobil_name in zip(names_in_cases, countries_not_found):
        
        if cases_name in countries_col:
            # replace country_name with
            if cases_name=='Taiwan*':
                
                # cannot be remove '*' with replace so use strip
                new_country_ser=new_country_ser.str.strip('*')
            
            else:
                # other names are ok
                new_country_ser=new_country_ser.str.replace(cases_name, mobil_name)
        
            
        elif cases_name in province_col:
            
            # find index at which cases_name == province
            index=cases_df[cases_df['Province/State']==cases_name].index[0]
            
            # replace current value with cases_name
            new_country_ser.loc[index]=mobil_name
        else:
            print('The following country did not have a match: {}'.format(cases_name))
    
    return new_country_ser       
            


# In[44]:


cases_countries= list(set(change_names()))

#check now if there are missing countries
match_countries(cases_countries, mobil_countries)


# In[45]:


# now to add the new col to the datset
new_col= change_names()
    
cases_df['Country/Region']=new_col


# In[46]:


# in the cases dataset there are countries which don't have their numbers summed up in one row
cases_group_countries= cases_df.groupby(by='Country/Region')

cases_group_countries.size().sort_values(ascending=False)


# In[47]:


# these rows will need to be reduced to one
cases_df_new=cases_group_countries.sum()


# In[48]:


# ensure num_countries matches number of rows in the dataset
cases_df_new.shape[0]==len(cases_countries)


# In[49]:

# get a list with dates columns only
date_cols=list(cases_df.columns[4:])

# start index
ind_i=date_cols.index('2/15/20')

# end index
ind_f=date_cols.index('5/9/20')

# now to limit cases_df_new to start-end dates of mobil_df
cases_df_new=cases_df_new[date_cols[ind_i:ind_f+1]]


# In[50]:


def create_cases_col(countries_ordered, days=85, cases_df_new=cases_df_new):
    
    """
    DESCRIPTION: create a cases col to concatenate with mobil_df_new
        
    INPUT: countries_ordered - (list) countries list arranged in the same manner as mobil_df_new
           days - (int) number of days in the dataset
           cases_df_new - (pandas.DataFrame) use its values to fill col 
        
    OUTPUT: cases_col - (list) column with len==mobil_df_new.shape[0]
    """
    
    # transpose cases_df_new for easier handeling
    new_cases_transpose= cases_df_new.transpose()
    
    # create an empty list 
    cases_col= []
    
    # create a list of nans for countries not in dataset
    nans=[np.nan for i in range(days)]
    
    # loop through every country
    for country in countries_ordered:
        
        try:
            # add the country's values to the list
            cases_col.extend(cases_df_new.loc[country])
            
        except:
            # country not in dataset, ie Puerto Rico
            cases_col.extend(nans)
            
    
    return cases_col
    


# In[51]:


# get ordered list
countries_ordered=list(pd.unique(mobil_df_new.country_region))

#create cases_col
cases_col=create_cases_col(countries_ordered=countries_ordered)


# In[52]:


# add col to dataframe
mobil_df_new['corona_cases']= cases_col


# In[53]:


# Final form
mobil_df_new.head()


# ## Implementation

# In[54]:


"""For each country, we will look at all location types individually and determine greatest mobility slow-down based on
    changes from baseline and the time at which they were reached"""

# use groupby to get the values based on country
group_country=mobil_df_new.groupby(by='country_region')


# In[55]:


# create dataframe with minimum mobility
min_df=group_country.min()

# max residential reflects min mobility
min_df['residential_percent_change_from_baseline']= group_country.max()['residential_percent_change_from_baseline']


# In[56]:


# get 10 countries with greatest mobility drop

get_first_n= lambda col: col.nsmallest(10).index 

# get n countries with least mobility drop
get_last_n= lambda col: col.nlargest(10).index 


# In[57]:


def sort(col_name, op='min', n=10, df=min_df):
    """
    DESCRIPTION: sort values for the specified column in an ascending order, and plot n largest and smallest values
        
    INPUT: col_name - (str) name of column in df
           op - (str) type of df (min or max)
           n - (str) number of countries to graph 
           df - (pandas.DataFrame) used to sort and plot values
        
    OUTPUT: col_sorted - (pandas.Series) sorted col_name column from df 
    """
    
    # get col with sorted values
    col_sorted= df[col_name].sort_values()
 
    
    
    return col_sorted
    
    


# In[58]:


'''finding min and max is easy enough, however more work needed to be done to find dates at which they occurred'''

# get indecies at which min occurs
min_idx_df= group_country.idxmin()

# get max indecies for Residential
min_idx_df['residential_percent_change_from_baseline']= group_country.idxmax()['residential_percent_change_from_baseline']



# define following function to find max and min dates for specified column
def get_date(col_name, df=mobil_df_new, idx_df=min_idx_df):
    """
    DESCRIPTION: get series of dates given series of indecies. also plots dates in one dimension
        
    INPUT: col_name - (str) name of column in df 
           df - (pandas.DataFrame) the data
           idx_df - (pandas.DataFrame) df which has min index for every col&country combination
        
    OUTPUT: col_min_dates - (pandas.Series) column with dates where min occurs for col_name  
    """
    
    # get the indexes for specified column
    col_min_index=idx_df[col_name]
    
    # do not consider countries which had all nans for col
    col_min_index=col_min_index.dropna()
    
    # ensure index is int
    col_min_index=col_min_index.astype(int)
    
    # def lambda to get and index and return a date
    get_date= lambda index: df.loc[index]['date']
    
    # apply to col with min indicies
    col_min_dates= col_min_index.apply(get_date)
    
    
    return col_min_dates


# In[59]:


# look at output of function
dates=pd.to_datetime(get_date('parks_percent_change_from_baseline'))
dates


# In[60]:


'''create the min mobility dates dataframe'''

# retail-residential
cols_min= min_idx_df.columns[:-1]

# dates at which min mobility occur for particular location type and country
dates_df=pd.DataFrame(map(get_date, min_idx_df[cols_min])).transpose()

# convert dates to datetime
dates_df=dates_df.apply(pd.to_datetime)


# In[61]:


# define a function to get n countries with earliest and latest mobility slow-down
def earliest_countries(col, n=10):
    
    """
    DESCRIPTION: create 2 dataframes with n earliest and n latest countries in response (to reach min)
        
    INPUT: col - (pandas.Series) col from dataframe
 
    OUTPUT: nsmallest.index - (pandas.core.indexes.base.Index) n countries with earliest date
            
              
    """
    # get 10 countries with earliest date
    nsmallest= col.nsmallest(n)
    
    # return name of country
    return nsmallest.index
    
        


# In[62]:


# define a function to get n countries with earliest and latest mobility slow-down
def latest_countries(col, n=10):
    
    """
    DESCRIPTION: create 2 dataframes with n earliest and n latest countries in response (to reach min)
        
    INPUT: col - (pandas.Series) col from dataframe
 
    OUTPUT: nlargest_df - (pandas.core.indexes.base.Index) n countries with latest date
              
    """
    # get 10 countries with earliest date
    nlargest= col.nlargest(n)
    
    # return name of country
    return nlargest.index
    
        


# In[63]:


# Plot the dates in 1-d 
def show_timeline(cols, dates=None, dates_for=''):
   
    """
    DESCRIPTION: plots dates of min/max occurrance in one dimension for all cols
        
    INPUT: cols - (list) list of column names to plot 
 
    OUTPUT: X - (list) list of lines to plot x-axis
        Y - (list) list of lines to plot y-axis
   
    """
    X=[]
    Y=[]
    
    # if dates is not provided, This is used to plot countries
    if dates is None:
        # loop through every column
        for i, col in enumerate(cols):


            # get dates ready for plotting
            dates=pd.to_datetime(get_date(col))

            # for plotting in 1-d
            y=np.zeros(dates.shape)

            X.append(dates)
            Y.append(y-i)

    
    # used to plot others, e.g. continents
    else:
        # loop through every column
        for i, col in enumerate(cols):


            # get dates ready for plotting
            col_dates=dates[col]

            # for plotting in 1-d
            y=np.zeros(col_dates.shape)
            
            X.append(col_dates)
            Y.append(y-i)

   
    return X, Y


# In[64]:


'''Implement continent analysis: average mobility slow-down'''

def get_continent_mean(df=min_df.drop('corona_cases', axis=1)):
    """
    DESCRIPTION: gets the mean of min mobility for each continent
        
    INPUT: df - (pandas.DataFrame) country for index, location types for columns, mins for values.
 
    OUTPUT: continent_mean_df - (pandas.DataFrame) dataframe with mean for each continent and location type 
    """
    # group mins based on continent
    group_continent_min= df.groupby(by='continent')

    # Compute mean of mins for each continent
    continent_mean_df=group_continent_min.mean()
    
    return continent_mean_df


# In[65]:


'''Implement continent analysis: mean dates of mobility slow-down'''

def get_continent_mean_date(df=dates_df, continent_col=min_df.continent):
    """
    DESCRIPTION: gets the median date at which min occurs for each continent and location type
        
    INPUT: df - (pandas.DataFrame) country for index, location types for columns, dates for values.
           continent_col - (pandas.Series) continent column from dataset
 
    OUTPUT: continent_median_date - (pandas.DataFrame) dataframe with mean date for each continent and location type 
    """
    # make a copy of df
    df_new=df.copy()
    
    # add continent column
    df_new['continent']= continent_col
    
    # group mins based on continent
    group_continent_min_date= df_new.groupby(by='continent')
    
    # get list of continents
    continents=list(group_continent_min_date.groups.keys())
    
    # create df to fill
    continent_mean_date_df=pd.DataFrame(columns=group_continent_min_date.dtypes.columns)
    
    # cannot obtain mean date from groupby
    for continent in continents:
        
        #print(group_continent_min_date.get_group(continent))
        
        # for continent, get mean for all cols
        mean_date=group_continent_min_date.get_group(continent).apply(pd.Series.mean)
        
        # get rid of time and keep date
        mean_date_only=mean_date.apply(pd.Timestamp.date)
        
        # put in dataframe
        continent_mean_date_df.loc[continent]= mean_date_only
    
    
    return continent_mean_date_df


# ## Refinement

# In[66]:


''' Because column names are long, dataframe will look too wide. It will be nicer to get rid of 
    "percent_change_from baseline", which prevents from looking at data in a simplified manner '''

# use this function to make data easier to look at
def clean_col_name(df):
    
    """
    DESCRIPTION: remove long form of column name
        
    INPUT: df - (pandas.DataFrame) dataframe for which columns are replaced
 
    OUTPUT: df_new - (pandas.DataFrame) ==df with short version of column names
    """
    # remove '_percent_change_from_baseline'
    reduce= lambda col_name: col_name.replace('_percent_change_from_baseline', '')

    # make a copy of df
    df_new=df.copy()
    
    # dates dataframe
    df_new.columns= list(map(reduce, df_new.columns))

    return df_new


# #### Solving problems with min day

# In[67]:


'''The method (metric) employed in observing global mobility in the times of covid-19 has several issues.
   one of these issues is special events, which could cause a considerable change from baseline while not having any
   relation to the virus. For this reason, a new method will take the average of a week and determine which week had 
   the lowest average in mobility. This will help circumvent the problem with special events'''

def get_week_avg(col):
    """
    DESCRIPTION: get average value in a week for numeric and date columns
        
    INPUT: col - (pandas.Series) column which is divided in weeks
 
    OUTPUT: means_col - (pandas.Series) col with the mean of every week 
    """
    
    # get number of weeks
    num_weeks= int(col.shape[0]/7)
    
    # get remainder days (to add to the last week)
    rem= col.shape[0] % 7
    
    means= []
    
    # check if col is a dates col
    if isinstance(col.iloc[0], str):
    
        # loop for every week
        for i in range(num_weeks):

            # if last week
            if i==(num_weeks-1):
                week=pd.to_datetime(col.iloc[i*7:])


            else:
                # take elements by week
                week=pd.to_datetime(col.iloc[i*7:(i+1)*7])


            # take the mean and append date only
            means.append(week.mean().date())
    
    # case if col is not a date col (numeric)
    else:
        
        # loop for every week
        for i in range(num_weeks):

            # if last week
            if i==(num_weeks-1):
                
                # append all values beginning from final week
                week=col.iloc[i*7:]


            else:
                # take elements by week
                week=col.iloc[i*7:(i+1)*7]


            # append mean of week to list
            means.append(week.mean())
            
    means_col= pd.Series(means)
    
    return means_col    


# In[68]:


def country_in_weeks(col):
    """
    DESCRIPTION: organize country code and continent row to look at week-wise
        
    INPUT: col - (pandas.Series) column which is divided in weeks
 
    OUTPUT: ser - (pandas.Series) country code or continent with num_weeks elements
    """
    
    # get number of weeks in data
    num_weeks= int(col.shape[0]/7)
    
    # get only num_weeks elements and reset index to match avg_week function
    ser=col.iloc[:num_weeks].reset_index(drop=True)
    
    return ser
    


# In[69]:


'''Use two functions above to create a mobility dataset organized week-wise with values in averages.
   covid-cases col will be dropped because it is not in the scope of the analysis'''

def create_mobil_week(group=group_country, countries=countries_ordered, columns=list(mobil_df_new)[:-1], 
                      country_attribute_cols=['country_region_code', 'continent'], cols_to_drop=[],
                      group_col='country_region'):
    """
    DESCRIPTION: get average of every week for every country
        
    INPUT: group - (pandas.core.groupby.generic.DataFrameGroupBy) dataset grouped by country
           countries - (list) list of countries as ordered as dataset (mobil_df_new)
           columns - (list) list of columns in dataset (except corona_cases if default)
           country_attribute_cols - (list) specifies cols which are attributes of country
           cols_to_drop - (list) columns to not include from dataset
           group_col - (str) column by which dataset was grouped
 
    OUTPUT: df_week - (pandas.DataFrame) dataset reduced to week averages 
    """
    
    # create df_week dataframe
    df_week=pd.DataFrame(columns=columns)
    
    
    # loop through every country to get average
    for country in countries:
        
        # get rows of country
        country_df=group.get_group(country)
        
        # ready country_df for get_avg_week
        country_df_date_numeric= country_df.drop(labels=country_attribute_cols+cols_to_drop, axis=1)
        
        # ready country_df for country_in_weeks
        country_df_country_cols= country_df[country_attribute_cols]
        
        # get week numeric averages and week mid date
        temp_df_1= country_df_date_numeric.apply(get_week_avg)
        
        # get country attribute cols to match num rows of temp_df_1
        temp_df_2= country_df_country_cols.apply(country_in_weeks)
        
        # get country_region col
        temp_col=pd.Series([country for i in range(temp_df_1.shape[0])], name=group_col)
        
        # concatenate temps
        concat= pd.concat([temp_col, temp_df_2 , temp_df_1], axis=1, sort= True)
        
        # append concat df to df_week
        df_week=df_week.append(concat, sort=True)
        
    # reset df_week index
    df_week=df_week.reset_index(drop=True)
    
    # organize columns to have the same order as mobil_df_new
    df_week= df_week[columns]
        
        
    return df_week   


# In[70]:


mobil_df_week=create_mobil_week(cols_to_drop=['corona_cases'])


# In[71]:


# group mobil_df_week by country and create min_week_df and min_week_idx_df to use for implementation functions

# group by country
group_country_week= mobil_df_week.groupby(by='country_region')

# find greatest mobility drop (week avg)
min_week_df=group_country_week.min()

# modify resedential
min_week_df['residential_percent_change_from_baseline']= group_country_week.max()['residential_percent_change_from_baseline']

# find index at which it occurs (median date of the week)
min_week_idx_df= group_country_week.idxmin()

# modify resedential
min_week_idx_df['residential_percent_change_from_baseline']= group_country_week.idxmax()['residential_percent_change_from_baseline']


# In[72]:


# create dates_df_week for implementation functions

# retail-resedential
cols_min= list(min_week_idx_df.columns)

# define lambda function to for get date with differents arguments
get_week_date= lambda col_name: get_date(col_name=col_name, df=mobil_df_week, idx_df=min_week_idx_df)

# dates at which min occur for particular location type and country
dates_df_week=pd.DataFrame(map(get_week_date, cols_min)).transpose()

# convert it from datetime.date to datetime
dates_df_week= dates_df_week.apply(pd.to_datetime)

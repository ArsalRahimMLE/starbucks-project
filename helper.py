#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm


# ## Helping Functions

# In[2]:


def countplot(df, col, rotate=False):
    """
    draw a countplot for given column 
    Input: dataframe and column
    Output: Countplot 
    """
    fig, ax = plt.subplots(figsize=(7,4))
    g = sns.countplot(df[col],order=df[col].value_counts().index.tolist())
    ax.set_title(col+ ' distribution')
    ylabels = ['{:,.0f}'.format(y) + 'K' for y in g.get_yticks()/1000]
    g.set_yticklabels(ylabels)
    if rotate:
        ax.tick_params(axis='x', rotation=90)
    plt.show()


# In[3]:


def save_csv(list_of_dataframes, file_names):
    """
    Store csv for each dataframe 
    """
    for i, df in enumerate(list_of_dataframes):
        loc = 'data/' + str(file_names[i]) +'.csv'
        df.to_csv(loc,index_label=False )
    


# ## 1: Load Data:

# In[4]:




# In[5]:




# ## 2: Data Cleaning:

# ### 2.1: Transcript:

# In[6]:




# __Issues:__
# 1. Transactions and offers are in single column. 
# 2. Time seems to bit out of order. 

# 1.Transactions and offers are in single column. 

# In[8]:


def fetch_from_json(json):
    try:
        return json['amount']
    except(KeyError):
        if 'offer id' in json:
            return json['offer id']
        else:
            return json['offer_id']
    


def separate_tran_offer(df):
    df = df.copy()
    cols = ['offer received','offer viewed','offer completed']
    # add offer where col exist # 
    df.loc[df.event.isin(cols),"offer"] = (df.loc[df.event.isin(cols),"value"]).apply(fetch_from_json)
    # add amount where transaction exist #
    df.loc[df.event=='transaction',"amount"] = df.loc[df.event=='transaction',"value"].apply(fetch_from_json)
    df.drop(['value'],axis=1, inplace=True)
    return df




# 2. Time seems to bit out of order. 

# -------leave it for the moment-----

# In[ ]:





# ### 2.2: Promotions:

# In[9]:




# __Issues:__
# 1. Channels are in the form of list in a single column. 
# 2. Multiple columns such as diffculty, duration and offer type for a single promotion. 

# 1. Channels are in the form of list in a single column. 

# In[10]:


def one_hot_channel(df):
    """
    create separate column for each channel
    Input: dataframe 
    Output: dataframe containing separate columns for each channel
    """
    # make a copy #
    df = df.copy()
    # create a list of all channels #
    channels = df.channels.max()
    # create separate column for each channel #
    for channel in channels:
        df[channel] = df['channels'].apply(lambda channels: 1 if channel in channels else 0)
    
    # drop the channels column # 
    df.drop(['channels'],axis=1,inplace=True)
    return df



# 2. Multiple columns such as diffculty, duration and offer type for a single promotion. 

# In[11]:


def merge_cols_promotions(df):
    """
    merge columns (dfficulty, duration, reward and offer type) in single column 
    
    Input: dataframe 
    Output: dataframe contains a merge column for promotion 
    """
    # make a copy of the dataframe #
    df = df.copy()
    # merge columns # 
    df.loc[:,'promotion'] = df['offer_type'].astype(str)+'_'+df['reward'].astype(str)+'_'+                                                 df['difficulty'].astype(str)+'_'+df['duration'].astype(str)
    # drop the individual columns # 
    #df.drop(['reward','offer_type','duration','difficulty'],axis=1, inplace=True)
    return df 
# __Issues:__
# 1. Date is not in format (became member on). 
# 2. There are missing values in gender (as None), income (as NaN) and in some place is age is abnormal (for instance 118). 

# 1. Date is not in format (became member on). 

# In[14]:


def convert_into_dateformat(df, col):
    """
    Function to convert date into pandas datatime format. 
    
    Input: dataframe and column (contain date)
    Output: dataframe containing date (date time format)
    """
    # make a copy of dataframe #
    df = df.copy()
    # convert it into string #
    df[col] = df[col].astype(str)
    df[col] = pd.to_datetime(df[col])
    
    return df


def merge_data(profile_dt, transcript_dt, promotions_dt):
    """
    Function to merge all three datasets (profiles, promotions and transcripts)
    
    Input: dataframe for profiles, promotions and transcripts 
    Output: merged dataframe
    """
    profile_dt = profile_dt.copy()
    transcript_dt = transcript_dt.copy()
    promotions_dt = promotions_dt.copy()
    # 1. cleaning of datasets: 
    
    # clean transcript data #
    transcript_dt = separate_tran_offer(transcript_dt)
    # clean promotions data #
    promotions_dt = one_hot_channel(promotions_dt)
    promotions_dt = merge_cols_promotions(promotions_dt)
    # clean profile data #
    profile_dt = convert_into_dateformat(profile_dt,'became_member_on')
    
    # merge datasets: 
    
    # merge transcript and promotions # 
    trans_pro_dt = transcript_dt.merge(promotions_dt,how="left", left_on="offer", right_on="id").drop('id', axis=1)
    # merge profile to the recent merged dataframe #
    trans_pro_profile_dt = trans_pro_dt.merge(profile_dt,how="left", left_on="person", right_on="id").drop('id', axis=1)
    
    return trans_pro_profile_dt


# In[19]:


# ## 4. Data Visualization: 

# 1. `event` distribution. 

# 2. Distribution of `amount`,`reward`,`difficulty` and `duration`

# In[22]:


def find_customer_with_offer_type(df,transactions, event_type='offer completed'):
    """
    Find customers who have completed any offer and calculate their avg spending #
    """
    # make a copy #
    df = df.copy()
    # make a dataframe # 
    persons_df = pd.DataFrame(columns =['person','average spending'])
    # Those who completed the offer #
    persons = df[df['event']==event_type]['person'].unique()
    print(len(persons))
    # for each calculate the amount their avg spend #
    for person in persons:
        amount = transactions[transactions['person']==person]['amount'].mean()
        persons_df = persons_df.append({'person' : person , 'average spending' : amount} , ignore_index=True)
    return persons_df



# In[36]:


# find customer who didn't complete offer #
def find_regular_cutomers(df,transactions, offer_comp_persons):
    """
    Find regular customers and calulate their avg spending #
    """
    # make a copy #
    df = df.copy()
    # make a dataframe # 
    persons_df = pd.DataFrame(columns =['person','average spending'])
    # find regular customer #
    regular_customers = df[~df.person.isin(offer_comp_persons)]['person'].unique()
    print(len(regular_customers))
    # for each calculate the amount their avg spend #
    for person in regular_customers:
        amount = transactions[transactions['person']==person]['amount'].mean()
        persons_df = persons_df.append({'person' : person , 'average spending' : amount} , ignore_index=True)
    return persons_df



# merge regular and offer customers:

# In[37]:



def event_statistics(df, customers, event_types):
    """
    Calculate statistics such as count, mean, median etc of different events for each customer 
    """
    try:
        event_stats = pd.read_csv('data/event_counts.csv')
    
    except:
        # make a copy # 
        df = df.copy()
        # make a dataframe #
        event_stats = pd.DataFrame(columns=['person'])
        # make a dictionary #
        event_dict = {}
        # calculate statistics of event type for each customer #
        for person in tqdm(customers):
            for event in event_types:
                event_dict[event] = df[(df['person']==person) & (df['event']==str(event))]['event'].count()
            event_dict['person'] = person
            event_stats = event_stats.append(event_dict , ignore_index=True)
            event_dict = {}
    return event_stats




# Store the data. 

# In[45]:




# In[46]:



#     Let's label valid and invalid transactions

# In[47]:


def find_invalid_trans(df):
    """
    Separate transactions which are motivated by offers and which are regular transactions
    """
    # make a copy #
    df = df.copy()
    # define helping variables #
    customers = df.person.unique()
    invalid = []
    # actual show time: #
    for person in tqdm(customers):
        # make a subset of customer #
        subset_dt = df[df['person']==person]
        # look for complete offers #
        comp_off_dt = subset_dt[subset_dt['event']=='offer completed']
        # for each offer calculate start & end time #
        for offer in comp_off_dt.offer:
            offer_row = comp_off_dt[comp_off_dt['offer']==offer]
            start_time = ((offer_row['time'].values) -  (offer_row['duration'].values*24))[0]
            endtime = offer_row['time'].values[0]
            #print('start-time:',start_time,'\nend-time',endtime)
            # take a subset from start to end time #
            subset = subset_dt[(subset_dt['time'] >= start_time) & (subset_dt['time'] <= endtime)]
            # check if the offer was viewed or not # 
            offer_viewed = subset[(subset['offer']==offer) & (subset['event']=='offer viewed')]
            if offer_viewed.shape[0]==0:
                invalid.append(offer_row.index[0])
      
    # create a flag for invalid #
    df.loc[:, 'invalid'] = 0
    df.loc[invalid, 'invalid'] = 1
    return df

        
    


# In[ ]:



# ## Feature Engineering:#

# In[48]:


def add_invalid_feature(customers, trans_df):
    """
    Add feature: count of invalid (regular) transactions per customer 
    """
    # make a copy #
    customers = customers.copy()
    # set the index #
    customers.set_index(['id'],inplace=True)
    # calculate the counts per customer #
    agg = trans_df[trans_df['invalid'] == 1]['person'].value_counts()
    # add the feature to customer #
    customers['invalid'] = agg
    # fill NaN (in case there is no invalid transaction against a specifc customer)
    customers['invalid'] = customers['invalid'].fillna(0)
    return customers


# In[49]:


def add_event_features(df, customers):
    """
    Add feature: count of transaction, offer received, viewed and completed
    """
    # make a copy # 
    customers = customers.copy()
    # make a list of unique events #
    cols = df.event.unique()
    # make a list of unique customers #
    list_of_customers = customers.index.unique()
    # calculate event statistics
    events_statistics_df = event_statistics(merged_dt,list_of_customers,cols)
    # set the index #
    events_statistics_df.set_index(['person'],inplace=True)
    # add features to customer data #
    for event in cols:
        customers.loc[:,event] = events_statistics_df.loc[:,event]
    return customers


# In[50]:


def add_amount_features(transactions, customers):
    """
    Add feature: avg and total spending of customer 
    """
    # make a copy #
    customers = customers.copy()
    # make variables #
    features = ['avg spending','total spending']
    func = ['mean','sum']
    # calculate avg and total spending of customer #
    agg = transactions.groupby(by=['person'])['amount'].agg(func)
    # add features to customer #
    for i,feature in enumerate(features):
        customers.loc[:,feature] = agg.loc[:,func[i]]
    return customers
    


# In[85]:


def add_ratio_features(customers):
    """
    Add feature: add ratio as feature such as ratio of completion with viewed per customer and many more.
    """
    # make a copy #
    customers = customers.copy()
    # calculate ratio #
    
    # -- 1. completion ratio (offer complete/offer viewed)
    customers.loc[:,'ratio_offer_completion'] = customers.loc[:,'custom offer completed']/customers.loc[:,'custom offer viewed']
    # -- 2. view ratio (offer viewed/offer received)
    customers.loc[:,'ratio_offer_viewed'] = customers.loc[:,'custom offer viewed']/customers.loc[:,'custom offer received']
    
    return customers


# In[67]:


def add_cal_offer_details(df, customers):
    """
    calculate the total offer completed, offer viewed and offer received by user(bogo and discount)
    """
    events = ['offer received','offer viewed','offer completed']
    df = df.copy()
    customers = customers.copy()
    users = customers.index
    for event in events:
        holder_dict = {}
        for user in tqdm(users):
            # fetch all entries of specific customer #
            subset = df[df['person']==user]
            # event for each customer #
            holder_df = subset[(subset['event']==event) & (subset['offer_type'].isin(['bogo','discount']))]
            holder_dict[user] = len(holder_df)
        col_name = 'custom '+event
        count = pd.Series(holder_dict)
        customers[col_name] = count
    return customers


# In[52]:



def remove_incorrect_records(customers):
    """
    Remove records which doesn't make sense. (such as offer completed when offer viewed was 0)
    """
    # make a copy #
    customers = customers.copy()
    # remove records #
    # -- 1. remove records where offer completed but it was not viewed #
    customers = customers[~((customers['offer viewed']==0) & (customers['offer completed']>0))]
    return customers


# ## Firing Up the functions:

# In[78]:



def transaction_preprocessing(profiles,transcripts,promotions):
    """
    Preprocess and integrate other details into transactions before starting feature engineering 
    """
    try:
        # if available then load it from the directory #
        trans_dt = pd.read_csv('data/trans_dt.csv')
        trans_clean_dt = pd.read_csv('data/trans_clean_dt.csv')
        print('File load from local directory as it was already available')
    except:
        # include all the details in transcript #
        trans_dt = merge_data(profiles, transcripts, promotions)
        # clean trans for valid entries #
        trans_dt = find_invalid_trans(trans_dt)
        trans_clean_dt = trans_dt[trans_dt['invalid']==0]
        # save the data #
        save_csv(list_of_dataframes=[trans_dt,trans_clean_dt],file_names=['trans_dt','trans_clean_dt'])
    return trans_dt, trans_clean_dt
        


# In[86]:


def feature_engineering(trans_dt, trans_clean_dt, profiles, promotions):
    """
    Apply all the transformations before applying algorithm
    """
    try:
        customers = pd.read_csv('data/customers_data_FE.csv')
    except:
        # add invalid count for each customer #
        customers = add_invalid_feature(profiles,trans_dt)
        # add event statistics #
        customers = add_event_features(trans_dt,customers)
        # add amount features #
        customers = add_amount_features(trans_dt, customers)
        customers = add_cal_offer_details(trans_clean_dt,customers)
        customers = add_ratio_features(customers)
        save_csv(list_of_dataframes=[customers],file_names=['customers_data_FE'])
    return customers


# In[87]:



# In[89]:





# In[ ]:





# In[ ]:





# In[63]:

#!/usr/bin/env python
# coding: utf-8

# # vX3

# # **Accessing working environment Kaggle**

# In[1]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **Importing libraries**

# In[2]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 200)
from time import time
import datetime as dt
import gc # clear garbage


# # Debugging f-ions

# In[3]:


def debugging_ids(df):
    return print(f'Debugging submitted dataframe: \nUnique installation_ids: {len(set(df.installation_id))} \nRows & columns count {df.shape}')


# # **Loading data**

# In[4]:


load_columns = ['event_id',
                'game_session',
                'timestamp',                
                'installation_id',
                'event_count',
                'event_code',
                'game_time',
                'title',
                'type',
                'world',
                'event_data']

path = '/kaggle/input/data-science-bowl-2019/' # create url path to the datasets

t0 = time()

print('Loading datasets...')
X_train = pd.read_csv(path + 'train.csv', usecols = load_columns)
X_labels = pd.read_csv(path + 'train_labels.csv')
# specs = pd.read_csv(path + 'specs.csv')
#X_test = pd.read_csv(path + 'test.csv', usecols = load_columns)
#submission = pd.read_csv(path + 'sample_submission.csv')
print("Datasets loaded successfully! \nLoading time:", round(time() - t0, 3), "s")


# # **Data preparation**

# ### **(T) Reducing train df with users having accuracy scores (17000 -> 3614 installation_ids)**

# In[5]:


# X_train has 17000 installation_id's, however there are only for 3614 installation_id's (X_labels and X_train) with Assessment attempt
# Reducing X_train to 17000 -> 3614 installation_ids
X_train = X_train[X_train['installation_id'].isin(set(X_labels.installation_id))]


# ### **Extracting accuracy of previous Assessment attempts**
# 
# * Preparing train set which is identical to train_labels except:
# * accuracy differs for 46 observations due to saving in more floating points (16 ours vs 9 train_labels.csv)
# * removed the last assessment's (target) row

# #### (T) Create X_train_gt by extracting only rows with assessments events

# In[6]:


# Creating X_train_gt to hold only rows with assessment attempts

X_train_gt = pd.DataFrame(data=None)

# X_train_gt will be used only for accuracy features extraction
# First, filter assessment events only

X_train_gt = X_train[((X_train['event_code'] == 4100) & 
                 (X_train['title'].isin(['Cart Balancer (Assessment)', 
                                    'Cauldron Filler (Assessment)', 
                                    'Mushroom Sorter (Assessment)', 
                                    'Chest Sorter (Assessment)']))) | 
                ((X_train['event_code'] == 4110) & 
                 (X_train['title'] == 'Bird Measurer (Assessment)'))].copy(deep=True)   


# In[7]:


# debugging
debugging_ids(X_train_gt)


# In[8]:


#X_train_gt[X_train_gt['installation_id'] == '0006c192']


# #### (T) Drop columns which will be processed later

# In[9]:


# Fourth, drop columns which will be processed separately

X_train_gt.drop(['event_id', 
                 'timestamp', 
                 'event_count', 
                 'event_code', 
                 'game_time',
                 'type',
                 'world',], axis=1, inplace=True)


# In[10]:


gc.collect()


# In[11]:


# debugging
debugging_ids(X_train_gt)


# #### (T) Extract accuracy features from 'event_data'

# In[12]:


# Fifths, extract correct and incorrect assessment attempts per user from 'event_data'
# Create num_correct and num_incorrect columns

corr = '"correct":true'
incorr = '"correct":false'

X_train_gt['num_correct'] = X_train_gt['event_data'].apply(lambda x: 1 if corr in x else 0)
X_train_gt['num_incorrect'] = X_train_gt['event_data'].apply(lambda x: 1 if incorr in x else 0)


# In[13]:


# debugging
debugging_ids(X_train_gt)


# In[14]:


#X_train_gt


# In[15]:


# Sixths, aggregate (sum) correct and incorrect assessment attempts 
# per 'game_session', 'installation_id' and assessment 'title'
# As provided in grount truth (labels.csv)

# previous aggregation was made together with sorting to match train_labels format
#X_train_gt = X_train_gt.sort_values(['installation_id', 'game_session'], ascending=True).groupby(['game_session', 'installation_id', 'title'], as_index=False, sort=False).agg(sum)
# a) difficult to extract last assessment
# b) difficult to truncate
# c) difficult to accumulate actions before assessment
X_train_gt = X_train_gt.groupby(['game_session', 'installation_id', 'title'], as_index=False, sort=False).agg(sum)


# In[16]:


#X_train_gt


# In[17]:


#X_labels


# In[18]:


# # Great, because w/o sorting by game_session and installation_id 
# # we preserve the original order of events by timestamp 
# X_train_gt[X_train_gt['installation_id'] == '0006c192']


# In[19]:


#X_labels[X_labels['installation_id'] == '0006c192']


# In[20]:


#X_train[(X_train['installation_id'] == '0006c192') & ((X_train['event_code'] == 4100) | (X_train['event_code'] == 4110))]


# In[21]:


# Sevenths, create 'accuracy' feature = corr / (corre + incorr)

X_train_gt['accuracy'] = X_train_gt['num_correct'] / (X_train_gt['num_correct'] + X_train_gt['num_incorrect'])

# Eighths, create 'accuracy_group' feature
# 3: the assessment was solved on the first attempt
# 2: the assessment was solved on the second attempt
# 1: the assessment was solved after 3 or more attempts
# 0: the assessment was never solved

# If accuracy is 0.0 (no correct attempts), accuracy group is 0 as all observations in X_train_gt by now has at least one attempt
# If accuracy is 1.0 (that is no incorrect attempts), accuracy group is 3
# If accuracy is 0.5 (there is equal amount of correct and incorrect attempts), accuracy group is 2
# Any other case means that accuracy group equals 1, that is 3 or more attempts were needed to make a correct attempt    

X_train_gt['accuracy_group'] = X_train_gt['accuracy'].apply(lambda x: 0 if x == 0.0 else (3 if x == 1.0 else (2 if x == 0.5 else 1)))


# In[22]:


# debugging
debugging_ids(X_train_gt)


# In[23]:


# # task is to forecast 'accuracy_group' in the last 'game_session' of single 'installation_id'
# # E.g. 'installation_id' '0006a69f' last assessment
# # in last 'game_session' 'a9ef3ecb3d1acc6a' was 'Bird Measurer (Assessment)'
# # our task is to forecast that his 'accuracy_group' was '3' 
# X_train_gt


# In[24]:


# # Comparing with ground truth sample:
# X_labels.head(8)
# # As we removed sorting, only overall count should match


# In[25]:


# # Double check accuracy figures in X_train_gt and X_labels

# print(f'SUM (OK)')
# print(f'X_train_gt has accuracy_group sum of {sum(X_train_gt["accuracy_group"])} \nX_labels has accuracy_group sum of {sum(X_labels["accuracy_group"])}')

# print(f'\nTYPE (OK)')
# print(f'Type of X_train_gt num_correct is {type(X_train_gt["num_correct"][0])} \nType of X_labels num_correct is {type(X_labels["num_correct"][0])}')
# print(f'Type of X_train_gt num_incorrect is {type(X_train_gt["num_incorrect"][0])} \nType of X_labels num_incorrect is {type(X_labels["num_incorrect"][0])}') 
# print(f'Type of X_train_gt accuracy is {type(X_train_gt["accuracy"][0])} \nType of X_labels accuracy is {type(X_labels["accuracy"][0])}') 
# print(f'Type of X_train_gt accuracy_group is {type(X_train_gt["accuracy_group"][0])} \nType of X_labels accuracy_group is {type(X_labels["accuracy_group"][0])}')

# print(f'\nDIFFERENCES')
# print(f'Difference between accuracy column in X_train_gt and X_labels is: {set(X_train_gt["accuracy"] - X_labels["accuracy"])}')
# print(f'Difference between accuracy_group column in X_train_gt and X_labels is: {set(X_train_gt["accuracy_group"] - X_labels["accuracy_group"])}')
# print(f'Accuracy set len in X_train_gt is: {len(set(X_train_gt["accuracy"]))}')
# print(f'Accuracy set len in X_labels is: {len(set(X_labels["accuracy"]))}')
# print(f'Difference between num_correct column in X_train_gt and X_labels is: {set(X_train_gt["num_correct"] - X_labels["num_correct"])}')
# print(f'Difference between num_incorrect column in X_train_gt and X_labels is: {set(X_train_gt["num_incorrect"] - X_labels["num_incorrect"])}')

# print(f'\nEQUAL VALUES ROW BY ROW')

# booltest_session = X_train_gt.game_session == X_labels.game_session
# booltest_ids = X_train_gt.installation_id == X_labels.installation_id
# booltest_title = X_train_gt.title == X_labels.title
# booltest_num_correct = X_train_gt.num_correct == X_labels.num_correct
# booltest_num_incorrect = X_train_gt.num_incorrect == X_labels.num_incorrect
# booltest_accuracy = X_train_gt.accuracy == X_labels.accuracy
# booltest_accuracy_group = X_train_gt.accuracy_group == X_labels.accuracy_group

# print(f'Equal values (TRUE) of game_session in X_train_gt and X_labels: \n{booltest_session.value_counts()}')
# print(f'Equal values (TRUE) of installation_id in X_train_gt and X_labels: \n{booltest_ids.value_counts()}')
# print(f'Equal values (TRUE) of title in X_train_gt and X_labels: \n{booltest_title.value_counts()}')
# print(f'Equal values (TRUE) of num_correct in X_train_gt and X_labels: \n{booltest_num_correct.value_counts()}')
# print(f'Equal values (TRUE) of num_incorrect in X_train_gt and X_labels: \n{booltest_num_incorrect.value_counts()}')
# print(f'Equal values (TRUE) of accuracy in X_train_gt and X_labels: \n{booltest_accuracy.value_counts()}')
# print(f'Equal values (TRUE) of accuracy_group in X_train_gt and X_labels: \n{booltest_accuracy_group.value_counts()}')

# # Changelog:
# # Index was fixed by applying .sort_values(['installation_id', 'game_session'], ascending=True) in the groupby part
# # Now difference between accuracy_group columns in X_train_gt and X_labels should be {0}


# In[26]:


## Debugging 46 accuracy scores which do not match.
# not_matching_accuracy_df = X_train_gt.accuracy - X_labels.accuracy
# not_matching_accuracy_df = not_matching_accuracy_df[not_matching_accuracy_df != 0]
# #len(not_matching_accuracy_df) = 46
# X_train_gt[X_train_gt.index.isin(not_matching_accuracy_df.index)]
# # X_labels[X_labels.index.isin(not_matching_accuracy_df.index)]
# # Conclusion: We produce 16 digits after comma, train_labels.csv has 9
#X_train_gt[X_train_gt.index.isin(not_matching_accuracy_df.index)].to_csv("different_accuracies.csv", index = False)


# ### (T) Accuracy groups

# In[27]:


X_train_gt['acc_0'] = X_train_gt['accuracy_group'].apply(lambda x: 1 if x == 0 else 0)
X_train_gt['acc_1'] = X_train_gt['accuracy_group'].apply(lambda x: 1 if x == 1 else 0)
X_train_gt['acc_2'] = X_train_gt['accuracy_group'].apply(lambda x: 1 if x == 2 else 0)
X_train_gt['acc_3'] = X_train_gt['accuracy_group'].apply(lambda x: 1 if x == 3 else 0)


# In[28]:


# debugging
# X_train_gt[X_train_gt['installation_id'] == '0006a69f']


# In[29]:


# debugging
debugging_ids(X_train_gt)


# ### (T) Accuracy groups per assessment 'title'

# In[30]:


# Accuracy group per assessment title
# Ref: https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas/27475029
# (condition, output value, else)

X_train_gt['bird_accg_0'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['accuracy_group'] == 0), 1, 0)
X_train_gt['bird_accg_1'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['accuracy_group'] == 1), 1, 0)
X_train_gt['bird_accg_2'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['accuracy_group'] == 2), 1, 0)
X_train_gt['bird_accg_3'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['accuracy_group'] == 3), 1, 0)

X_train_gt['cart_accg_0'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['accuracy_group'] == 0), 1, 0)
X_train_gt['cart_accg_1'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['accuracy_group'] == 1), 1, 0)
X_train_gt['cart_accg_2'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['accuracy_group'] == 2), 1, 0)
X_train_gt['cart_accg_3'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['accuracy_group'] == 3), 1, 0)

X_train_gt['cauldron_accg_0'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['accuracy_group'] == 0), 1, 0)
X_train_gt['cauldron_accg_1'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['accuracy_group'] == 1), 1, 0)
X_train_gt['cauldron_accg_2'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['accuracy_group'] == 2), 1, 0)
X_train_gt['cauldron_accg_3'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['accuracy_group'] == 3), 1, 0)

X_train_gt['chest_accg_0'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 0), 1, 0)
X_train_gt['chest_accg_1'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 1), 1, 0)
X_train_gt['chest_accg_2'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 2), 1, 0)
X_train_gt['chest_accg_3'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 3), 1, 0)

X_train_gt['mushroom_accg_0'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 0), 1, 0)
X_train_gt['mushroom_accg_1'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 1), 1, 0)
X_train_gt['mushroom_accg_2'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 2), 1, 0)
X_train_gt['mushroom_accg_3'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['accuracy_group'] == 3), 1, 0)


# In[31]:


# debugging
#X_train_gt['mushroom_accg_0'][17688]


# In[32]:


# debugging
#X_train_gt[X_train_gt['installation_id'] == '0006a69f']


# In[33]:


# debugging
debugging_ids(X_train_gt)


# ### (T) Accuracy (corr, incorr, accuracy) per assessment

# In[34]:


# Accuracy group per assessment title
# Ref: https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas/27475029
# (condition, output value, else)
# E.g. if Bird Measurer has num_correct = 1, add 1, elsewise add 0

X_train_gt['bird_correct'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['num_correct'] == 1), 1, 0)
X_train_gt['bird_incorrect'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)') & (X_train_gt['num_incorrect'] > 0), X_train_gt['num_incorrect'], 0)
X_train_gt['bird_accuracy'] = np.where((X_train_gt['title'] == 'Bird Measurer (Assessment)'), X_train_gt['accuracy'], 0)

X_train_gt['cart_correct'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['num_correct'] == 1), 1, 0)
X_train_gt['cart_incorrect'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)') & (X_train_gt['num_incorrect'] > 0), X_train_gt['num_incorrect'], 0)
X_train_gt['cart_accuracy'] = np.where((X_train_gt['title'] == 'Cart Balancer (Assessment)'), X_train_gt['accuracy'], 0)

X_train_gt['cauldron_correct'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['num_correct'] == 1), 1, 0)
X_train_gt['cauldron_incorrect'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)') & (X_train_gt['num_incorrect'] > 0), X_train_gt['num_incorrect'], 0)
X_train_gt['cauldron_accuracy'] = np.where((X_train_gt['title'] == 'Cauldron Filler (Assessment)'), X_train_gt['accuracy'], 0)

X_train_gt['chest_correct'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['num_correct'] == 1), 1, 0)
X_train_gt['chest_incorrect'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)') & (X_train_gt['num_incorrect'] > 0), X_train_gt['num_incorrect'], 0)
X_train_gt['chest_accuracy'] = np.where((X_train_gt['title'] == 'Chest Sorter (Assessment)'), X_train_gt['accuracy'], 0)

X_train_gt['mushroom_correct'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['num_correct'] == 1), 1, 0)
X_train_gt['mushroom_incorrect'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_train_gt['num_incorrect'] > 0), X_train_gt['num_incorrect'], 0)
X_train_gt['mushroom_accuracy'] = np.where((X_train_gt['title'] == 'Mushroom Sorter (Assessment)'), X_train_gt['accuracy'], 0)


# In[35]:


# debugging
# X_train_gt[X_train_gt['installation_id'] == '0006a69f']


# In[36]:


# debugging
debugging_ids(X_train_gt)


# ### Removing last assessment from train set
# 
# * X_train_gt at this point has 41549 assessments
# * Can not remove just last one before aggregation 

# In[37]:


# Remove the last assessment's attempt from train (new from 200115)

# Build temporary df which holds last assessment
X_train_gt_last = X_train_gt.groupby('installation_id').tail(1).copy(deep=True)
X_train_gt_last_index_list = list(X_train_gt_last.index)

# Removing last assessment attempt from test set
# 'installation_id's drop 3614->3021 as we have users who had just single attempt
X_train_gt = X_train_gt.loc[~X_train_gt.index.isin(X_train_gt_last_index_list)]


# In[38]:


# debugging
debugging_ids(X_train_gt_last)


# In[39]:


# # debugging
# X_train_gt_last.head(5)


# In[40]:


# # debugging, good case of 0006c192
# X_train[(X_train['installation_id'] == '0006c192') & ((X_train['event_code'] == 4100) | (X_train['event_code'] == 4110))]


# In[41]:


# X_train_gt_last[(X_train_gt_last['installation_id'] == '0006c192')]


# In[42]:


# debugging
debugging_ids(X_train_gt)


# ### (~T) Aggregation
# 
# Tested the build, updated avoiding extra df, but haven't double-checked sample means or sums

# In[43]:


X_train_gt_sum_list = ['num_correct', 'num_incorrect', 
       'bird_correct', 'bird_incorrect',
       'cart_correct', 'cart_incorrect', 'cauldron_correct',
       'cauldron_incorrect', 'chest_correct',
       'chest_incorrect', 'mushroom_correct',
       'mushroom_incorrect', 'acc_0',
       'acc_1', 'acc_2', 'acc_3', 'bird_accg_0', 'bird_accg_1', 'bird_accg_2',
       'bird_accg_3', 'cart_accg_0', 'cart_accg_1', 'cart_accg_2',
       'cart_accg_3', 'cauldron_accg_0', 'cauldron_accg_1', 'cauldron_accg_2',
       'cauldron_accg_3', 'chest_accg_0', 'chest_accg_1', 'chest_accg_2',
       'chest_accg_3', 'mushroom_accg_0', 'mushroom_accg_1', 'mushroom_accg_2',
       'mushroom_accg_3']

X_train_gt_mean_list = ['accuracy',
       'accuracy_group', 'bird_accuracy',
       'cart_accuracy', 'cauldron_accuracy', 'chest_accuracy', 'mushroom_accuracy']


# In[44]:


#len(X_train_gt_sum_list), len(X_train_gt_mean_list)


# In[45]:


X_train_gt_sum_df = X_train_gt.groupby(['installation_id'], as_index=False, sort=False)[X_train_gt_sum_list].agg(sum)


# In[46]:


#X_train_gt_sum_df


# In[47]:


X_train_gt_mean_df = X_train_gt.groupby(['installation_id'], as_index=False, sort=False)[X_train_gt_mean_list].agg('mean')


# In[48]:


#X_train_gt_mean_df


# In[49]:


#X_train_gt_unchaged_df = X_train_gt.groupby(['installation_id'], as_index=False, sort=False)[X_train_gt_unchanged_list].last()


# In[50]:


X_train_gt = pd.merge(X_train_gt_sum_df, X_train_gt_mean_df, how='left', on=['installation_id'])


# In[51]:


del X_train_gt_sum_df, X_train_gt_mean_df
gc.collect()


# In[52]:


#X_train_gt


# In[53]:


# debugging
debugging_ids(X_train_gt)


# ## Adding users w/o previous assessment attempts

# In[54]:


train_features_list = X_train_gt.columns


# In[55]:


print(f'X_train iids: {len(set(X_train.installation_id))} \nX_train_gt iids: {len(set(X_train_gt.installation_id))} \nX_labels iids: {len(set(X_labels.installation_id))}')


# In[56]:


train_users_wo_assessments = set(X_train.installation_id) - set(X_train_gt.installation_id)
len(train_users_wo_assessments)


# ### Creating empty df matching test's columns

# In[57]:


train_users_wo_assessments_df = pd.DataFrame(0, index=np.arange(len(train_users_wo_assessments)), columns=train_features_list)


# In[58]:


train_users_wo_assessments_df


# ### Adding 'installation_id's w/o prior assessments

# In[59]:


# We have created installation_id column with zero values. Now will assign missing 'installation_id's:
train_users_wo_assessments_df['installation_id'] = train_users_wo_assessments


# In[60]:


train_users_wo_assessments_df


# ### Merging 'installation_id's with and w/o assessments

# In[61]:


X_train_gt = X_train_gt.append(train_users_wo_assessments_df, ignore_index=True)


# In[62]:


X_train_gt


# In[63]:


# debugging
debugging_ids(X_train_gt)


# In[64]:


# # debugging
# # we lost the order of 'installation_id', but submission is sorted ascending
# booltrain_label = X_train_gt.installation_id.sort_values(ascending=True).reset_index(drop=True) == X_labels.installation_id
# set(booltrain_label)


# In[65]:


del train_users_wo_assessments_df
gc.collect()


# ### (T) Sorting to match order of initial train set
# * Because after merger of users with previous assessments and without we lost the initial ordering

# In[66]:


X_train_gt = X_train_gt.sort_values('installation_id', ascending=True).reset_index(drop=True)


# In[67]:


# X_labels


# In[68]:


# # debugging
# # check if sorting of 'installation_id's matches train_labels sorting
# # for this need to drop duplicates in X_labels as it contain 17690 rows with 'installation_id's
# # ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.drop_duplicates.html
# # reseting index and dropping old index via reset_index(drop=True)
# # does not lose the sorting
# # THIS PART TO BE UNCOMMENTED:
# X_labels_unique_installation_id = X_labels.installation_id.drop_duplicates().reset_index(drop=True)
# booltrain_label = X_train_gt.installation_id == X_labels_unique_installation_id
# set(booltrain_label)


# In[69]:


# del X_labels_unique_installation_id, booltrain_label
# gc.collect()


# In[70]:


# debugging
debugging_ids(X_train_gt)


# ### Adding previous assessments count

# In[71]:


X_train_gt['previous_assessments_count'] = X_train_gt['num_correct'] + X_train_gt['num_incorrect']

# debugging
debugging_ids(X_train_gt)


# ### Adding 'forecasted_assessment' feature

# In[72]:


X_train_gt.shape, X_train_gt_last.shape


# In[73]:


#X_train_gt_last


# In[74]:


# X_train[(X_train['installation_id'] == '0006c192') & ((X_train['event_code'] == 4100) | (X_train['event_code'] == 4110))]


# In[75]:


# # X_train_gt_last is taking X_train index 4137->11337808
# train_forecasted_assessment_df = X_train_gt_last.sort_values('installation_id', ascending=True).reset_index(drop=True)
# train_forecasted_assessment_df


# In[76]:


# # check if last df had the right 'title' for forecasted assessment
# X_labels.head(20)


# In[77]:


# # double-check sorting - OK
# boollast_label = train_forecasted_assessment_df.installation_id == X_labels_unique_installation_id
# set(boollast_label)


# In[78]:


# train_forecasted_assessment_df.shape


# In[79]:


#X_train_gt


# In[80]:


# # Need to reset X_train_gt_last index for boolean comparison
# X_train_gt_last


# In[81]:


# Debugging - double-check sorting of X_train_gt_last & X_train_gt
X_train_gt_last = X_train_gt_last.reset_index(drop=True)
# Above we updated the X_train_gt_last index to match 0-3613 (total of 3614)
booltrain_last = X_train_gt.installation_id == X_train_gt_last.installation_id
set(booltrain_last)


# In[82]:


del booltrain_last
gc.collect()


# In[83]:


# # Updated index:
# X_train_gt_last


# In[84]:


X_train_gt['forecasted_assessment'] = X_train_gt_last['title'].map({'Bird Measurer (Assessment)': 0,
                                                                            'Cart Balancer (Assessment)': 1, 
                                                                            'Cauldron Filler (Assessment)': 2, 
                                                                            'Chest Sorter (Assessment)': 3, 
                                                                            'Mushroom Sorter (Assessment)': 4})


# In[85]:


# debugging
debugging_ids(X_train_gt)


# In[86]:


set(X_train_gt.forecasted_assessment), X_train_gt.forecasted_assessment.count()


# # (~T) Adding non accuracy features
# ### bugs:
# #### - data is not truncated after forecasted_event
# #### - we take last assessment, which might better off be random
# 
# Given that test set contains almost half of installation_ids without previous assessments, we need to add other than accuracy features for model to pick up

# ## (~T) event_code

# #### Preparing event_code features

# In[87]:


#X_train


# In[88]:


def event_code(df):
    df = pd.get_dummies(data=df, columns=['event_code'])
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' |-|!|\)|\(', '')
    df = df.groupby(['installation_id'], as_index=False, sort=False).agg(sum)  
    return df


# In[89]:


# Uses ~3 GB of RAM for this operation (9->12->9)
X_train_eventcode = X_train.filter(['installation_id', 'event_code'], axis=1)
X_train_eventcode = event_code(X_train)


# In[90]:


#X_train_eventcode


# #### Merging event_code features to the main train set

# In[91]:


# Add event_code features to the main dataframe
X_train_gt = pd.merge(X_train_gt, X_train_eventcode, on=['installation_id'])
# # Count nan in df for debugging purposes
# X_train_gt.isna().sum()

del X_train_eventcode
gc.collect()

# debugging
debugging_ids(X_train_gt)


# In[92]:


X_train_gt


# ## (~T) Title, type, world and event_code

# #### Preparing title, type and world features

# In[93]:


gc.collect()


# In[94]:


# Uses RAM 9.1->13.8->8.7
def title_type_world(df):
    df = pd.get_dummies(data=df, columns=['title', 'type', 'world'])
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' |-|!|\)|\(', '')
    df = df.groupby(['installation_id'], as_index=False, sort=False).agg(sum) 
    return df


# In[95]:


# Create new X_train_titletypeworldfeat which holds time only title, type and world features
X_train_titletypeworldfeat = X_train.filter(['installation_id', 'title', 'type', 'world'], axis=1)
X_train_titletypeworldfeat = title_type_world(X_train_titletypeworldfeat)


# In[96]:


#X_train_titletypeworldfeat


# #### Merging title, type and world features to the main train set

# In[97]:


# Add title, type and world features to the main dataframe
X_train_gt = pd.merge(X_train_gt, X_train_titletypeworldfeat, on=['installation_id'])
# # Count nan in df for debugging purposes
# X_train_gt.isna().sum()

del X_train_titletypeworldfeat
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ## (~T) Other features
# 
# * all_actions_time
# 
# *     Aggregate amount (in ms) of time spent on Assessments, Activities and Games
# *     Clips do not have time spent feature
# 
# * all_actions_time
# * action_duration_mean (!!!)
# * event_code_count_mean
# * number_of_sessions_nu
# * event_count_mean (!!!)

# ###### (T) all_actions_time

# In[98]:


# nstallation_id	game_time
# 0	0006a69f	36368
# 1	0006c192	216374
# 2	00129856	39701
# 3	001d0ed0	38115
# 4	00225f67	26517
# ...	...	...
# 3609	ff9305d7	59417
# 3610	ff9715db	28408
# 3611	ffc90c32	43142
# 3612	ffd2871d	54533
# 3613	ffeb0b1b	71511

# vs
# installation_id	game_time
# 0	0006a69f	36368
# 1	0006c192	216374
# 2	00129856	39701
# 3	001d0ed0	38115
# 4	00225f67	26517
# ...	...	...
# 3609	ff9305d7	59417
# 3610	ff9715db	28408
# 3611	ffc90c32	43142
# 3612	ffd2871d	54533
# 3613	ffeb0b1b	71511


# In[99]:


# Tested, works well, except truncation after last assessment
# Creating all_actions_time (games, activities and assessments)
# RAM: 8.7->8.5-8.7 GB
feat_gametime = X_train[X_train['type'].isin(['Assessment', 'Game', 'Activity'])]

# Extracting last assessment's time
feat_gametime_last = feat_gametime.groupby(['installation_id', 'game_session'], as_index=False, sort=False)[['game_time', 'type']].last()
feat_gametime_last = feat_gametime_last[feat_gametime_last['type'] == 'Assessment'].groupby('installation_id', as_index=False, sort=False)['game_time'].last()

# Finalizing the whole time
feat_gametime = feat_gametime.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
feat_gametime = feat_gametime.groupby('installation_id', as_index=False, sort=False)['game_time'].sum()

# Removing last assessments time which is not available in test set
feat_gametime['game_time'] = feat_gametime['game_time'] - feat_gametime_last['game_time']
# Difference is correct, tested

# Merging to the main train set
X_train_gt['all_actions_time'] = feat_gametime['game_time']

# Deleting
del feat_gametime, feat_gametime_last 
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ###### action_duration_mean

# In[100]:


# Creating action_duration_mean (games, activities and assessments) (!!!)
# RAM: 8.7->9.6->8.7 GB
feat_gametimemean = X_train[X_train['type'].isin(['Assessment', 'Game', 'Activity'])]
feat_gametimemean = feat_gametimemean.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
feat_gametimemean = feat_gametimemean.groupby('installation_id', as_index=False, sort=False)['game_time'].mean()

# Merging to the main train set
X_train_gt['action_duration_mean'] = feat_gametimemean['game_time']

# Deleting
del feat_gametimemean
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ###### event_code_count_mean

# In[101]:


# Creating event_code_count_mean (!!!)
# RAM: OK, flat
feat_eventcodecountmean = X_train.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['event_code'].count()
feat_eventcodecountmean = feat_eventcodecountmean.groupby('installation_id', as_index=False, sort=False)['event_code'].mean()

# Merging to the main train set
X_train_gt['event_code_count_mean'] = feat_eventcodecountmean['event_code']

# Deleting
del feat_eventcodecountmean
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ##### number_of_sessions_nu

# In[102]:


# Creating event_code_count_mean
# RAM: OK, flat
feat_numberofsessions = X_train.groupby(['installation_id'], as_index=False, sort=False)['game_session'].count()

# Merging to the main train set
X_train_gt['number_of_sessions_nu'] = feat_numberofsessions['game_session']

# Deleting
del feat_numberofsessions
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ##### event_count_mean

# In[103]:


# Creating event_count_mean (!!!)
# RAM: OK, flat
feat_eventcountmean = X_train.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['event_count'].last()
feat_eventcountmean = feat_eventcountmean.groupby('installation_id', as_index=False, sort=False)['event_count'].mean()

# Merging to the main train set
X_train_gt['event_count_mean'] = feat_eventcountmean['event_count']

# Deleting
del feat_eventcountmean
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ## (~T) timestamp

# In[104]:


# bug - taking the last even, which might be not assessment
# could replace with mean

import datetime as dt

def timestamp_split(df):
    df['timestamp'] = pd.to_datetime(df['timestamp']) # converting argument to pandas datetime
#    df['year'] = df['timestamp'].dt.year # all are in 2019
    df['month'] = (df['timestamp'].dt.month).astype(int)
    df['day'] = (df['timestamp'].dt.day).astype(int) # returns day of the month 1-31
    df['hour'] = (df['timestamp'].dt.hour).astype(int) 
    df['minute'] = (df['timestamp'].dt.minute).astype(int)
#    df['second'] = df['timestamp'].dt.second # doubt it could give anything
    df['dayofweek'] = (df['timestamp'].dt.dayofweek).astype(int) # returns day of week in 0-6 integer format
    df['dayofyear'] = (df['timestamp'].dt.dayofyear).astype(int) # returns numeric day of year, might be useful for summer holidays
    df['quarter'] = (df['timestamp'].dt.quarter).astype(int)
    df['is_weekend'] = (np.where(df['dayofweek'].isin(['Sunday','Saturday']), 1, 0)).astype(int)
    df.drop(['timestamp'], axis=1, inplace=True)
    return df


# In[105]:


# RAM 8.7->10->9.3
# Create new X_train_timefeat which holds time only features  
feat_time = X_train.filter(['installation_id', 'timestamp'], axis=1)
# Prepare time features from given timestamp 
feat_time = timestamp_split(feat_time)


# In[106]:


# Defining as last (bug)
feat_time = feat_time.groupby('installation_id', as_index=False).last()

# Merging to the main train set
X_train_gt = pd.merge(X_train_gt, feat_time, on=['installation_id'])

# Deleting
del feat_time
gc.collect()


# In[107]:


del X_train, X_labels
gc.collect()

# debugging
debugging_ids(X_train_gt)


# ## Adding train target

# In[108]:


# Update 200117, major bug fix
X_train_gt['Y_target'] = X_train_gt_last['accuracy_group']

# debugging
debugging_ids(X_train_gt)


# ## Preparing X, y

# In[109]:


X_train_model = X_train_gt.copy(deep=True)

#del X_train_gt
#gc.collect()


# In[110]:


X_train_model.isna().sum()


# In[111]:


# # Casting categorical features to str (must in Catboost & Eli5)
# X_train_model['forecasted_assessment'] = X_train_model['forecasted_assessment'].astype(str)
# categorical_features = ['forecasted_assessment'] #200119
# type(X_train_model.forecasted_assessment[3612])


# In[112]:


# Elsewise LightGBMError: Do not support special JSON characters in feature name.
X_train_model.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train_model.columns]


# ##### StandardScaler

# In[113]:


# Dropping non numeric column 'installation_id'
X_train_model = X_train_model.drop(['installation_id'], axis=1)


# In[114]:


# Defining scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# ##### Targets

# In[115]:


# Setting target & features
y = X_train_model.Y_target
feature_names = X_train_model.columns.drop(['Y_target'])
X = X_train_model[feature_names]


# In[116]:


# Scaling
X_scaled = scaler.fit_transform(X.astype(np.float64))


# ##### Resampling

# In[117]:


# import tensorflow as tf
# # from collections import Counter
# # from sklearn.datasets import make_classification
# # from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
# from imblearn import undersampling, oversampling
# from imblearn import under_sampling 
# from imblearn import over_sampling
# from imblearn.over_sampling import SMOTE


# In[118]:


# # Ref: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
# # from collections import Counter
# # from sklearn.datasets import make_classification
# from imblearn.over_sampling import SMOTE
# # X, y = make_classification(n_classes=2, 
# #                            class_sep=2,
# #                            weights=[0.1, 0.9], 
# #                            n_informative=3, 
# #                            n_redundant=1, 
# #                            flip_y=0,
# #                            n_features=20, 
# #                            n_clusters_per_class=1, 
# #                            n_samples=1000, 
# #                            random_state=10)

# # print('Original dataset shape %s' % Counter(y))

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_scaled, y)
# print('Resampled dataset shape %s' % Counter(y_res))


# # Metric

# In[119]:


# Check Cohen Kappa Score:
from sklearn.metrics import cohen_kappa_score


# # Model w XGBoost

# In[120]:


# from sklearn.model_selection import train_test_split
# train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.metrics import accuracy_score
# import xgboost as xgb

# xgb_clf = xgb.XGBClassifier(learning_rate=0.5,
#                     n_estimators=2000,
#                     max_depth=6,
#                     min_child_weight=0,
#                     gamma=0,
#                     reg_lambda=1,
#                     subsample=1,
#                     colsample_bytree=0.75,
#                     scale_pos_weight=1,
#                     objective='multi:softprob',
#                     num_class=4,
#                     verbose=200,
#                     random_state=42,
#                     early_stopping_rounds=10,
#                     verbose_eval=True)

# xgb_model = xgb_clf.fit(train_X, train_y)
# xgb_preds = xgb_model.predict(val_X)
# xgb_proba = xgb_model.predict_proba(val_X)

# xgb_kappa_score = cohen_kappa_score(val_y, xgb_preds, weights='quadratic')

# print(f'\n****')
# print(f'Accuracy of predictions is: {accuracy_score(val_y, xgb_preds)}')
# # NB! Add weights='quadratic' to get same result as QWK 
# print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, xgb_preds, weights="quadratic")}')


# # Model w Catboost

# In[121]:


#X_train_model.head(10)


# In[122]:


#X_train_model.filter(items=['installation_id', 'num_correct', 'num_incorrect', 'forecasted_assessment'])


# In[123]:


list(X_train_model.columns)


# In[124]:


# # Catboost Classification
# # Important: X_scaled added 200121
# from sklearn.model_selection import train_test_split
# train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state = 0)

# from catboost import CatBoostClassifier
# from sklearn.metrics import accuracy_score

# params_cb = {
#             'max_depth' : 5,
#             'learning_rate' : 0.01,
#             'n_estimators' : 1493,
#             'verbose' : 200,
# #            'od_type': 'Iter',
#             'loss_function' : 'MultiClass' #200109 new
#             }

# cbc_model = CatBoostClassifier(**params_cb)
# cbc_model.fit(train_X, train_y)
# #cbc_model.fit(train_X, train_y, eval_set=(val_X, val_y), early_stopping_rounds=10, use_best_model=True) #200119 use_best suggestion for bestIteration = 2679, Shrink model to first 2680 iterations
# cbc_preds = cbc_model.predict(val_X)

# # Save Catboost accuracy
# cbc_score = accuracy_score(val_y, cbc_preds)
# print(f'\n****')
# print(f'Accuracy of predictions is: {accuracy_score(val_y, cbc_preds)}')

# # Check Cohen Kappa Score:
# from sklearn.metrics import cohen_kappa_score
# cbc_kappa_score = cohen_kappa_score(val_y, cbc_preds, weights='quadratic')

# # NB! Add weights='quadratic' to get same result as QWK 
# print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, cbc_preds, weights="quadratic")}')


# # Model w Catboost regressor

# In[125]:


# # CatBoostRegressor
# # Stopped by overfitting detector  (10 iterations wait)
# # bestTest = 0.8568023128
# # bestIteration = 1141
# # Shrink model to first 1142 iterations.
# # ****
# # Accuracy of predictions is: 0.734110203229486


# from sklearn.model_selection import train_test_split
# train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state = 0)

# from catboost import CatBoostRegressor
# #from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

# params_cb = {
#             'max_depth' : 5,
#             'learning_rate' : 0.01,
#             'n_estimators' : 1142,
#             'verbose' : 200,
# #            'od_type': 'Iter',
#             'loss_function' : 'RMSE' #200109 new
#             }

# cbc_model = CatBoostRegressor(**params_cb)
# cbc_model.fit(train_X, train_y)
# #cbc_model.fit(train_X, train_y, eval_set=(val_X, val_y), early_stopping_rounds=10, use_best_model=True) #200119 use_best suggestion for bestIteration = 2679, Shrink model to first 2680 iterations
# cbc_preds = cbc_model.predict(val_X)

# # Save Catboost accuracy
# cbc_score = mean_squared_error(val_y, cbc_preds)
# print(f'\n****')
# print(f'Accuracy of predictions is: {mean_squared_error(val_y, cbc_preds)}')

# # # Check Cohen Kappa Score:
# # from sklearn.metrics import cohen_kappa_score
# # cbc_kappa_score = cohen_kappa_score(val_y, cbc_preds, weights='quadratic')

# # # NB! Add weights='quadratic' to get same result as QWK 
# # print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, cbc_preds, weights="quadratic")}')


# In[126]:


# # GridSearchCV
# from sklearn.metrics import cohen_kappa_score, make_scorer
# from sklearn.model_selection import GridSearchCV

# kappa_scorer = make_scorer(cohen_kappa_score()
# grid = GridSearchCV(CatBoostClassifier(), param_grid={'C': [1, 10]}, scoring=kappa_scorer)


# In[127]:


# # CV to assess model's quality
# # Ref: https://scikit-learn.org/stable/modules/model_evaluation.html

# # Creat 
# from sklearn.metrics import cohen_kappa_score, make_scorer
# kappa_scorer = make_scorer(cohen_kappa_score)

# # from sklearn import svm, datasets
# from sklearn.model_selection import cross_val_score
# clf_cbc = CatBoostClassifier(**params_cb)
# #cross_val_score(clf_cbc, X, y, cv=5, scoring='accuracy')
# #scores = cross_val_score(clf_cbc, X, y, cv=5, scoring='accuracy')
# cross_val_score(clf_cbc, X, y, cv=5, scoring=kappa_scorer) # scoring=cohen_kappa_score

# #cross_val_score(clf, X, y, cv=5, scoring='recall_macro')
# #array([0.96..., 0.96..., 0.96..., 0.93..., 1.        ])
# #>>> model = svm.SVC()
# #>>> cross_val_score(model, X, y, cv=5, scoring='wrong_choice')
# #Traceback (most recent call last):


# In[128]:


# # Permutation Importance

# import eli5
# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(cbc_model, random_state=1).fit(val_X, val_y)
# eli5.show_weights(perm, top=160, feature_names = list(feature_names)) #val_X.columns.tolist() -> list(feature_names)


# In[129]:


# # Permutation Importance XGBoost

# import eli5
# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(xgb_model, random_state=1).fit(val_X, val_y)
# eli5.show_weights(perm, top=150, feature_names = val_X.columns.tolist())


# In[130]:


#type(val_X.forecasted_assessment[1087])


# # PCA

# In[131]:


# Apply PCA for dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=10).fit(X)
X_pca = pca.transform(X)
print(sum(pca.explained_variance_ratio_))


# # Model w LightGBM
# * First - Classifier

# In[132]:


# len(X), len(y), len(train_X), len(train_y), len(val_X), len(val_y)


# In[133]:


# # Light GBM Classifier

# from sklearn.model_selection import train_test_split
# train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2765, random_state = 0)

# import lightgbm as lgb
# # from sklearn.metrics import accuracy_score
# from sklearn.metrics import log_loss
# # create dataset for lightgbm
# lgb_train = lgb.Dataset(train_X, train_y)
# lgb_eval = lgb.Dataset(val_X, val_y)

# # specify parameters
# params_lgb = {
#             'boosting_type': 'gbdt',
#             'objective': 'multiclass', #            'objective': 'multiclass',
#             'num_class': 4,
#             'metric': '',
#             'num_leaves': 31,
#             'learning_rate': 0.01,
#             'feature_fraction': 0.9,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'verbose': 0,
#            'is_unbalance': True,
#             'num_iterations': 3000
#             }

# print('Starting training...')
# # train
# gbm_model = lgb.train(params_lgb,
#                       lgb_train,
#                      num_boost_round=20,
#                      valid_sets=lgb_eval,
#                      early_stopping_rounds=5
#                      )

# print('Starting predicting...')
# # predict
# gbm_pred = gbm_model.predict(val_X, num_iteration=gbm_model.best_iteration)
# # eval
# print(':', )
# print(f'log_loss of predictions is: {log_loss(val_y, gbm_preds)}')
# #print(f'Accuracy of predictions is: {accuracy_score(val_y, gbm_preds)}')
# #print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, gbm_pred, weights="quadratic")}')


# In[134]:


# LGBM with PCA X_pca

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X_pca, y, random_state = 0)

import lightgbm as lgb
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# create dataset for lightgbm
lgb_train = lgb.Dataset(train_X, label=train_y)
lgb_eval = lgb.Dataset(val_X, label=val_y, reference=lgb_train)

# specify parameters
params_lgb = {'n_estimators': 10000,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
             'max_depth': 15,
            'lambda_l1': 1,  
            'lambda_l2': 1,
            'verbose': 100,
#            'early_stopping_rounds': 100, 
            'eval_metric': 'cappa'
            }

print('Starting training...')
# train
gbm_model = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval) #, early_stopping_rounds=100)
#gbm_model = lgb.train(params_lgb, lgb_train, valid_sets=lgb_eval) #new200120 , verbose_eval=verbosity

print('Starting predicting...')
# predict
gbm_pred = gbm_model.predict(val_X, num_iteration=gbm_model.best_iteration)
# eval
print(':', )
print(f'The rmse of prediction is: {mean_squared_error(val_y, gbm_pred) ** 0.5}')
#print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, gbm_pred, weights="quadratic")}')


# In[135]:


# # Ref: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py

# from sklearn.model_selection import train_test_split
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# import lightgbm as lgb
# # from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

# # create dataset for lightgbm
# lgb_train = lgb.Dataset(train_X, label=train_y)
# lgb_eval = lgb.Dataset(val_X, label=val_y, reference=lgb_train)

# # specify parameters
# params_lgb = {'n_estimators': 10000,
#             'boosting_type': 'gbdt',
#             'objective': 'regression',
#             'metric': 'rmse',
#             'subsample': 0.75,
#             'subsample_freq': 1,
#             'learning_rate': 0.04,
#             'feature_fraction': 0.9,
#              'max_depth': 15,
#             'lambda_l1': 1,  
#             'lambda_l2': 1,
#             'verbose': 100,
#             'early_stopping_rounds': 100, 'eval_metric': 'cappa'
#             }

# print('Starting training...')
# # train
# #gbm_model = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
# gbm_model = lgb.train(params_lgb, lgb_train, valid_sets=lgb_eval) #new200120 , verbose_eval=verbosity

# print('Starting predicting...')
# # predict
# gbm_pred = gbm_model.predict(val_X, num_iteration=gbm_model.best_iteration)
# # eval
# print(':', )
# print(f'The rmse of prediction is: {mean_squared_error(val_y, gbm_pred) ** 0.5}')
# #print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(val_y, gbm_pred, weights="quadratic")}')


# In[136]:


# # Train on all dataset LightGBM Reg

# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error

# # Create dataset for lightgbm on full train set
# lgb_train = lgb.Dataset(X, label=y)

# # specify parameters
# params_lgb = {'n_estimators': 142,
#             'boosting_type': 'gbdt',
#             'objective': 'regression',
#             'metric': 'rmse',
#             'subsample': 0.75,
#             'subsample_freq': 1,
#             'learning_rate': 0.04,
#             'feature_fraction': 0.9,
#              'max_depth': 15,
#             'lambda_l1': 1,  
#             'lambda_l2': 1,
#             'verbose': 100,
# #            'early_stopping_rounds': 100, 
#             'eval_metric': 'cappa'
#             }

# print('Starting training...')
# # train
# #gbm_model = lgb.train(params_lgb, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
# gbm_model = lgb.train(params_lgb, lgb_train) #new200120 , verbose_eval=verbosity
# print('Training done...')

# print('***')
# print('Starting predicting...')
# # predict
# gbm_pred = gbm_model.predict(X, num_iteration=gbm_model.best_iteration)
# # eval
# print('***')
# print(f'The rmse of prediction is: {mean_squared_error(y, gbm_pred) ** 0.5}')
# #print(f'Skikit-learn Cohen Kappa Score (Quadratic) of predictions is: {cohen_kappa_score(y, gbm_pred, weights="quadratic")}')


# In[137]:


# # LightGBM

# submission = pd.read_csv(path + 'sample_submission.csv')
# gbm_preds = gbm_model.predict(X_test_gt)

# submission['accuracy_group'] = gbm_preds

# submission['accuracy_group_weight0'] = np.where((submission['accuracy_group'] <= 0.81387600), 0, 0)
# submission['accuracy_group_weight1'] = np.where((submission['accuracy_group'] > 0.81387600) & (submission['accuracy_group'] <= 1.09392700), 1, 0)
# submission['accuracy_group_weight2'] = np.where((submission['accuracy_group'] > 1.09392700) & (submission['accuracy_group'] <= 1.42779600), 2, 0)
# submission['accuracy_group_weight3'] = np.where((submission['accuracy_group'] > 1.42779600), 3, 0)
# submission['accuracy_group'] = submission['accuracy_group_weight0'] + submission['accuracy_group_weight1'] + submission['accuracy_group_weight2'] + submission['accuracy_group_weight3']
# submission = submission.drop(['accuracy_group_weight0', 'accuracy_group_weight1', 'accuracy_group_weight2', 'accuracy_group_weight3'], axis=1)

# submission.to_csv("submission.csv", index = False)

# submission.accuracy_group.value_counts()


# In[138]:


# gbm_preds = gbm_model.predict(X_test_gt)
# submission['accuracy_group'] = np.round(gbm_preds).astype(int)
# submission.to_csv("submission.csv", index = False)
# submission.head()
# submission.accuracy_group.value_counts()


# In[139]:


del X_train_model
gc.collect()


# # Preparing test set

# In[140]:


# Preparing test set
X_test = pd.read_csv(path + 'test.csv', usecols = load_columns)
submission = pd.read_csv(path + 'sample_submission.csv')


# In[141]:


def extract_accuracy_set_test(df):
    X_test_gt = pd.DataFrame(data=None)
    
    # X_test_gt will be used only for accuracy features extraction
    # First, filter assessment events only
    # Second, drop columns which will be processed separately
    
    X_test_gt = df[((df['event_code'] == 4100) & 
                     (df['title'].isin(['Cart Balancer (Assessment)', 
                                        'Cauldron Filler (Assessment)', 
                                        'Mushroom Sorter (Assessment)', 
                                        'Chest Sorter (Assessment)']))) | 
                    ((df['event_code'] == 4110) & 
                     (df['title'] == 'Bird Measurer (Assessment)'))].copy(deep=True)

    
#     #quick add of assessments_time
    
#     X_test_game_time = X_test_gt.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
#     X_test_game_time = X_test_game_time.groupby('installation_id', as_index=False, sort=False)['game_time'].sum()
    
    X_test_gt.drop(['event_id', 
                     'timestamp', 
                     'event_count', 
                     'event_code', 
                     'game_time',
                     'type',
                     'world',], axis=1, inplace=True)
    
    # Third, extract correct and incorrect assessment attempts per user from 'event_data'
    # Create num_correct and num_incorrect columns
    
    corr = '"correct":true'
    incorr = '"correct":false'
    
    X_test_gt['num_correct'] = X_test_gt['event_data'].apply(lambda x: 1 if corr in x else 0)
    X_test_gt['num_incorrect'] = X_test_gt['event_data'].apply(lambda x: 1 if incorr in x else 0)
    
    # Fourth, aggregate (sum) correct and incorrect assessment attempts 
    # per 'game_session', 'installation_id' and assessment 'title'
    # As provided in grount truth (labels.csv)
    
    X_test_gt = X_test_gt.sort_values(['installation_id', 'game_session'], ascending=True).groupby(['game_session', 'installation_id', 'title'], as_index=False, sort=False).agg(sum)
    
    # Fifths, create 'accuracy' feature = corr / (corre + incorr)
    
    X_test_gt['accuracy'] = X_test_gt['num_correct'] / (X_test_gt['num_correct'] + X_test_gt['num_incorrect'])
    
    # Sixths, create 'accuracy_group' feature
    # 3: the assessment was solved on the first attempt
    # 2: the assessment was solved on the second attempt
    # 1: the assessment was solved after 3 or more attempts
    # 0: the assessment was never solved

    # If accuracy is 0.0 (no correct attempts), accuracy group is 0 as all observations in X_test_gt by now has at least one attempt
    # If accuracy is 1.0 (that is no incorrect attempts), accuracy group is 3
    # If accuracy is 0.5 (there is equal amount of correct and incorrect attempts), accuracy group is 2
    # Any other case means that accuracy group equals 1, that is 3 or more attempts were needed to make a correct attempt    

    X_test_gt['accuracy_group'] = X_test_gt['accuracy'].apply(lambda x: 0 if x == 0.0 else (3 if x == 1.0 else (2 if x == 0.5 else 1)))
   
    return X_test_gt

X_test_gt = extract_accuracy_set_test(X_test)


# In[142]:


# debugging
debugging_ids(X_test_gt)


# ### (T) Assessment count
# **Adjusted** for test set as:
# * not all users took assessment
# * in test.csv our forecasted assessment is not under 4100 or 4110 code, therefore does not include in gt df
# * feature shows how many unique assessments user took before, not total count of non-unique assessments

# In[143]:


# Creating the last assessment coll
X_test_gt['previous_assessments_count'] = X_test_gt.groupby('installation_id')['title'].transform('count')
# Difference with train prep:
# No need to reduce by one as last one under 4100 or 4110 code is not the one we are forecasting
# X_test_gt['previous_assessments_count'] = X_test_gt['previous_assessments_count'].apply(lambda x: x -1 if x > 1 else 0)


# In[144]:


#X_test_gt.head(2)


# In[145]:


# X_test[(X_test['installation_id'] == '01242218') & ((X_test['event_code'] == 4100) | (X_test['event_code'] == 4110))]


# In[146]:


# debugging
debugging_ids(X_test_gt)


# ### (~T) Accuracy groups
# 
# * Should be fine as we do not have forecasted assessment's, that is do not count additional 0 accuracy_group

# In[147]:


#Accuracy groups
X_test_gt['acc_0'] = X_test_gt['accuracy_group'].apply(lambda x: 1 if x == 0 else 0)
X_test_gt['acc_1'] = X_test_gt['accuracy_group'].apply(lambda x: 1 if x == 1 else 0)
X_test_gt['acc_2'] = X_test_gt['accuracy_group'].apply(lambda x: 1 if x == 2 else 0)
X_test_gt['acc_3'] = X_test_gt['accuracy_group'].apply(lambda x: 1 if x == 3 else 0)


# In[148]:


# X_test_gt.head(5)


# In[149]:


# debugging
debugging_ids(X_test_gt)


# ### (T) accuracy_group per assessment title

# In[150]:


# 'accuracy_group' per assessment 'title'
# Ref: https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas/27475029
# (condition, output value, else)

X_test_gt['bird_accg_0'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['accuracy_group'] == 0), 1, 0)
X_test_gt['bird_accg_1'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['accuracy_group'] == 1), 1, 0)
X_test_gt['bird_accg_2'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['accuracy_group'] == 2), 1, 0)
X_test_gt['bird_accg_3'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['accuracy_group'] == 3), 1, 0)

X_test_gt['cart_accg_0'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['accuracy_group'] == 0), 1, 0)
X_test_gt['cart_accg_1'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['accuracy_group'] == 1), 1, 0)
X_test_gt['cart_accg_2'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['accuracy_group'] == 2), 1, 0)
X_test_gt['cart_accg_3'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['accuracy_group'] == 3), 1, 0)

X_test_gt['cauldron_accg_0'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['accuracy_group'] == 0), 1, 0)
X_test_gt['cauldron_accg_1'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['accuracy_group'] == 1), 1, 0)
X_test_gt['cauldron_accg_2'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['accuracy_group'] == 2), 1, 0)
X_test_gt['cauldron_accg_3'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['accuracy_group'] == 3), 1, 0)

X_test_gt['chest_accg_0'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 0), 1, 0)
X_test_gt['chest_accg_1'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 1), 1, 0)
X_test_gt['chest_accg_2'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 2), 1, 0)
X_test_gt['chest_accg_3'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 3), 1, 0)

X_test_gt['mushroom_accg_0'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 0), 1, 0)
X_test_gt['mushroom_accg_1'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 1), 1, 0)
X_test_gt['mushroom_accg_2'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 2), 1, 0)
X_test_gt['mushroom_accg_3'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['accuracy_group'] == 3), 1, 0)

# debugging
debugging_ids(X_test_gt)


# ### (T) Accuracy (num_correct, num_incorrect, accuracy) per assessment

# In[151]:


# {title}_correct, {title}_incorrect, {title}_accuracy per 'installation_id' per assessment 'title'
# Ref: https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas/27475029
# (condition, output value, else)
# E.g. if Bird Measurer has num_correct = 1, add 1, elsewise add 0
# If Bird Measurer has num_incorrect = 12, add 12, elsewise add 0

X_test_gt['bird_correct'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['num_correct'] == 1), 1, 0)
X_test_gt['bird_incorrect'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)') & (X_test_gt['num_incorrect'] > 0), X_test_gt['num_incorrect'], 0)
X_test_gt['bird_accuracy'] = np.where((X_test_gt['title'] == 'Bird Measurer (Assessment)'), X_test_gt['accuracy'], 0)

X_test_gt['cart_correct'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['num_correct'] == 1), 1, 0)
X_test_gt['cart_incorrect'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)') & (X_test_gt['num_incorrect'] > 0), X_test_gt['num_incorrect'], 0)
X_test_gt['cart_accuracy'] = np.where((X_test_gt['title'] == 'Cart Balancer (Assessment)'), X_test_gt['accuracy'], 0)

X_test_gt['cauldron_correct'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['num_correct'] == 1), 1, 0)
X_test_gt['cauldron_incorrect'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)') & (X_test_gt['num_incorrect'] > 0), X_test_gt['num_incorrect'], 0)
X_test_gt['cauldron_accuracy'] = np.where((X_test_gt['title'] == 'Cauldron Filler (Assessment)'), X_test_gt['accuracy'], 0)

X_test_gt['chest_correct'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['num_correct'] == 1), 1, 0)
X_test_gt['chest_incorrect'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)') & (X_test_gt['num_incorrect'] > 0), X_test_gt['num_incorrect'], 0)
X_test_gt['chest_accuracy'] = np.where((X_test_gt['title'] == 'Chest Sorter (Assessment)'), X_test_gt['accuracy'], 0)

X_test_gt['mushroom_correct'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['num_correct'] == 1), 1, 0)
X_test_gt['mushroom_incorrect'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)') & (X_test_gt['num_incorrect'] > 0), X_test_gt['num_incorrect'], 0)
X_test_gt['mushroom_accuracy'] = np.where((X_test_gt['title'] == 'Mushroom Sorter (Assessment)'), X_test_gt['accuracy'], 0)

# debugging
debugging_ids(X_test_gt)


# ### (T) Aggregation of features
# 
# * Leaving single row per 'installation_id'
# 
# ##### Headline in train: Saving the index of last (forecasted) assessment
# 
# * No need to separate FC assessments row from the rest as it is not included in test set
# * Will perform only aggregation

# In[152]:


# Not applicable to test set:
# # We prepare a dataframe which stores the index of last assessment of each installation_id with assessment attempt
# last_observations_index_df = X_test_gt.reset_index().groupby('installation_id', as_index=False)['index'].last()
# last_observations_index_list = list(last_observations_index_df['index']) 
# X_test_gt.drop(['game_session', 'title'], axis=1, inplace=True)
# # Creating a copy dataframe with last_observations and without them
# X_test_gt_last = X_test_gt.loc[X_test_gt.index.isin(last_observations_index_list)]
# X_test_gt_remainder = X_test_gt.loc[~X_test_gt.index.isin(last_observations_index_list)]

X_test_gt_remainder_sum_list = X_train_gt_sum_list

# X_test_gt_remainder_sum_list = ['num_correct', 'num_incorrect', 
#        'bird_correct', 'bird_incorrect',
#        'cart_correct', 'cart_incorrect', 'cauldron_correct',
#        'cauldron_incorrect', 'chest_correct',
#        'chest_incorrect', 'mushroom_correct',
#        'mushroom_incorrect', 'acc_0',
#        'acc_1', 'acc_2', 'acc_3', 'bird_accg_0', 'bird_accg_1', 'bird_accg_2',
#        'bird_accg_3', 'cart_accg_0', 'cart_accg_1', 'cart_accg_2',
#        'cart_accg_3', 'cauldron_accg_0', 'cauldron_accg_1', 'cauldron_accg_2',
#        'cauldron_accg_3', 'chest_accg_0', 'chest_accg_1', 'chest_accg_2',
#        'chest_accg_3', 'mushroom_accg_0', 'mushroom_accg_1', 'mushroom_accg_2',
#        'mushroom_accg_3']

X_test_gt_remainder_mean_list = X_train_gt_mean_list

# X_test_gt_remainder_mean_list = ['accuracy',
#        'accuracy_group', 'bird_accuracy',
#        'cart_accuracy', 'cauldron_accuracy', 'chest_accuracy', 'mushroom_accuracy']

# !!! Should add 'forecasted_assessment'
# Removed 'sessions_with_assessment_count'
X_test_gt_remainder_unchanged_list = ['previous_assessments_count']

# Difference in train set:
# X_test_gt_remainder_unchanged_list = ['Y_target', 'forecasted_assessment', 'previous_assessments_count', 'sessions_with_assessment_count'] 

# Difference in train set:
# We do not define X_test_gt_remainder and take all in X_test_gt
X_test_gt_sum = X_test_gt.groupby(['installation_id'], as_index=False, sort=False)[X_test_gt_remainder_sum_list].agg(sum)
X_test_gt_mean = X_test_gt.groupby(['installation_id'], as_index=False, sort=False)[X_test_gt_remainder_mean_list].agg('mean')
X_test_gt_unchaged = X_test_gt.groupby(['installation_id'], as_index=False, sort=False)[X_test_gt_remainder_unchanged_list].last()

# Merge both
X_test_gt_remainder = pd.merge(X_test_gt_sum, X_test_gt_mean, how='left', on=['installation_id'])
X_test_gt = pd.merge(X_test_gt_remainder, X_test_gt_unchaged, how='left', on=['installation_id'])

# Not applicable to test set:
# # Returning the installation_ids which had no previous assessments before the forecasted one
# #X_test_gt = pd.concat([X_test_gt_remainder, X_test_gt_last]).sort_index().reset_index(drop=True) index got broken while grouping by
# X_test_gt = X_test_gt_remainder.append(X_test_gt_last, ignore_index=True)

# # Questionable re sorting as it drops installation_id, need to test
# X_test_gt = pd.concat([X_test_gt_remainder, X_test_gt_last]).drop_duplicates('installation_id').reset_index(drop=True)


# In[153]:


X_test_gt.head(5)


# In[154]:


# debugging
debugging_ids(X_test_gt)


# In[155]:


# # !debugging, finding heavy user
# X_test_gt[X_test_gt['num_correct'] == X_test_gt.num_correct.max()]


# In[156]:


# # !debugging on heavy user
# X_test[(X_test['installation_id'] == '56a739ec') & (X_test['event_code'] == 4100) & (X_test['title'] == 'Cart Balancer (Assessment)')]


# ### Adding users w/o previous assessment attempts
# 
# * Test set specific as in train set we used only 'intallation_id's with at least one assessment attempt 

# In[157]:


test_features_list = X_test_gt.columns
X_test_gt.columns


# In[158]:


test_users_wo_assessments = set(X_test.installation_id) - set(X_test_gt.installation_id)
len(test_users_wo_assessments)


# ### Creating empty df matching test's columns
# 
# * Filled with 0
# * Alternatively could test with Nan, None or -1

# In[159]:


test_users_wo_assessments_df = pd.DataFrame(0, index=np.arange(len(test_users_wo_assessments)), columns=test_features_list)


# In[160]:


test_users_wo_assessments_df


# ### Adding 'installation_id's w/o prior assessments

# In[161]:


# We have created installation_id column with zero values. Now will assign missing installation_id
test_users_wo_assessments_df['installation_id'] = test_users_wo_assessments


# In[162]:


test_users_wo_assessments_df.head(2)


# ### (~T) Merging 'installation_id's with and w/o assessments

# In[163]:


X_test_gt = X_test_gt.append(test_users_wo_assessments_df, ignore_index=True)


# In[164]:


# debugging
debugging_ids(X_test_gt)


# In[165]:


# debugging
len(set(X_test_gt.installation_id))


# In[166]:


# debugging
# we lost the order of 'installation_id', but submission is sorted ascending
booltest_sub = X_test_gt.installation_id.sort_values(ascending=True).reset_index(drop=True) == submission.installation_id
set(booltest_sub)


# ### (T) Sorting to match order of submission

# In[167]:


X_test_gt = X_test_gt.sort_values('installation_id', ascending=True).reset_index(drop=True)


# In[168]:


X_test_gt.head(10)


# In[169]:


# debugging sorting
booltest_train = X_test_gt.installation_id == submission.installation_id
set(booltest_train)


# In[170]:


# debugging
debugging_ids(X_test_gt)


# ### (T) Adding 'forecasted_assessment' feature
# 
# * To both 'installation_id's with and w/o assessment attempt
# * It fixes initial bug where 'installation_id's w/o assessment attempt got their last attempted assessment as their 'forecasted_assessment' 

# In[171]:


# Create the forecasted_assessment_df which will contain all test set's installation_ids last forecasted_assessment

forecasted_assessment_df = X_test.groupby(['installation_id'], as_index=False, sort=False).agg('last')

# Reduce forecasted_assessment_df to users only w/o assessment (1000 -> 443):
# forecasted_assessment_df = forecasted_assessment_df[forecasted_assessment_df.installation_id.isin(test_users_wo_assessments)]
# Reseting the index, otherwise will get Nans when mapping:
# forecasted_assessment_df.reset_index()


# In[172]:


# forecasted_assessment_df.shape


# * Add 'forecasted_assessment' feature to the test set

# In[173]:


# Add forecasted_assessment number to X_test_gt:
# Map is how train set has assigned values to assessment titles:
# 0 Bird Measurer (Assessment)
# 1 Cart Balancer (Assessment)
# 2 Cauldron Filler (Assessment)
# 3 Chest Sorter (Assessment)
# 4 Mushroom Sorter (Assessment)
X_test_gt['forecasted_assessment'] = forecasted_assessment_df['title'].map({'Bird Measurer (Assessment)': 0,
                                                                                               'Cart Balancer (Assessment)': 1, 
                                                                                               'Cauldron Filler (Assessment)': 2, 
                                                                                               'Chest Sorter (Assessment)': 3, 
                                                                                    'Mushroom Sorter (Assessment)': 4})


# In[174]:


X_test_gt.head(2)


# In[175]:


# debugging
set(X_test_gt.forecasted_assessment), X_test_gt.forecasted_assessment.count()


# In[176]:


# debugging
debugging_ids(X_test_gt)


# In[177]:


# # debugging
# X_test_gt.loc[441, ['forecasted_assessment']]


# In[178]:


# # debugging
# X_test_gt.loc[441,]


# In[179]:


# # debugging OK - 'forecasted_assessment' of '779b71a3' is 'Chest Sorter (Assessment)' or encoded 3 
# X_test[X_test['installation_id'] == '779b71a3'].tail()


# # Adding none acc features to the test set

# ## (~T) event_code
# 
# #### Preparing event_code features

# In[180]:


# Uses ~3 GB of RAM for this operation (9->12->9)
X_test_eventcode = X_test.filter(['installation_id', 'event_code'], axis=1)
X_test_eventcode = event_code(X_test)

#### Merging event_code features to the main test set

# Add event_code features to the main dataframe
X_test_gt = pd.merge(X_test_gt, X_test_eventcode, on=['installation_id'])

del X_test_eventcode
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ## (~T) Title, type, world and event_code
# 
# #### Preparing title, type and world features

# In[181]:


# Create new X_test_titletypeworldfeat which holds time only title, type and world features
X_test_titletypeworldfeat = X_test.filter(['installation_id', 'title', 'type', 'world'], axis=1)
X_test_titletypeworldfeat = title_type_world(X_test_titletypeworldfeat)

#### Merging title, type and world features to the main test set

# Add title, type and world features to the main dataframe
X_test_gt = pd.merge(X_test_gt, X_test_titletypeworldfeat, on=['installation_id'])
# # Count nan in df for debugging purposes
# X_test_gt.isna().sum()

del X_test_titletypeworldfeat
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ## (~T) Other features
# 
# * all_actions_time
# 
# * Aggregate amount (in ms) of time spent on Assessments, Activities and Games
# * Clips do not have time spent feature
# 
# * all_actions_time
# * action_duration_mean (!!!)
# * event_code_count_mean
# * number_of_sessions_nu
# * event_count_mean (!!!)

# ###### all_actions_time

# In[182]:


# Creating all_actions_time (games, activities and assessments)
# RAM: 8.7->8.5-8.7 GB
feat_gametime = X_test[X_test['type'].isin(['Assessment', 'Game', 'Activity'])]
feat_gametime = feat_gametime.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
feat_gametime = feat_gametime.groupby('installation_id', as_index=False, sort=False)['game_time'].sum()

# Merging to the main test set
X_test_gt['all_actions_time'] = feat_gametime['game_time']

# Deleting
del feat_gametime
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ###### action_duration_mean

# In[183]:


# Creating action_duration_mean (games, activities and assessments) (!!!)
# RAM: 8.7->9.6->8.7 GB
feat_gametimemean = X_test[X_test['type'].isin(['Assessment', 'Game', 'Activity'])]
feat_gametimemean = feat_gametimemean.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
feat_gametimemean = feat_gametimemean.groupby('installation_id', as_index=False, sort=False)['game_time'].mean()

# Merging to the main test set
X_test_gt['action_duration_mean'] = feat_gametimemean['game_time']

# Deleting
del feat_gametimemean
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ###### event_code_count_mean

# In[184]:


# Creating event_code_count_mean (!!!)
# RAM: OK, flat
feat_eventcodecountmean = X_test.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['event_code'].count()
feat_eventcodecountmean = feat_eventcodecountmean.groupby('installation_id', as_index=False, sort=False)['event_code'].mean()

# Merging to the main test set
X_test_gt['event_code_count_mean'] = feat_eventcodecountmean['event_code']

# Deleting
del feat_eventcodecountmean
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ##### number_of_sessions_nu

# In[185]:


# Creating event_code_count_mean
# RAM: OK, flat
feat_numberofsessions = X_test.groupby(['installation_id'], as_index=False, sort=False)['game_session'].count()

# Merging to the main test set
X_test_gt['number_of_sessions_nu'] = feat_numberofsessions['game_session']

# Deleting
del feat_numberofsessions
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ##### event_count_mean

# In[186]:


# Creating event_count_mean (!!!)
# RAM: OK, flat
feat_eventcountmean = X_test.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['event_count'].last()
feat_eventcountmean = feat_eventcountmean.groupby('installation_id', as_index=False, sort=False)['event_count'].mean()

# Merging to the main test set
X_test_gt['event_count_mean'] = feat_eventcountmean['event_count']

# Deleting
del feat_eventcountmean
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ### (~T) timestamp
# 
# #### bug - taking the last even, which might be not assessment
# #### could replace with mean

# In[187]:


# RAM 8.7->10->9.3
# Create new X_test_timefeat which holds time only features  
feat_time = X_test.filter(['installation_id', 'timestamp'], axis=1)
# Prepare time features from given timestamp 
feat_time = timestamp_split(feat_time)

# Defining as last (bug)
feat_time = feat_time.groupby('installation_id', as_index=False).last()

# Merging to the main test set
X_test_gt = pd.merge(X_test_gt, feat_time, on=['installation_id'])

# Deleting
del feat_time
gc.collect()

#del X_test
gc.collect()

# debugging
debugging_ids(X_test_gt)


# ### (legacy) title, type and world

# In[188]:


# # Re-using f-ion used in train
# # Create new titletypeworldfeat which holds time only title, type and world features  
# X_test_titletypeworldfeat = X_test.filter(['installation_id', 'title', 'type', 'world'], axis=1)
# # Prepare title, type and world features from given timestamp 
# X_test_titletypeworldfeat = title_type_world(X_test_titletypeworldfeat)
# X_test_titletypeworldfeat


# In[189]:


# # debugging OK, 'ffe00ca8' has 5 rows in 'world' 'CRYSTALCAVES'
# X_test[(X_test['installation_id'] == 'ffe00ca8') & (X_test['world'] == 'CRYSTALCAVES')]


# ### (legacy) merge of timestamp, type, title and world features to main test set

# ##### debugging index before merger
# 
# * to avoid incorrectly assigning features from another 'installation_id's  

# In[190]:


# # debugging sorting of timefeat

# booltest_timefeat = X_test_gt.installation_id == X_test_timefeat.installation_id
# set(booltest_timefeat)


# In[191]:


# # debugging sorting of X_test_titletypeworldfeat

# booltest_titletypeworldfeat = X_test_gt.installation_id == X_test_titletypeworldfeat.installation_id
# set(booltest_titletypeworldfeat)


# In[192]:


# # debugging
# debugging_ids(X_test_gt)


# ##### merging time features

# In[193]:


# # debugging
# debugging_ids(X_test_timefeat)


# In[194]:


# # Merging new features to main test set

# # Add time features to the main dataframe
# X_test_gt = pd.merge(X_test_gt, X_test_timefeat, on=['installation_id'])


# In[195]:


# len(set(X_test_gt.installation_id)), X_test_gt.shape


# ##### merging titletypeworld features

# In[196]:


# # debugging - count nan in df - OK
# X_test_gt.isna().sum()


# In[197]:


# # Add title, type and world features to the main dataframe
# X_test_gt = pd.merge(X_test_gt, X_test_titletypeworldfeat, on=['installation_id'])


# In[198]:


# len(set(X_test_gt.installation_id)), X_test_gt.shape


# In[199]:


# # Count nan in df for debugging purposes
#set(X_test_gt.isna().sum())


# In[200]:


# # debugging sorting
# booltest_sub = X_test_gt.installation_id == submission.installation_id
# set(booltest_sub)


# #### Cleaning unused dfs and variables

# In[201]:


#del X_test, X_test_gt_remainder_sum_list, X_test_gt_remainder_mean_list, X_test_gt_remainder_unchanged_list, X_test_gt_sum, X_test_gt_mean, X_test_gt_unchaged, test_features_list, test_users_wo_assessments, test_users_wo_assessments_df, forecasted_assessment_df, X_test_timefeat, X_test_titletypeworldfeat
gc.collect()


# ## (~T) all_actions_time
# 
# * Aggregate amount (in ms) of time spent on Assessments, Activities and Games
# * Clips do not have time spent feature

# In[202]:


# #### Adding feature all_actions_time 
# feat_gametime_test = X_test[X_test['type'].isin(['Assessment', 'Game', 'Activity'])]
# #feat_gametime_test

# feat_gametime_test = feat_gametime_test.groupby(['installation_id', 'game_session'], as_index=False, sort=False)['game_time'].last()
# #feat_gametime_test

# feat_gametime_test = feat_gametime_test.groupby('installation_id', as_index=False, sort=False)['game_time'].sum()
# feat_gametime_test

# # debugging
# #X_test[X_test['installation_id'] == 'b37e2b2d']

# #feat_gametime_test[feat_gametime_test['installation_id'] == 'b37e2b2d']


# # Submission

# In[203]:


len(set(X_test_gt.installation_id)), X_test_gt.shape


# In[204]:


# debugging - check if df feature types
X_test_gt.info()


# In[205]:


# debugging sorting
booltest_sub = X_test_gt.installation_id == submission.installation_id
set(booltest_sub)


# In[206]:


# drop installation_id
X_test_gt = X_test_gt.drop(['installation_id'], axis=1)


# In[207]:


len(set(X_test_gt.index)), X_test_gt.shape


# In[208]:


# # cast forecasted assessment to str for cat_features
# X_test_gt['forecasted_assessment'] = X_test_gt['forecasted_assessment'].astype(str)
# type(X_test_gt.forecasted_assessment[0])
# X_train_gt.previous_assessments_count


# In[209]:


# Elsewise LightGBMError: Do not support special JSON characters in feature name.
X_test_gt.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test_gt.columns]


# In[210]:


list(X_test_gt.columns)


# #### Scaler

# In[211]:


X_test_gt_scaled = scaler.fit_transform(X_test_gt.astype(np.float64))


# In[212]:


# # Catboost clf submission
# # Important! X_test_gt_scaled added
# cbc_preds = cbc_model.predict(X_test_gt_scaled)
# submission['accuracy_group'] = cbc_preds.astype(int)
# submission.to_csv("submission.csv", index = False)
# submission.head()


# In[213]:


# #Catboost reg submission
# # Important! X_test_gt_scaled added
# cbc_preds = cbc_model.predict(X_test_gt_scaled)
# submission['accuracy_group'] = np.ceil(cbc_preds).astype(int)
# submission.to_csv("submission.csv", index = False)
# submission.head()


# ##### PCA for test set

# In[214]:


# Apply PCA for dimension reduction
#from sklearn.decomposition import PCA
#pca_test = PCA(n_components=10).fit(X_test_gt)
X_test_gt = pca.transform(X_test_gt)
print(sum(pca.explained_variance_ratio_))


# ##### Weighting and submission

# In[215]:


# LightGBM

submission = pd.read_csv(path + 'sample_submission.csv')
gbm_preds = gbm_model.predict(X_test_gt)

submission['accuracy_group'] = gbm_preds

submission['accuracy_group_weight0'] = np.where((submission['accuracy_group'] <= 0.81387600), 0, 0)
submission['accuracy_group_weight1'] = np.where((submission['accuracy_group'] > 0.81387600) & (submission['accuracy_group'] <= 1.09392700), 1, 0)
submission['accuracy_group_weight2'] = np.where((submission['accuracy_group'] > 1.09392700) & (submission['accuracy_group'] <= 1.42779600), 2, 0)
submission['accuracy_group_weight3'] = np.where((submission['accuracy_group'] > 1.42779600), 3, 0)
submission['accuracy_group'] = submission['accuracy_group_weight0'] + submission['accuracy_group_weight1'] + submission['accuracy_group_weight2'] + submission['accuracy_group_weight3']
submission = submission.drop(['accuracy_group_weight0', 'accuracy_group_weight1', 'accuracy_group_weight2', 'accuracy_group_weight3'], axis=1)

submission.to_csv("submission.csv", index = False)

submission.accuracy_group.value_counts()


# In[216]:


submission.accuracy_group.value_counts()


# In[217]:


# # xgboost submission
# xgb_preds = xgb_model.predict(X_test_gt)
# submission['accuracy_group'] = xgb_preds.astype(int)
# submission.to_csv("submission.csv", index = False)
# submission.head()


# In[218]:


# LightGBM submission
# gbm_preds = gbm_model.predict(X_test_gt)
# submission['accuracy_group'] = np.round(gbm_preds).astype(int)
# submission.to_csv("submission.csv", index = False)
# submission.head()
# submission.accuracy_group.value_counts()


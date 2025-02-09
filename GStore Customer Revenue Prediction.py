#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


csv_path = 'D:/GoogleProject/train.csv'  # Replace with your actual file path

dataframe = pd.read_csv(csv_path, encoding ='utf8')

dataframe.info()


# In[3]:


small_df = dataframe.copy()

#small_df.info()


# In[4]:


# List of columns to drop
columns_to_drop = [
    'sessionId', 
    'socialEngagementType'
]

# Drop the unnecessary columns
small_df = small_df.drop(columns=columns_to_drop)

# Display the resulting dataframe
small_df.head(5)


# In[5]:


print(small_df['totals'].head(1))


# In[6]:


import pandas as pd
from pandas import json_normalize


# In[7]:


# Assuming 'small_df' is already loaded
import pandas as pd
from pandas import json_normalize
from ast import literal_eval 
# Flatten the 'totals' and 'geoNetwork' columns if they are in JSON format using literal_eval
df_totals = json_normalize(small_df['totals'].apply(literal_eval))  # safely parse JSON-like strings
df_geoNetwork = json_normalize(small_df['geoNetwork'].apply(literal_eval))  # safely parse JSON-like strings



# Now, drop the original columns and concatenate the flattened versions
small_df = small_df.drop(columns=['totals', 'geoNetwork'])  # Drop the original JSON columns
small_df = pd.concat([small_df, df_totals, df_geoNetwork], axis=1)  # Concatenate the flattened columns


# Finally, convert the DataFrame to CSV format
small_df.to_csv('flattened_google_analytics.csv', index=False)

# Display the first few rows of the flattened DataFrame
print(small_df.head())


# In[8]:


small_df.info()


# In[9]:


# List of columns to keep for customer revenue prediction
columns_to_keep = [
    'channelGrouping',
    'device',
    'fullVisitorId',
    'trafficSource',
    'visitStartTime',
    'transactionRevenue',
    'continent',
    'subContinent',
    'country',
    'region',
    'metro',
    'city',
    'newVisits',
    'bounces'
]

# Drop all columns not in the 'columns_to_keep' list
small_df_reduced = small_df[columns_to_keep]


# In[10]:


small_df_reduced.info()


# In[11]:


small_df_reduced.head(1)


# In[12]:


import pandas as pd
from ast import literal_eval
from pandas import json_normalize

# Sample DataFrame
# small_df_reduced = pd.read_csv('path_to_your_data.csv')

# Function to safely evaluate and parse the JSON-like strings
def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return {}

# Safely parse the 'device' and 'trafficSource' columns
df_device = json_normalize(small_df_reduced['device'].apply(safe_literal_eval))  # Flatten 'device' column
df_trafficSource = json_normalize(small_df_reduced['trafficSource'].apply(safe_literal_eval))  # Flatten 'trafficSource' column

# Drop original columns from small_df_reduced and merge the flattened columns
small_df_reduced = small_df_reduced.drop(['device', 'trafficSource'], axis=1)

# Concatenate the original DataFrame with the flattened 'device' and 'trafficSource' columns
small_df_final = pd.concat([small_df_reduced, df_device, df_trafficSource], axis=1)

# Save the final DataFrame to CSV
small_df_final.info()


# In[13]:


# Filter the DataFrame to keep only rows where 'transactionRevenue' is non-null
filtered_df = small_df_final[small_df_final['transactionRevenue'].notna()]

# Verify the result by checking the first few rows of the filtered DataFrame
filtered_df.head()

# Save the filtered DataFrame to a new CSV
filtered_df.to_csv('filtered_google_analytics.csv', index=False, encoding='utf-8')


# In[14]:


filtered_df.info()


# In[15]:


filtered_df.head()


# In[17]:


# Define the columns to drop
columns_to_drop = [
    'device', 'campaign', 'source', 'medium', 'keyword',
    'adwordsClickInfo.criteriaParameters', 'referralPath',
    'adContent', 'adwordsClickInfo.gclId', 'campaignCode'
]

# Drop the columns
filtered_df = filtered_df.drop(columns=columns_to_drop)

# Verify the result by checking the first few rows
filtered_df.head()

# Save the updated DataFrame to a new CSV
filtered_df.to_csv('filtered_google_analytics_updated.csv', index=False, encoding='utf-8')


# In[18]:


filtered_df.info()


# In[19]:


filtered_df.head()


# In[20]:


# Define the columns to drop
columns_to_drop = ['newVisits', 'bounces']

# Drop the columns
filtered_df = filtered_df.drop(columns=columns_to_drop)

# Verify the result by checking the first few rows
filtered_df.head()

# Save the updated DataFrame to a new CSV
filtered_df.to_csv('filtered_google_analytics_final.csv', index=False, encoding='utf-8')


# In[21]:


filtered_df.head()


# In[22]:


filtered_df.info()


# In[23]:


from sklearn.model_selection import train_test_split

# Split the data into train (80%) and test (20%)
train_data, test_data = train_test_split(filtered_df, test_size=0.2, random_state=42)

# Verify the shapes of the datasets
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Save the train and test data to separate CSV files
train_data.to_csv('google_analytics_train.csv', index=False, encoding='utf-8')
test_data.to_csv('google_analytics_test.csv', index=False, encoding='utf-8')


# In[24]:


train_data.head()


# In[25]:


# Ensure you work on a copy of the train_data
train_data = train_data.copy()

# Convert the 'transactionRevenue' column to float type
train_data["transactionRevenue"] = train_data["transactionRevenue"].astype('float')

# Group the data by 'fullVisitorId' and calculate the sum of 'transactionRevenue'
grouped_df = train_data.groupby("fullVisitorId")["transactionRevenue"].sum().reset_index()

# Plot the sorted log-transformed transaction revenues
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(range(grouped_df.shape[0]), np.sort(np.log1p(grouped_df["transactionRevenue"].values)), alpha=0.5)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Log(TransactionRevenue + 1)', fontsize=12)
plt.title('Log-Transformed Transaction Revenue per Visitor', fontsize=14)
plt.show()


# In[26]:


# Count the number of instances in the train set with non-zero revenue
nzi = pd.notnull(train_data["transactionRevenue"]).sum()

# Count the number of unique customers with non-zero total transaction revenue
nzr = (grouped_df["transactionRevenue"] > 0).sum()

# Print the results
print("Number of instances in train set with non-zero revenue:", nzi, "and ratio is:", nzi / train_data.shape[0])
print("Number of unique customers with non-zero revenue:", nzr, "and the ratio is:", nzr / grouped_df.shape[0])


# In[27]:


# Calculate the number of unique visitors in train_data
print("Number of unique visitors in train set:", train_data.fullVisitorId.nunique(), "out of rows:", train_data.shape[0])

# Calculate the number of unique visitors in test_data
print("Number of unique visitors in test set:", test_data.fullVisitorId.nunique(), "out of rows:", test_data.shape[0])

# Calculate the number of common visitors in both train_data and test_data
common_visitors = len(set(train_data.fullVisitorId.unique()).intersection(set(test_data.fullVisitorId.unique())))
print("Number of common visitors in train and test set:", common_visitors)


# In[28]:


# Identify columns in train_data with constant values
const_cols = [col for col in train_data.columns if train_data[col].nunique(dropna=False) == 1]
const_cols


# In[29]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Assuming train_data is already loaded and preprocessed
# Split data into features (X) and target (y)
X = train_data.drop(["transactionRevenue", "fullVisitorId"], axis=1)  # Drop target and unique identifier
y = train_data["transactionRevenue"].fillna(0)  # Replace missing revenues with 0

# One-hot encoding for categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")






# In[30]:


import lightgbm as lgb
print(lgb.__version__)


# In[ ]:





# In[ ]:





# In[ ]:





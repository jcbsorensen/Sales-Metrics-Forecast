import numpy as np, pandas as pd
import matplotlib.pyplot as plt


## Fill COVID closing period 2020-Mar -> 2020-Jun with Median value
def mend_closed_period(df, store_list, product_list):
    tmp = df.copy()
    for store in store_list:
        for product in product_list:
            median = df.loc[(df['Salesdate'] >= '2020-02-01') & (df['Salesdate'] < '2020-03-01') & (df['Team Name'] == store) & (df['Product Subcategory'] == product), 'Sales'].median()
            if (pd.isna(median)):
                median = 0
            tmp.loc[(df['Team Name'] == store) & (df['Product Category'] == product) & (df['Salesdate'] >= '2020-03-17') & (df['Salesdate'] <= '2020-05-10'), 'Sales'] = median
            if (tmp['Sales'].isna().values.any()):
                print('warning', ' - ', store, product)
    return tmp


def DataPrepping(raw_data):
    # Reduce Data
    print('Reducing Data....')
    sales_df = raw_data.loc[:'2020-10-31']
    # sales_df = sales_df.loc[sales_df.Sales < 250]
    sales_df['Salesdate'] = sales_df.index
    print('Reducing Data - Completed')

    # Select only team names that have sold anything the last few months
    print('Selecting only relevant stores....')
    relevant_stores = sales_df.loc[(sales_df.index >= '2020-08-01') & (sales_df.Sales != 0), 'Team Name'].unique()
    sales_df = sales_df.loc[sales_df['Team Name'].isin(relevant_stores)]
    sales_df = sales_df.loc[sales_df['Team Name'] != '12410 - Telenor Haderslev']  # closed store
    print('Selecting only relevant stores - Completed')

    # Mend closed store periods
    #print('Mend closed store periods with median....')
    #products = sales_df['Product Subcategory'].unique()
    #stores = ['12304 - Telenor Herningcentret', '12403 - Telenor Kolding Storcenter', '11109 - Telenor Fisketorvet',
    #          '11408 - Telenor Næstved Storcenter', '83056 - Telenor i Bilka Slagelse',
    #          '11406 - Telenor Slagelse', '83057 - Telenor i Bilka Næstved', '11210 - Telenor Helsingør',
    #          '11127 - Telenor Frederiksberg Centret', '11211 - Telenor Hillerød Slotsarkaderne',
    #          '83055 - Telenor i Bilka Ishøj', '83063 - Telenor i Bilka Horsens',
    #          '11504 - Telenor Odense Rosengårdscent.', '83002 - Telenor i Bilka Sønderborg',
    #          '83006 - Telenor i Bilka Herning',
    #          '83065 - Telenor i Bilka Viborg', '12215 - Telenor Århus Storcenter Nord',
    #          '83062 - Telenor i Bilka Kolding', '12104 - Telenor Aalborg Storcenter', '83051 - Telenor i Bilka Tilst',
    #          '83058 - Telenor i Bilka Holstebro', '12206 - Telenor Århus Bruuns Galleri', '11113 - Telenor Fields']
    #sales_df = mend_closed_period(df=sales_df, store_list=stores, product_list=products).copy()
    #print('Mend closed store periods with median - Completed')

    ## Remove Elgiganten rows with products they dont sell
    print('Remove Elgiganten specific products....')
    sales_df.reset_index(inplace=True)  # remove the date index (otherwise too many rows will be deleted)
    sales_df.drop(sales_df.loc[sales_df['Team Name'].str.contains('Elgiganten') & (sales_df['Product Subcategory'] == 'NM') & (sales_df['Product Category'] == 'Other Products')].index, inplace=True)
    sales_df.drop(sales_df.loc[sales_df['Team Name'].str.contains('Elgiganten') & (sales_df['Product Category'].isin(['Broadband', 'MBB', 'HW Only', 'Accessories', 'Insurance']))].index, inplace=True)
    sales_df.set_index('Date', inplace=True)
    print('Remove Elgiganten specific products - Completed')

    # Differentiate the 2 types of NM
    sales_df.loc[(sales_df['Product Category'] == 'Voice') & (sales_df['Product Subcategory'] == 'NM'), 'Product Subcategory'] = 'NM Voice'
    sales_df.loc[(sales_df['Product Category'] == 'MBB') & (sales_df['Product Subcategory'] == 'NM'), 'Product Subcategory'] = 'NM MBB'
    print('Differentiate NM - Completed')

    # Remove Team Names with too many zeros
    sales_df = sales_df.loc[~sales_df['Team Name'].isin(['Order', 'SoMe', 'Channel Support', 'KAS', '11304 - Telenor Roskilde Ros Torv', 'Implementation', 'Written'])]
    print('Remove teams with too many zeros - Completed')

    return sales_df

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:56:16 2021

@author: Home
"""


import pandas as pd
import numpy as np
df  = pd.read_csv("casestudy.csv")


# total Revenue #
total_rev = df[['net_revenue','year']]
total_rev = total_rev.astype({"year": str})
total_rev = total_rev.groupby("year")
total_rev = pd.DataFrame({'net_revenue' : total_rev['net_revenue'].sum()}).reset_index()
total_rev = total_rev.sort_values(by=['year'],ascending =True)

#we can automate it would just require iterating over the Dataframe


first_year= total_rev['year'][0] 
second_year= total_rev['year'][1] 
third_year= total_rev['year'][2] 

fy_df = df[df.year == 2015]
sy_df = df[df.year == 2016]
ty_df = df[df.year == 2017]


#comparing the two emails columns from the dataframes



# left join on dataframes


cust_df_year2015_2016 = sy_df.merge(fy_df,how = 'left',left_on = 'customer_email',right_on = 'customer_email')
existing_cust2016 = cust_df_year2015_2016[cust_df_year2015_2016['year_y'] == 2015.0]
existing_cust2016['revenue_diff']= existing_cust2016['net_revenue_x'] - existing_cust2016['net_revenue_y']
new_cust_2016 = cust_df_year2015_2016[cust_df_year2015_2016['year_y'] != 2015.0]


cust_df_year2016_2017 = ty_df.merge(sy_df,how = 'left',left_on = 'customer_email',right_on = 'customer_email')
existing_cust2017 = cust_df_year2016_2017[cust_df_year2016_2017['year_y'] == 2016.0]
existing_cust2017['revenue_diff']= existing_cust2017['net_revenue_x'] - existing_cust2017['net_revenue_y']

new_cust_2017 = cust_df_year2016_2017[cust_df_year2016_2017['year_y'] != 2016.0]

lis = [['2015','0']]



new_cust_revenue =pd.DataFrame(lis, columns = ['year_x','net_revenue_x'])
new_cust_revenue = pd.concat([new_cust_revenue,new_cust_2016[['year_x','net_revenue_x']]])
new_cust_revenue = pd.concat([new_cust_revenue,new_cust_2017[['year_x','net_revenue_x']]])

new_cust_revenue = new_cust_revenue.astype({"year_x": str})
new_cust_revenue = new_cust_revenue.groupby("year_x")
new_cust_revenue = pd.DataFrame({'net_revenue' : new_cust_revenue['net_revenue_x'].sum()}).reset_index()
new_cust_revenue =new_cust_revenue.sort_values(by=['year_x'],ascending =True)


#existing cust growth


exixting_cust_growth =pd.DataFrame(lis, columns = ['year_x','revenue_diff'])
existing_cust_growth = pd.concat([exixting_cust_growth,existing_cust2016[['year_x','revenue_diff']]])
existing_cust_growth = pd.concat([existing_cust_growth,existing_cust2017[['year_x','revenue_diff']]])

existing_cust_growth = existing_cust_growth.astype({"year_x": str})
existing_cust_growth = existing_cust_growth.groupby("year_x")
existing_cust_growth = pd.DataFrame({'net_revenue' : existing_cust_growth['revenue_diff'].sum()}).reset_index()
existing_cust_growth =existing_cust_growth.sort_values(by=['year_x'],ascending =True)

#existing Cust Revenue


existing_cust_revenue = existing_cust2016[['year_y','net_revenue_y']]
existing_cust_revenue = existing_cust_revenue.rename(columns={"year_y": "year_x", "net_revenue_y": "net_revenue_x"})
existing_cust_revenue = pd.concat([existing_cust_revenue,existing_cust2016[['year_x','net_revenue_x']]])
existing_cust_revenue = pd.concat([existing_cust_revenue,existing_cust2017[['year_x','net_revenue_x']]])

existing_cust_revenue = existing_cust_revenue.astype({"year_x": str})
existing_cust_revenue = existing_cust_revenue.groupby("year_x")
existing_cust_revenue = pd.DataFrame({'net_revenue' : existing_cust_revenue['net_revenue_x'].sum()}).reset_index()
existing_cust_revenue =existing_cust_revenue.sort_values(by=['year_x'],ascending =True)

#existing_cust_revenue_prior year.

existing_cust_revenue_prior = np.asarray(existing_cust_revenue)

existing_cust_revenue_prior = pd.DataFrame(existing_cust_revenue_prior, columns = ['year_x','revenue-x'])
existing_cust_revenue_prior['revenue-x'][2] = existing_cust_revenue_prior['revenue-x'][1]
existing_cust_revenue_prior['revenue-x'][1] = existing_cust_revenue_prior['revenue-x'][0]
existing_cust_revenue_prior['revenue-x'][0]  = 0


#total Customer Current year
total_cust = df[['customer_email','year']]
total_cust = total_cust.astype({"year": str})
total_cust = total_cust.groupby("year")
total_cust = pd.DataFrame({'count' : total_cust['customer_email'].size()}).reset_index()
total_cust = total_cust.sort_values(by=['year'],ascending =True)

#total Customer previous year
total_cust_prev = np.asarray(total_cust)

total_cust_prev = pd.DataFrame(total_cust_prev, columns = ['year_x','count'])
total_cust_prev['count'][2] = total_cust_prev['count'][1]
total_cust_prev['count'][1] = total_cust_prev['count'][0]
total_cust_prev['count'][0]  = 0

#New Customers

new_cust_2016
new_cust_2017

lis = [['2015','0'],['2016',new_cust_2016['customer_email'].count()],['2017',new_cust_2017['customer_email'].count()]]

new_cust = pd.DataFrame(lis, columns = ['year_x','count'])

#lost_customer

lis = [['2015','0'],['2016',(total_cust['count'][1] - total_cust['count'][0])],['2017',(total_cust['count'][2] - total_cust['count'][1])]]
lost_cust = pd.DataFrame(lis, columns = ['year_x','count'])


#Revenue lost

rev_lost2016 = fy_df.merge(sy_df,how = 'left',left_on = 'customer_email',right_on = 'customer_email')
rev_lost20161 = rev_lost2016[rev_lost2016['year_y'] != 2016.0]


rev_lost2017 = sy_df.merge(ty_df,how = 'left',left_on = 'customer_email',right_on = 'customer_email')
rev_lost20171 = rev_lost2017[rev_lost2017['year_y'] != 2017.0]

lis = [['2015','0'],['2016',rev_lost20161['net_revenue_x'].sum()],['2017',rev_lost20171['net_revenue_x'].sum()]]
lost_rev = pd.DataFrame(lis, columns = ['year_x','value'])




#
total_rev = total_rev.rename(columns={"year": "year", "net_revenue": "Current Year Revenue"})
total_rev.to_csv("total_rev.csv")

new_cust_revenue = new_cust_revenue.rename(columns={"year_x": "year", "net_revenue": "New Customer Revenue"})
new_cust_revenue.to_csv("new_cust_revenue.csv")


existing_cust_growth = existing_cust_growth.rename(columns={"year_x": "year", "net_revenue": "Existing Customer Growth"})
existing_cust_growth.to_csv("existing_cust_growth.csv")

lost_rev = lost_rev.rename(columns={"year_x": "year", "value": "Revenue Lost from attrition"})
lost_rev.to_csv("lost_rev.csv")

existing_cust_revenue =existing_cust_revenue.rename(columns={"year_x": "year", "net_revenue": "existing_cust_revenue_Current_year"})
existing_cust_revenue.to_csv("existing_cust_revenue.csv")



existing_cust_revenue_prior =  existing_cust_revenue_prior.rename(columns={"year_x": "year", "revenue-x": "existing_cust_revenue_Prior_year"})
existing_cust_revenue_prior.to_csv("existing_cust_revenue_prior.csv")

total_cust=total_cust.rename(columns={"year_x": "year", "count": "Total customer this year"})
total_cust.to_csv("total_cust.csv")

total_cust_prev=total_cust_prev.rename(columns={"year_x": "year", "count": "Total customer Prior year"})
total_cust_prev.to_csv("total_cust_prev.csv")

new_cust = new_cust.rename(columns={"year_x": "year", "count": "New customer this year"})
new_cust.to_csv("new_cust.csv")

lost_cust = lost_cust.rename(columns={"year_x": "year", "count": "Lost Customer this year"})
lost_cust.to_csv("lost_cust.csv")



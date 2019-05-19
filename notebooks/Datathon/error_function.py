#!/usr/bin/python
"""
Copyright 2019 Accenture and/or its affiliates.  All Rights Reserved.  
You may not use, copy, modify, and/or distribute this code and/or its documentation without permission from Accenture.
Please contact the Advanced Analytics-Operations Analytics team and/or Frode Huse Gjendem (lead) with any questions.

\brief This is the starter script for the Accenture's Health Datathon 2019 competition.

\version 1.0

\date $Date: 2019/05

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



'''
Function to select certain/uncertain weights
You will not need to ever create the following parameters
They are created (and this is called) within the next function "calc_weights"
@paramters
    patient_info: vector holding every month you want to evaluate, 1 for whichever column months_survived = 1, NA otherwise
    df_aux: vector that contains the probability of survival at each month, fixed values for a test set
@return
    the vector of weights

'''
def get_weights(patient_info, df_aux):
    x = np.array(patient_info)
    
    # weight will always be 1 if we know they are dead
    if (x[~np.isnan(x)][0]==1):
        x = np.ones_like(x)
    else:
        y = np.argwhere(~np.isnan(x))
        not_zero_index = y.ravel()[0]
        
        x[:not_zero_index] = 1
        x[not_zero_index:] = df_aux['prob'][not_zero_index:]
    return (x)


'''
Function to calculate the weight matrix
@paramters
    y_df: The test set for which you are calculating the weights
        Format: Index = ID, Rows = patients, cols = ['specific_death', 'months_survival']
@ return
    the weights matrix with the weights for each patient at each time t  
'''
def calc_weights(y_df):
    
    # Create a matrix with patient id in the index and months_survival as header, specific_death as values.   
    df_y_pivot = y_df.pivot(columns="months_survival", values='specific_death')
    # The table changes order of rows after pivoting it, we need to reorder it again by doing reindex_axis.
    df_y_pivot= df_y_pivot.reindex(y_df.index, axis=0).reset_index()
    
    # We need to calculate the weights based on the entire time initially, then cut it off after the fact
    all_months_survival = np.arange(0,y_df.months_survival.max()+1)
    months_complementary = np.setdiff1d(all_months_survival, y_df.months_survival.unique())
    df_complementary = pd.DataFrame(np.nan, index=df_y_pivot.index, columns=months_complementary )
    df_y_pivot = pd.concat([df_y_pivot,df_complementary],axis=1)[all_months_survival]
    
    # Get aux matrix to provide in the get_weights function. 
    # Probability of being alive at each month based on patients you are certain about (excluding patients censored pior to month)
    df_aux = pd.DataFrame(data=np.arange(0,y_df.months_survival.max()+1),columns=["months_survival"])
    df_aux['prob_num'] = df_aux['months_survival'].apply(lambda x : (y_df['months_survival'] > x).values.sum())
    df_aux['prob_den'] = df_aux['months_survival'].apply(lambda x : ((y_df['months_survival'] < x) & (y_df['specific_death']==1)).values.sum())
    df_aux['prob'] = (df_aux['prob_num']/(df_aux['prob_num']+df_aux['prob_den']))

    df_aux = df_aux[['months_survival','prob']].sort_values('months_survival').reset_index().drop('index',axis=1)

    
    #Get weights
    df_weights = df_y_pivot.apply(lambda x: get_weights(x,df_aux),axis=1)
    
    new_weights = pd.DataFrame(np.vstack(df_weights),columns=all_months_survival )

    new_weights = np.apply_along_axis(np.cumprod, 1, new_weights)
    
    new_weights = pd.DataFrame(new_weights)
    
    return new_weights


''' 
Fill up the Y_true matrix's value 
You will not need to ever creat the following parameter.
It is created (and this is called) within the next function "populate_actual".
@paramters
    patient_info: vector holding every month you want to evaluate, 1 for whichever column months_survived = 1, NA otherwise
'''
def apply_non_zero(patient_info):
    x = np.array(patient_info)
    if (x[~np.isnan(x)][0]==0):
        x = np.ones_like(x)
    else:
        y = np.argwhere(~np.isnan(x))
        not_zero_index = y.ravel()[0]
        x[:(not_zero_index)] = 1
        x[(not_zero_index):] = 0   

    return (pd.Series(x))

 
'''
Build the Y_true matrix
@paramters
    y_df: The test set for which you are calculating the weights
        Format: Index = ID, Rows = patients, cols = ['specific_death', 'months_survival']
    years: Number of years for which you want to evaluate your model. Default = 10.
'''
def populate_actual(y_df, years = 10):
    #Create a matrix by pivoting table
    df_y_true_pivot = y_df.pivot(columns = "months_survival", values='specific_death')
    #The table changes order of rows after pivoting it, we need to reorder it again by doing reindex_axis.
    df_y_true_pivot = df_y_true_pivot.reindex(y_df.index, axis=0).reset_index()


    all_months_survival = np.arange(0,y_df.months_survival.max()+1)
    #Get the month that we don't have any patient record in our dataset.
    
    months_complementary = np.setdiff1d(all_months_survival, y_df.months_survival.unique())
    df_complementary = pd.DataFrame(np.nan, index=df_y_true_pivot.index, columns=months_complementary )
    
    #Add the complementary dataframe to create a full-month dataframe
    df_y_true_pivot = pd.concat([df_y_true_pivot,df_complementary],axis=1)[all_months_survival]

    # Fill NaN value to either 0 or 1
    df_y_pro = df_y_true_pivot.apply(lambda x: apply_non_zero(x),axis=1)
    
    
    chosen_months_survival = np.arange(0,years*12+1)
    df_y_pro = df_y_pro[chosen_months_survival].values
    
    return df_y_pro


'''
Function to compute the weighted Brier score
@paramters
    pred: is the prediction matrix
        Format: Index = ID, Rows = patients, cols = months, values = probability of being alive (alive = 1, dead = 0)
    actual: is the actual value matrix.
        Format: Index = ID, Rows = patients, cols = ['specific_death', 'months_survival']
    weights: is the matrix of weights associated to the predictions
    years_cutoff: is the time for which the error is computed
@return
    the vector of actual status over time
'''
def brier_score_loss_weighted(pred, actual, weights, years_cutoff = 10):
    
    # Select the desired period
    weights = weights.iloc[:,:(12*years_cutoff)+1]
    
    # obtain the unique time values in the y data of your survival data
    unique_times = range(0, (12*years_cutoff)+1)
    
    # fill an empty matrix to hold the weights
    errors = np.empty([len(actual), len(unique_times)])
    
    # subset y_pred to be the number of years you want
    m_y_pred = pred.iloc[:,:(max(unique_times)+1)]

    try:
        m_y_true = np.matrix(populate_actual(actual, years_cutoff))
        m_y_pred = np.matrix(m_y_pred)
        m_weights = np.matrix(weights)
    except:
        print("Matrix format is required for y_true, y_predict and weights")
            
    errors = np.multiply(np.power(m_y_pred - m_y_true,2),m_weights)
    
    error = pd.DataFrame(errors)
    time = years_cutoff * 12
    
    # calculate the average error of all patients for each time
    all_dates = error.mean(axis=0)
    
    # subset desired dates
    desired_dates = pd.DataFrame(all_dates[0:(time+1)])
    
    # calculate the average error up until a point in time
    desired_error = np.mean(desired_dates)
    
    return desired_error










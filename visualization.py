#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:42:58 2018

@author: moeen
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


## Histogram of the target and five other features

def plot_1(data):
    
    
    f, ax1 = plt.subplots(3,2)
    
    ax1[0,0].hist(data['y'], bins = 50)
    ax1[0,0].set_title("Y")
    ax1[0,1].hist(data['X1'], bins = 50)
    ax1[0,1].set_title("X1")
    ax1[1,0].hist(data['X50'], bins = 50)
    ax1[1,0].set_title("X50")
    ax1[1,1].hist(data['X100'], bins = 50)
    ax1[1,1].set_title("X100")
    ax1[2,0].hist(data['X200'], bins = 50)
    ax1[2,0].set_title("X200")
    ax1[2,1].hist(data['X250'], bins = 50)
    ax1[2,1].set_title("X250")
    
    plt.tight_layout()
    plt.savefig("histogram.png", format="png", dpi=1000)
    plt.show()
    


## Plot catagorical variable X256 and numeric variable X8
    
def plot_2(data):
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    #Make list of unique catagorical variables 
    colors = list(data.X256.unique())
    
    for color in colors:
        # Select the building type
        subset = data[data['X256'] == color]

        # Density plot of Energy Star scores
        sns.kdeplot(subset['y'], label = color, shade = False, alpha = 0.9, ax=ax1);
    
    ax1.set_title('Density Plot (X256)')  
    
    # y vs X8 plot
    
    ax2.scatter(data['y'], data['X8'])
    ax2.set_title('Y Vs X8')
    ax2.set_xlabel("Y")
    ax2.set_ylabel("X8")
    fig = plt.gcf()
    
    fig.set_size_inches(12.5, 5.5)
    #fig.savefig('test2png.png', dpi=100)
    fig.savefig("x256_x8.png", format="png", dpi=1000)
    plt.show()
    

## Comparison between model error

def plot_3(model_names, errors):
    
    index = np.arange(4)
    bar_width = 0.4
    
    fig, ax = plt.subplots()
    ax.bar(index, errors, bar_width)
    ax.set_title('Model Comparison')
    ax.set_xlabel('Models')
    ax.set_ylabel('MSE')
    ax.set_xticks(index)
    ax.set_xticklabels(model_names, rotation = 'horizontal')
    plt.tight_layout()
    plt.savefig("comparison.png", format="png", dpi=1000)
    plt.show()
    
    
    
def plot_4(t_test, lr_pred, poly_pred, rf_pred, gr_pred, flag):
    
    plt.scatter(t_test, lr_pred, alpha = 0.8, label = "LinearRegression")
    plt.scatter(t_test, poly_pred, alpha = 0.8, label = "Polynomial")
    plt.scatter(t_test, rf_pred, alpha = 0.8, label = "RandomForest")
    plt.scatter(t_test, gr_pred, alpha = 0.8, label = "GradientBoost")
    plt.legend(loc = 'upper left', fontsize = 18, frameon = 0)
    if flag == 1:
        plt.title('Prediction Vs Real Value on validation data')
    else:
        plt.title('Prediction Vs Real Value on test data')
    plt.xlabel("Given Values")
    plt.ylabel("Predictions")
    plt.tight_layout()
    
    if flag == 1:
        plt.savefig("pred_validation.png", format="png", dpi=1000)
    else:
        plt.savefig("pred_test.png", format="png", dpi=1000)
    

    plt.show()
    
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib





#kde plot
#====================

def create_kde_plot(selected_features_df, selected_features):
   
    fig, axes = plt.subplots(3, 2,figsize=(12, 16))
    axes = axes.flatten()


    for i, feature in enumerate(selected_features):

        sns.kdeplot(
            data = selected_features_df,
            x = feature,
            hue = "diagnosis",
            ax = axes[i]
        )

        axes[i].set_title(f'{feature} vs Diagnosis', fontsize=12)

        lines = axes[i].get_lines()
        colors = ['blue' , 'orange']
        for line , color in zip(lines,colors):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
            if len(y_data) > 0:
                max_idx = np.argmax(y_data)
                peak_x = x_data[max_idx]
                peak_y = y_data[max_idx]
                
                axes[i].axvline(x=peak_x, color=color, linestyle=':', alpha=0.7)
                axes[i].text(peak_x,peak_y*0.8,f'{peak_x:.3f}',color=color,fontsize=20,ha='center',va='bottom',)

    
    fig.delaxes(axes[5])
    plt.tight_layout()
    
    return fig

# box plot
#========================

def create_box_plot(selected_features_df, selected_features):

    fig , axes = plt.subplots(3,2,figsize=(12, 16))
    axes = axes.flatten()

    for i,feature in enumerate(selected_features):
        
        sns.boxplot(data = selected_features_df,
        x = feature,
        hue = 'diagnosis',
        ax = axes[i],
        showmeans = True)

        axes[i].set_title(f'{feature} vs Diagnosis', fontsize=12)


    fig.delaxes(axes[5])
    plt.tight_layout()
    
    return fig

#violin plot
# =========== 

def create_violin_plot(selected_features_df, selected_features):
    fig , axes = plt.subplots(3,2 , figsize = (12,16))
    axes = axes.flatten()
    for i,feature in enumerate(selected_features):
        sns.violinplot(data = selected_features_df , x = feature , hue = 'diagnosis' , ax = axes[i] , inner = 'quartile')
        axes[i].set_title(f'{feature} vs Diagnosis', fontsize=12)

    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.show()
    return fig

# heat map
#================
def create_heat_map(selected_features_df):
    
    selected_features_df["diagnosis"] = selected_features_df["diagnosis"].map({
    "Benign": 0,
    "Malignant": 1
    })


    
    fig , axes = plt.subplots(figsize = (6,6))
    sns.heatmap(data = selected_features_df.corr() , vmin = -1 , vmax = 1 , center = 0,annot = True)

    return fig



def create_bar_plot(selected_features_df, selected_features):
    corr = selected_features_df[selected_features + ['diagnosis']].corr()['diagnosis']
    corr = corr.drop('diagnosis').sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(x=corr.index, y=corr.values)
    for i, value in enumerate(corr.values):
        plt.text(i,value + 0.02,f'{value:.2f}',ha='center')
        
    plt.title('Correlation of Top 5 Features with Diagnosis')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    
    return plt



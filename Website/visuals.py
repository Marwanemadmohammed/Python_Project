import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





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
    return fig

# heat map
#================

def create_heat_map(selected_features_df):
    plot_data = selected_features_df.copy()

    plot_data["diagnosis"] = plot_data["diagnosis"].map({
        "Benign": 0,
        "Malignant": 1
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        plot_data.corr(numeric_only=True),
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        ax=ax
    )

    fig.tight_layout()
    return fig





def create_bar_plot(selected_features_df, selected_features):
    corr = (
        selected_features_df[selected_features + ["diagnosis"]]
        .copy()
    )

    corr["diagnosis"] = corr["diagnosis"].map({
        "Benign": 0,
        "Malignant": 1
    })

    correlations = (
        corr.corr(numeric_only=True)["diagnosis"]
        .drop("diagnosis")
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        x=correlations.index,
        y=correlations.values,
        ax=ax
    )

    for i, value in enumerate(correlations.values):
        offset = 0.02 if value >= 0 else -0.04
        ax.text(i, value + offset, f"{value:.2f}", ha="center")

    ax.set_title("Correlation of Top 5 Features with Diagnosis")
    ax.set_xlabel("Features")
    ax.set_ylabel("Correlation")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig

#!/opt/anaconda3/bin/python

"""
This script is used to gather small Python functions.

Usage: type "from utils import <function>" to use on of its functions.

Contributors: Ambroise Odonnat.
"""

import seaborn as sns
import pandas as pd

def get_class_distribution(data, label, display):

    """
    Get proportion of each class in the dataset data.
    
    Args:
        data (array): Training trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        label (array): Corresponding labels,
        display (bool): Display histogram of class repartition.
        
    Returns:
            count_label (dictionnary): Keys are label and values are corresponding proportion.
    """
    
    count_label = {k:0 for k in label }
    for k in label:
        count_label[k] += 1
    
    # Plot distribution
    if display:
        print("Distribution of classes: \n",count_label)
        sns.barplot(data = pd.DataFrame.from_dict([count_label]).melt(), x = "variable", y="value",\
                    hue="variable").set_title('Number of spikes distribution')
    
    return count_label
#This script contains functions to help with coding

# Load Packages=====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# EDA Function======================================================================================
def make_genres_plot(df, var1, var2):
  df1 = df[[var1, var2]]
  df1_n = df1.groupby([var1, var2], as_index=False).size()
  
  p_var = sns.catplot(x=var1, y="size", kind="bar", hue=var2, data=df1_n)
  p_var.set_axis_labels(x_var=var1, y_var="Number")
  p_var=sns.move_legend(p_var, loc="lower center", ncol=2)
  plt.show()
  plt.close()








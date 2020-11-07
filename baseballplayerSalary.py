
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

picher_file_path = './sourceData/picher_stats_2017.csv'
batter_file_path = './sourceData/batter_stats_2017.csv'

picher = pd.read_csv(picher_file_path)
batter = pd.read_csv(batter_file_path)

picher.columns

picher.head()

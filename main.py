import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import datetime
from sklearn.impute import KNNImputer, MissingIndicator

DATASET_ARG = sys.argv[1]
if (DATASET_ARG == 'a'):
    FILES_PATH = 'data/training_setA/'
    OUT_PATH = 'data/imputed_A/'
elif (DATASET_ARG == 'b'):
    FILES_PATH = 'data/training_setB/'
    OUT_PATH = 'data/imputed_B/'
assert os.path.exists(FILES_PATH)
assert os.path.exists(OUT_PATH)


def load_dataset():
    files = os.listdir(FILES_PATH)
    dataset = [pd.read_csv(os.path.join(FILES_PATH,file),sep='|') for file in files]
    print("* files loaded *")
    return dataset


def had_sepsis(df):
  """ return whether the patient developed sepsis """
  return (1 in df['SepsisLabel'].unique())


def purge_columns(df):
  #remove columns with large amounts of missing data
  to_drop = ['EtCO2', 'HCO3', 'AST', 'Alkalinephos', 'Chloride',
             'Bilirubin_direct', 'Lactate', 'Phosphate',
             'TroponinI', 'PTT', 'Fibrinogen', 'BaseExcess']
  return (df.drop(columns=to_drop))


def map_to_array(dataset):
  #create list of np arrays for all patients
  arraylist = [df.values for df in dataset]
  #create one big np array containing all rows in A
  bigarray = np.vstack((arraylist[:]))
  #create mapping from patient index to rows in all_A
  rowmap = {}
  x = 0
  for i in range(0,len(arraylist)):
      rowmap[i] = {'start': x, 'end': x+(arraylist[i].shape[0])}
      x += arraylist[i].shape[0]
  return (bigarray,rowmap)


def process_data(dataset):
  #prepare data for KNN imputation
  #remove columns with large amounts of missing data
  purged = [purge_columns(df) for df in dataset]
  datacopy = purged.copy()
  #remove demographic information
  data = [df.iloc[:,0:22] for df in datacopy]
  #change from dataframes to arrays
  arraydata = map_to_array(data)
  return datacopy,arraydata[0],arraydata[1]


def rowprop(row,limit):
  p = np.sum(np.isnan(row))/len(row)
  return (p<=limit)


def get_sample_rows(bigarray):
  #list of whether each row has less than 5% missing data or not
  rowmask = [rowprop(row,0.05) for row in bigarray]
  #list of nearly full rows
  fullrows = [row for row, full in zip(bigarray,rowmask) if full]
  return fullrows


def knn_impute(bigarray):
  #perform knn_imputation using sample rows
  print('*STARTING IMPUTING*')
  print(datetime.datetime.now())

  #impute sample rows so they are full
  samplerows = get_sample_rows(bigarray)
  stack_samplerows = np.vstack((samplerows[:]))
  imputer = KNNImputer(n_neighbors=5, weights='distance')
  complete_samples = imputer.fit_transform(stack_samplerows)

  #do knn imputation on each row using samples
  for i in range(0,len(bigarray)):
    if (i%50000==0): print(f"Imputing row - {i}")
    if (rowprop(bigarray[i],0)==False): #if not full
      #make big array of row and samples
      big = np.vstack((bigarray[i],complete_samples))
      #do knn of row + samples
      imputer = KNNImputer(n_neighbors=5, weights='distance')
      filled = imputer.fit_transform(big)
      #extract and replace current row
      newrow = filled[0]
      bigarray[i] = newrow

  print('*FINISHED IMPUTING*')
  print(datetime.datetime.now())

  return bigarray


def map_to_df(data,bigarray,rowmap):
  #turn big array back into dataframe with column names
  attributes = data[0].columns.values
  # for each patient
  for i in range(0,len(data)):
    #get corresponding rows from big array using rowmap
    new = bigarray[ rowmap[i]['start']:rowmap[i]['end'] ,:]
    newT = new.T
    #get original patient dataframe
    df = data[i]
    #for each imputed column (ie not demographics), replace old with new
    if (i%2000==0): print(i)
    for j in range(0,new.shape[1]):
      df.iloc[:,j] = newT[j]
    data[i] = df

  return data


def main():
  dataset = load_dataset()
  datacopy, bigarray, rowmap = process_data(dataset)
  imputed_data = knn_impute(bigarray)
  df_data = map_to_df(datacopy,bigarray,rowmap)
  #write to csvs
  for i, df in enumerate(df_data, 1):
    filename = "p{}".format(i)
    filepath = os.path.join(OUT_PATH, filename)
    df.to_csv(filepath)




if __name__ == "__main__":
    main()

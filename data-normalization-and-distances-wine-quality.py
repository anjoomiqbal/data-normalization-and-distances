############################################
################ LIBRARY IMPORTS  
############################################
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tabulate import tabulate
from scipy.spatial import distance
   
############################################
################ DATA INPUT  
############################################
wine_data_red   = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')
wine_data_white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',sep=';')

############################################
################ DATA PRE-PROCESSING  
############################################

## Keeping only top 10 rows
wine_data_red   = wine_data_red[:10]
wine_data_white = wine_data_white[:10]

## Converting all values to float 
wine_data_red = wine_data_red.astype(float)
wine_data_white = wine_data_white.astype(float)

print('*******************************************')
print('TASK 1 - START')
print('*******************************************')

#####################################################
################ min-max normalized values
#####################################################
min_max_scale_wine_data_red     = preprocessing.MinMaxScaler().fit(wine_data_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
min_max_scale_wine_data_white   = preprocessing.MinMaxScaler().fit(wine_data_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
min_max_wine_data_red     = min_max_scale_wine_data_red.transform(wine_data_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
min_max_wine_data_white   = min_max_scale_wine_data_white.transform(wine_data_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])

print('*******************************************')
print('min-max normalized values for RED WINE')
print('*******************************************')
print(tabulate(min_max_wine_data_red, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))
print('*******************************************')
print('Min-value after min-max scaling for RED wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(min_max_wine_data_red[:,0].min(),
              min_max_wine_data_red[:,1].min(),
              min_max_wine_data_red[:,2].min(),
              min_max_wine_data_red[:,3].min(),
              min_max_wine_data_red[:,4].min(),
              min_max_wine_data_red[:,5].min(),
              min_max_wine_data_red[:,6].min(),
              min_max_wine_data_red[:,7].min(),
              min_max_wine_data_red[:,8].min(),
              min_max_wine_data_red[:,9].min(),
              min_max_wine_data_red[:,10].min(),
              min_max_wine_data_red[:,11].min()))
print('*******************************************')
print('Max-value after min-max scaling for RED wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(min_max_wine_data_red[:,0].max(),
              min_max_wine_data_red[:,1].max(),
              min_max_wine_data_red[:,2].max(),
              min_max_wine_data_red[:,3].max(),
              min_max_wine_data_red[:,4].max(),
              min_max_wine_data_red[:,5].max(),
              min_max_wine_data_red[:,6].max(),
              min_max_wine_data_red[:,7].max(),
              min_max_wine_data_red[:,8].max(),
              min_max_wine_data_red[:,9].max(),
              min_max_wine_data_red[:,10].max(),
              min_max_wine_data_red[:,11].max()))

print('*******************************************')
print('min-max normalized values for WHITE WINE')
print('*******************************************')
print(tabulate(min_max_wine_data_white, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))
print('*******************************************')
print('Min-value after min-max scaling for WHITE wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(min_max_wine_data_white[:,0].min(),
              min_max_wine_data_white[:,1].min(),
              min_max_wine_data_white[:,2].min(),
              min_max_wine_data_white[:,3].min(),
              min_max_wine_data_white[:,4].min(),
              min_max_wine_data_white[:,5].min(),
              min_max_wine_data_white[:,6].min(),
              min_max_wine_data_white[:,7].min(),
              min_max_wine_data_white[:,8].min(),
              min_max_wine_data_white[:,9].min(),
              min_max_wine_data_white[:,10].min(),
              min_max_wine_data_white[:,11].min()))
print('*******************************************')
print('Max-value after min-max scaling for WHITE wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(min_max_wine_data_white[:,0].max(),
              min_max_wine_data_white[:,1].max(),
              min_max_wine_data_white[:,2].max(),
              min_max_wine_data_white[:,3].max(),
              min_max_wine_data_white[:,4].max(),
              min_max_wine_data_white[:,5].max(),
              min_max_wine_data_white[:,6].max(),
              min_max_wine_data_white[:,7].max(),
              min_max_wine_data_white[:,8].max(),
              min_max_wine_data_white[:,9].max(),
              min_max_wine_data_white[:,10].max(),
              min_max_wine_data_white[:,11].max()))

#####################################################
################ Z-score normalized values
#####################################################
scale_wine_data_red     = preprocessing.StandardScaler().fit(wine_data_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
scale_wine_data_white   = preprocessing.StandardScaler().fit(wine_data_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
z_wine_data_red     = scale_wine_data_red.transform(wine_data_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])
z_wine_data_white   = scale_wine_data_white.transform(wine_data_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']])

print('*******************************************')
print('Z-score normalized values for RED WINE')
print('*******************************************')
print(tabulate(z_wine_data_red, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))
print('*******************************************')
print('Mean after standardization for RED wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(z_wine_data_red[:,0].mean(),
              z_wine_data_red[:,1].mean(),
              z_wine_data_red[:,2].mean(),
              z_wine_data_red[:,3].mean(),
              z_wine_data_red[:,4].mean(),
              z_wine_data_red[:,5].mean(),
              z_wine_data_red[:,6].mean(),
              z_wine_data_red[:,7].mean(),
              z_wine_data_red[:,8].mean(),
              z_wine_data_red[:,9].mean(),
              z_wine_data_red[:,10].mean(),
              z_wine_data_red[:,11].mean()))
print('*******************************************')
print('Standard deviation after standardization for RED wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(z_wine_data_red[:,0].std(),
              z_wine_data_red[:,1].std(),
              z_wine_data_red[:,2].std(),
              z_wine_data_red[:,3].std(),
              z_wine_data_red[:,4].std(),
              z_wine_data_red[:,5].std(),
              z_wine_data_red[:,6].std(),
              z_wine_data_red[:,7].std(),
              z_wine_data_red[:,8].std(),
              z_wine_data_red[:,9].std(),
              z_wine_data_red[:,10].std(),
              z_wine_data_red[:,11].std()))
print('*******************************************')
print('Z-score normalized values for RED WINE')
print('*******************************************')
print(tabulate(z_wine_data_white, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))
print('*******************************************')
print('Mean after standardization for WHITE wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(z_wine_data_white[:,0].mean(),
              z_wine_data_white[:,1].mean(),
              z_wine_data_white[:,2].mean(),
              z_wine_data_white[:,3].mean(),
              z_wine_data_white[:,4].mean(),
              z_wine_data_white[:,5].mean(),
              z_wine_data_white[:,6].mean(),
              z_wine_data_white[:,7].mean(),
              z_wine_data_white[:,8].mean(),
              z_wine_data_white[:,9].mean(),
              z_wine_data_white[:,10].mean(),
              z_wine_data_white[:,11].mean()))
print('*******************************************')
print('Standard deviation after standardization for WHITE wine data: \nfixed acidity={:.4f} \nvolatile acidity={:.4f} \ncitric acid={:.4f} \nesidual sugar={:.4f}, \nchlorides={:.4f} \nfree sulfur dioxide={:.4f} \ntotal sulfur dioxide={:.4f} \ndensity={:.4f} \npH={:.4f} \nsulphates={:.4f} \nalcohol={:.4f} \nquality={:.4f}, \n'
      .format(z_wine_data_white[:,0].std(),
              z_wine_data_white[:,1].std(),
              z_wine_data_white[:,2].std(),
              z_wine_data_white[:,3].std(),
              z_wine_data_white[:,4].std(),
              z_wine_data_white[:,5].std(),
              z_wine_data_white[:,6].std(),
              z_wine_data_white[:,7].std(),
              z_wine_data_white[:,8].std(),
              z_wine_data_white[:,9].std(),
              z_wine_data_white[:,10].std(),
              z_wine_data_white[:,11].std()))

#####################################################
################ mean subtracted normalized (msn) values
#####################################################
# Ref: https://docs.tibco.com/pub/spotfire/7.0.1/doc/html/norm/norm_subtract_the_mean.htm
# ref: https://stackoverflow.com/questions/35169368/subtract-every-column-in-dataframe-with-the-mean-of-the-that-column-python

msn_wine_data_red   = wine_data_red   - wine_data_red.mean()
msn_wine_data_white = wine_data_white - wine_data_white.mean()

print('*******************************************')
print('mean subtracted normalized values for RED WINE')
print('*******************************************')
print(tabulate(msn_wine_data_red, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))

print('*******************************************')
print('mean subtracted normalized values for WHITE WINE')
print('*******************************************')
print(tabulate(msn_wine_data_white, headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"], tablefmt='fancy_grid'))

print('*******************************************')
print('TASK 1 - END')
print('*******************************************')



print('*******************************************')
print('*******************************************')



print('*******************************************')
print('TASK 2 - START')
print('*******************************************')
# to compute distance need to convert the dataframe to an array
wine_data_red   = np.array(wine_data_red)
wine_data_white = np.array(wine_data_white)

#####################################################
################ manhatten distance
#####################################################
print('*******************************************')
print('manhatten distance')
print('*******************************************')
manhatten_dist_matrix = distance.cdist(wine_data_red, wine_data_red, metric='cityblock')
print(tabulate(manhatten_dist_matrix, tablefmt='fancy_grid'))

manhatten_dist_output = np.zeros([10,2])

# max values or the farthest values
manhatten_dist_output[:,1] = np.amax(manhatten_dist_matrix,axis=1)

# as the distances are 0s along the diagonal of the matrix
# after capturing the max values and before finding  the min values
# we are assigning inf to the diagonal values
for i in range (0, len(manhatten_dist_matrix[0,:])):
    for j in range (0, len(manhatten_dist_matrix[0,:])):
        if i==j:
            manhatten_dist_matrix[i,j] = np.inf
#print(tabulate(manhatten_dist_matrix, tablefmt='fancy_grid'))

# min values or the nearest values
manhatten_dist_output[:,0] = np.amin(manhatten_dist_matrix,axis=1)
print(tabulate(manhatten_dist_output,headers=["nearest", "farthest"], tablefmt='fancy_grid'))


#####################################################
################ euclidean distance
#####################################################
print('*******************************************')
print('euclidean distance')
print('*******************************************')
euclidean_dist_matrix = distance.cdist(wine_data_red, wine_data_red, metric='euclidean')
print(tabulate(euclidean_dist_matrix, tablefmt='fancy_grid'))

euclidean_dist_output = np.zeros([10,2])

# max values or the farthest values
euclidean_dist_output[:,1] = np.amax(euclidean_dist_matrix,axis=1)

# as the distances are 0s along the diagonal of the matrix
# after capturing the max values and before finding  the min values
# we are assigning inf to the diagonal values
for i in range (0, len(euclidean_dist_matrix[0,:])):
    for j in range (0, len(euclidean_dist_matrix[0,:])):
        if i==j:
            euclidean_dist_matrix[i,j] = np.inf
#print(tabulate(euclidean_dist_matrix, tablefmt='fancy_grid'))

# min values or the nearest values
euclidean_dist_output[:,0] = np.amin(euclidean_dist_matrix,axis=1)

print(tabulate(euclidean_dist_output,headers=["nearest", "farthest"], tablefmt='fancy_grid'))


#####################################################
################ cosine distance
#####################################################
print('*******************************************')
print('cosine distance')
print('*******************************************')
cosine_dist_matrix = distance.cdist(wine_data_red, wine_data_red, metric='cosine')
print(tabulate(cosine_dist_matrix, tablefmt='fancy_grid'))

cosine_dist_output = np.zeros([10,2])

# max values or the farthest values
cosine_dist_output[:,1] = np.amax(cosine_dist_matrix,axis=1)

# as the distances are 0s along the diagonal of the matrix
# after capturing the max values and before finding  the min values
# we are assigning inf to the diagonal values
for i in range (0, len(cosine_dist_matrix[0,:])):
    for j in range (0, len(cosine_dist_matrix[0,:])):
        if i==j:
            cosine_dist_matrix[i,j] = np.inf
#print(tabulate(cosine_dist_matrix, tablefmt='fancy_grid'))

# min values or the nearest values
cosine_dist_output[:,0] = np.amin(cosine_dist_matrix,axis=1)
print(tabulate(cosine_dist_output,headers=["nearest", "farthest"], tablefmt='fancy_grid'))




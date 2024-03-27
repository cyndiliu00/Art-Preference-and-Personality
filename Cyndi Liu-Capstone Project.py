# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:28:41 2023

@author: Cyndi
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
N_num = 11091634
np.random.seed(N_num)

art_data = pd.read_csv("theArt.csv")
user_data = np.genfromtxt('theData.csv', delimiter=',')
art_dataArray = np.array(art_data)

preference_ratings = user_data[:,:91]

#%% Cleansing Data for preference_ratings (Dealing with NaN)
# Element-wise removal for nans

new_preference_ratings = []
for i in range(90):
    M=preference_ratings[:,i]
    M=M[np.isfinite(M)]
    new_preference_ratings.append(M)
# I found that there is no nan in each column, so use the original "preference_ratings" data

#%% 1) Is classical art more well liked than modern art?
classical_art_ratings = preference_ratings[:,0:35].flatten()
modern_art_ratings = preference_ratings[:,35:70].flatten()

# Do EDA
a=plt.hist(classical_art_ratings,bins=7) 
b=plt.hist(modern_art_ratings,bins=7)
plt.show()

# From EDA, it can be seen that both modern_art_ratings and classical_art_ratings are not normally distributed
# Therefore, I choose to use non-parametric test (U test)

u1,p1 = stats.mannwhitneyu(classical_art_ratings,modern_art_ratings,alternative='greater')

# p=1.5881633286154516e-97, which is <0.05, so null hypothesis is rejected. Classical art is more well liked
# than modern art.

# Plot
combinedData=[classical_art_ratings, modern_art_ratings]
plt.boxplot(combinedData, labels=['Classical Art', 'Modern Art'])
plt.ylabel("Peference Ratings")
plt.title('Classical Art Ratings vs. Modern Art Ratings')
plt.show()

#%% 2) Is there a difference in the preference ratings for modern art vs. non-human (animals and computers)
#      generated art?

nonHuman_art_ratings = preference_ratings[:,70:].flatten()

# Do EDA
plt.hist(nonHuman_art_ratings, bins=7)
plt.show()

# From EDA, it can be seen that both non_human_art_ratings is also not normally distributed
# Therefore, I choose to use non-parametric test (U test)

u2,p2 = stats.mannwhitneyu(classical_art_ratings, nonHuman_art_ratings)
# p=0.0, which is <0.05, so null hypothesis is rejected. There is difference in the preference
# ratings for modern art vs. non-human generated art.

# plot
combinedData=[classical_art_ratings, nonHuman_art_ratings]
plt.boxplot(combinedData, labels=['Classical Art', 'Non-Human Art'])
plt.ylabel("Peference Ratings")
plt.title('Classical Art Ratings vs. Non-Human Art Ratings')
plt.show()


#%% 3) Do women give higher art preference ratings than men?

user_gender = user_data[:,216]

male_ratings = preference_ratings[user_gender==1].flatten()
female_ratings = preference_ratings[user_gender==2].flatten()

# Do EDA
plt.hist(male_ratings,bins=7)
plt.show()
plt.hist(female_ratings,bins=7)
plt.show()

# From EDA, it can be seen that both female_ratings and male_ratings are not normally distributed
# Therefore, I choose to use non-parametric test (U test)

u3,p3 = stats.mannwhitneyu(male_ratings, female_ratings, alternative='greater')

# p=0.8643548652654933, which is >0.05, so null hypothesis is not rejected. Women do give higher art preference 
# than men.

# Plot
combinedData=[male_ratings, female_ratings]
plt.boxplot(combinedData, labels=['Male', 'Female'])
plt.ylabel("Peference Ratings")
plt.title('Male Ratings vs. Female Ratings')
plt.show()


#%% 4) Is there a difference in the preference ratings of users with some art background 
#      (some art education) vs. none?

user_art_background = user_data[:,218]

noneBackground_ratings = preference_ratings[user_art_background==0].flatten()
someBackground_ratings = preference_ratings[user_art_background>0].flatten()

# Do EDA
#plt.hist(noneBackground_ratings,bins=7)
plt.hist(someBackground_ratings,bins=7)
#plot.show()

# From EDA, it can be seen that both noneBackground_ratings and someBackground_ratings are not normally distributed
# Therefore, I choose to use non-parametric test (U test)

u4,p4 = stats.mannwhitneyu(noneBackground_ratings, someBackground_ratings)

# p=3.0567413101500694e-09, which is <0.05, so null hypothesis is rejected. There is a difference in
# preference ratings of users with some art background vs. none.

# Plot
combinedData=[noneBackground_ratings, someBackground_ratings]
plt.boxplot(combinedData, labels=['None Art Background', 'Some Art Background'])
plt.ylabel("Peference Ratings")
plt.title("None Art Background Ratings vs. Some Art Background Ratings")
plt.show()


#%% 5) Build a regression model to predict art preference ratings from energy ratings only. 
#      Make sure to use cross-validation methods to avoid overfitting and characterize how 
#      well your model predicts art preference ratings.

# As the art preference ratings is predicted from energy ratings, which has only one predictor,
# I choose to build linear regression model.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

energy_ratings = user_data[:,91:182]
energy_mean = np.mean(energy_ratings,axis=1)
preference_mean = np.mean(preference_ratings, axis=1)
x = energy_mean.reshape(-1,1)
y = preference_mean.reshape(-1,1)

# Cross-validation
xTrain, xTest , yTrain, yTest = train_test_split(x, y, test_size=0.5, random_state=N_num)

model5 = LinearRegression().fit(xTrain, yTrain)
y_pred5 = model5.predict(xTest)
beta1 = model5.coef_
yInt1 = model5.intercept_
rmse = np.sqrt(np.mean((y_pred5 - yTest)**2))
rSqr = model5.score(xTest, yTest)

# Plot
yHat1 = beta1*xTrain+yInt1
plt.plot(xTrain,yTrain,'o',markersize=3)
plt.xlabel('Energy Ratings')
plt.ylabel('Art Preference Ratings')
plt.title('RMSE = {:.3f}'.format(rmse))
plt.show()




#%% 6) Build a regression model to predict art preference ratings from energy ratings and 
# demographic information. Make sure to use cross-validation methods to avoid overfitting 
# and comment on how well your model predicts relative to the “energy ratings only” model.

demographic = user_data[:,215:221]

# In order to build a regression model, I need to keep n the same so I use row-wise missing 
# data removal

#demographic = demographic[~np.isnan(demographic).any(axis=1)]  

temp1=demographic[~np.isnan(demographic).any(axis=1)]
temp2=energy_mean[~np.isnan(demographic).any(axis=1)]
X = np.column_stack((temp1,temp2))
temp3=preference_ratings[~np.isnan(demographic).any(axis=1)]
y=np.mean(temp, axis=1) # Take the mean of each user's rating for 91 art pieces
# After taking the means of preference rating, the range of preference ratings becomes
# narrow, which is the restricted range problem. It is not good for modeling 
# because these data are similar. Therefore, I scale the preference means up by.

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
temp2 = scaler.fit(ratingsMean)
y = scaler.transform(ratingsMean)
'''
# Cross-validation
XTrain, XTest , yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=N_num)

# As it would be multi-collinearity model, I will compute all linear, ridge, and lasso regression.
# Then see which one performs the best by assessment.

# Multiple Regression
model6 = LinearRegression().fit(XTrain, yTrain)
y_pred6 = model6.predict(XTest)
b1 = model6.coef_
yInt1 = model6.intercept_
rmse = np.sqrt(np.mean((y_pred6 - yTest)**2))

# Plot
yHat = b1[0]*XTrain[:,0] + b1[1]*XTrain[:,1] + b1[2]*XTrain[:,2] + b1[3]*XTrain[:,3] + b1[4]*XTrain[:,4] + b1[5]*XTrain[:,5] + yInt1
plt.plot(yHat,yTrain,'o',markersize=4)
plt.xlabel('Prediction Scaled Rating From Model')
plt.ylabel('Actual Scaled Rating')
plt.title('RMSE = {:.3f}'.format(rmse))

# Ridge Regression
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error 

ridge = Ridge(normalize = True)
alph = 0.1 
ridge.set_params(alpha = alph)
ridge.fit(XTrain, yTrain) #Fit the model
newBetas = ridge.coef_ #Compare to betas above
rSqr = ridge.score(XTrain, yTrain)

# Lasso Regression
from sklearn.linear_model import Lasso #This is to do the LASSO
from sklearn.preprocessing import scale #This is to fit it

numIt = 10000 #How many iterations - Lasso is an iterative algorithm

lasso = Lasso(max_iter = numIt, normalize = True) #Create LASSO model
lasso.set_params(alpha=alph)
lasso.fit(scale(XTrain), yTrain)
b1 = lasso.coef_
mse=mean_squared_error(yTest, lasso.predict(XTest))

#%% 7) Considering the 2D space of average preference ratings vs. average energy rating 
#     (that contains the 91 art pieces as elements), how many clusters can you – algorithmically 
#     - identify in this space? Make sure to comment on the identity of the clusters – do they 
#     correspond to particular types of art?

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

preference_avg = np.mean(preference_ratings, axis=0)
energy_avg = np.mean(energy_ratings, axis=0)

x = np.column_stack((preference_avg,energy_avg))

# How many clusters k to ask for? Do Silhouette: 
numClusters = 9
sSum = np.empty([numClusters,1])*np.NaN

# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 
    
# Plot the sum of the silhouette scores as a function of the number of clusters
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.title('Identifying the number of KMeans Clusters')
plt.show()

# It can be seen that when the number of cluster is 4, the sum of silhouette score has the highest
# value. Therefore, I can identify 4 clusters in this space of average preference ratings vs. 
# average energy ratings.

# kMeans:
numClusters = 4
kMeans = KMeans(n_clusters = numClusters).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=3)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Preference Rating')
    plt.ylabel('Energy Rating')
    
# Identifying the clusters with art type
classical_energy = energy_ratings[:,0:35].flatten()
modern_energy = energy_ratings[:,35:70].flatten()
nonHuman_energy = energy_ratings[:,70:].flatten()
classical_id = [np.mean(classical_art_ratings), np.mean(classical_energy)]
modern_id = [np.mean(modern_art_ratings), np.mean(modern_energy)]
nonHuman_id = [np.mean(nonHuman_art_ratings), np.mean(nonHuman_energy)]
print("classical:", classical_id)
print("modern:", modern_id)
print("non-human:", nonHuman_id)
#%% 8) Considering only the first principal component of the self-image ratings as inputs to 
#   a regression model – how well can you predict art preference ratings from that factor alone?

predictors = user_data[:,205:215]
outcomes = preference_ratings[~np.isnan(predictors).any(axis=1)]
predictors = predictors[~np.isnan(predictors).any(axis=1)]

'''
# To ascertain whether a PCA is indicated, do a correlation heatmap
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.title('Self-Image Correlation Heatmap')
plt.show()
# It suggests that the variables are correlated. PCA is indicated.
'''
# Do PCA
from sklearn.decomposition import PCA

zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors8 = pca.fit_transform(zscoredData)*-1

# Scree plot:
numPredictors = 10
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.axhline(y=1, color = 'r', linestyle = '--')
plt.show()

# Look at the loadings to find out the meaning of the first factor
plt.subplot(1,2,1) # 1st Factor 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:])
plt.ylabel('Effect Component')
plt.xlabel('Self-image Metric')
plt.title('1st Factor')
plt.show()
# 2nd Factor
plt.subplot(1,2,2) # Factor 2:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:])
plt.ylabel('Effect Component')
plt.xlabel('Self-image Metric')
plt.title('2nd Factor')
plt.show()

# Build the regression model to predict art preference ratings from the first principal component
x = predictors8[:,0].reshape(-1,1) 
y = np.mean(outcomes, axis=1).reshape(-1,1)
#y = stats.zscore(outcomes)

# Cross-validation
xTrain, xTest , yTrain, yTest = train_test_split(x, y, test_size=0.7, random_state=N_num)

model8 = LinearRegression().fit(xTrain, yTrain)
y_pred8 = model8.predict(xTest)
beta1 = model8.coef_
yInt1 = model8.intercept_
rmse = np.sqrt(np.mean((y_pred8 - yTest)**2))

# Plot
yHat1 = beta1 * xTrain + yInt1
plt.plot(xTrain,yTrain,'o',markersize=3)
plt.xlabel('First Principal Component of Self-Image')
plt.ylabel('Art Preference Ratings')
plt.title('RMSE = {:.3f}'.format(rmse))
plt.show()



#%% 9) Consider the first 3 principal components of the “dark personality” traits – use these as inputs 
#   to a regression model to predict art preference ratings. Which of these components significantly 
#   predict art preference ratings? Comment on the likely identity of these factors (e.g. narcissism, 
#   manipulativeness, callousness, etc.).

predictors = user_data[:,182:194]
outcomes = preference_ratings[~np.isnan(predictors).any(axis=1)]
predictors = predictors[~np.isnan(predictors).any(axis=1)]

# Do PCA
zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors9 = pca.fit_transform(zscoredData)*-1

# Scree plot:
numPredictors = 12
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.axhline(y=1, color = 'r', linestyle = '--')
plt.show()

# Look at the loadings to find out the meaning of the first factor
plt.subplot(1,3,1) # 1st Factor 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:])
plt.ylabel('Effect Component')
plt.xlabel('Dark Personality Metric')
plt.title('1st Factor')
# 2nd Factor
plt.subplot(1,3,2) # Factor 2:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:])
plt.title('2nd Factor')
#3rd Factor
plt.subplot(1,3,3) # Factor 3:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[2,:])
plt.title('3rd Factor')
plt.show()

# Build Linear Regression Model
X = predictors9[:,0:3]
y = np.mean(outcomes, axis=1) # Take the mean of each user's rating for 91 art pieces

# Cross-Validation
XTrain, XTest , yTrain, yTest = train_test_split(X, y, test_size=0.5, random_state=N_num)

model9 = LinearRegression().fit(XTrain, yTrain)
y_pred9 = model9.predict(XTest)
b0,b1 = model9.intercept_, model9.coef_
rmse = np.sqrt(np.mean((y_pred9 - yTest)**2))

yHat = b1[0]*XTrain[:,0] + b1[1]*XTrain[:,1] + b1[2]*XTrain[:,2] + b0
plt.plot(yHat,yTrain,'o',markersize=4)
plt.xlabel('Prediction Scaled Rating From Model')
plt.ylabel('Actual Scaled Rating')
plt.title('RMSE = {:.3f}'.format(rmse))
plt.show()

# Lasso Regression
from sklearn.linear_model import Lasso 
from sklearn.preprocessing import scale 

numIt = 10000 

lasso = Lasso(max_iter = numIt, normalize = True) #Create LASSO model
lasso.set_params(alpha=0.1)
lasso.fit(XTrain, yTrain)
b1 = lasso.coef_
mse=mean_squared_error(yTest, lasso.predict(XTest))

#%% Extra: PCA for action preference
predictors = user_data[:,194:205]
outcomes = preference_ratings[~np.isnan(predictors).any(axis=1)]
predictors = predictors[~np.isnan(predictors).any(axis=1)]

# Do PCA
zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors10 = pca.fit_transform(zscoredData)*-1

# Scree plot:
numPredictors = 11
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.axhline(y=1, color = 'r', linestyle = '--')
plt.show()

# Look at the loadings to find out the meaning of the first factor
plt.subplot(1,3,1) # 1st Factor 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:])
plt.title('1st Factor')
# 2nd Factor
plt.subplot(1,3,2) # Factor 2:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:])
plt.title('2nd Factor')
#3rd Factor
plt.subplot(1,3,3) # Factor 3:
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[2,:])
plt.title('3rd Factor')

#%% 10) Can you determine the political orientation of the users (to simplify things and avoid gross 
#   class imbalance issues, you can consider just 2 classes: 
#   “left” (progressive & liberal) vs. “non-left” (everyone else)) from all the other information 
#   available, using any classification model of your choice? Make sure to comment on the classification 
#   quality of this model.

# To determine/predict the political orientation of the users, I decide to build models that predict the
# political orientation from self-image, dark personality traits, and action preferences respectively.
# Therefore, I will build 3 random forests models to determin the political orientation from these predictors.

# Import Data
user_political = user_data[:,217]
selfImage = user_data[:,205:215]
actionPreferences = user_data[:,194:205]
darkTraits = user_data[:,182:194]

temp = np.column_stack((selfImage,actionPreferences,darkTraits))  # columns 0-9: selfImage; columns 10-20: actionPreference; columns 21-32: darkTraits
combinedData = temp[~np.isnan(temp).any(axis=1)]
temp2=user_political[~np.isnan(temp).any(axis=1)]
X_data = combinedData[~np.isnan(temp2)]
temp3 = temp2[~np.isnan(temp2)]
new_user_political = np.where(temp3 <= 2, 1, 0)

#%% a) Political Orientation vs. Self-Image

predictors = X_data[:,0:10]
yOutcomes = new_user_political

# Do PCA
zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors_a = pca.fit_transform(zscoredData)*-1

# Classifying political orientation with random forests 

# First: Self image
# Mixing in the labels Y:
X = np.column_stack((predictors_a[:,0],predictors_a[:,1]))
plt.plot(X[np.argwhere(yOutcomes==0),0],X[np.argwhere(yOutcomes==0),1],'o',markersize=2,color='green')
plt.plot(X[np.argwhere(yOutcomes==1),0],X[np.argwhere(yOutcomes==1),1],'o',markersize=2,color='blue')
plt.xlabel('1st factor')
plt.ylabel('2nd factor')
plt.legend(['Non Left','Left'])
plt.show()
# Note: From the graph, it is hard to see a trend, which means the outcomes are not 
# determined by these factor. The variability may be large. This means that there won't be a linear 
# solution. The dataset is not linearly separable. There is no line we could draw to separate the 
# dataset cleanly into left political orientation and non-left orientation.

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Split the dataset (80% Train, 20% Test)
XTrain, XTest , yTrain, yTest = train_test_split(X, yOutcomes, test_size=0.2, random_state=N_num)

numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(XTrain,yTrain) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(XTest) 

# Assess model accuracy:
modelAccuracy = accuracy_score(yTest,predictions)
print('Random forest model accuracy:',modelAccuracy)
confusion = confusion_matrix(yTest,predictions)
plot_confusion_matrix(clf, XTest, yTest, cmap=plt.cm.Blues)
plt.show()

# The random forest model accuracy is 0.5

#%% b) Political Orientation vs. Action Preferences

predictors = X_data[:,10:21]
yOutcomes = new_user_political

# Do PCA
zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors_b = pca.fit_transform(zscoredData)*-1

# Classifying political orientation with random forests 

# First: Self image
# Mixing in the labels Y:
X = np.column_stack((predictors_b[:,0],predictors_b[:,1],predictors_b[:,2]))
plt.plot(X[np.argwhere(yOutcomes==0),0],X[np.argwhere(yOutcomes==0),1],'o',markersize=2,color='green')
plt.plot(X[np.argwhere(yOutcomes==1),0],X[np.argwhere(yOutcomes==1),1],'o',markersize=2,color='blue')
plt.xlabel('1st factor')
plt.ylabel('2nd factor')
plt.legend(['Non Left','Left'])
plt.show()
# Note: From the graph, it is hard to see a trend, which means the outcomes are not 
# determined by these factor. The variability may be large. This means that there won't be a linear 
# solution. The dataset is not linearly separable. There is no line we could draw to separate the 
# dataset cleanly into left political orientation and non-left orientation.

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#Split the dataset (80% Train, 20% Test)
XTrain, XTest , yTrain, yTest = train_test_split(X, yOutcomes, test_size=0.2, random_state=N_num)

numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(XTrain,yTrain) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(XTest) 

# Assess model accuracy:
modelAccuracy = accuracy_score(yTest,predictions)
print('Random forest model accuracy:',modelAccuracy)
# The random forest model accuracy: 0.6964285714285714
confusion = confusion_matrix(yTest,predictions)
plot_confusion_matrix(clf, XTest, yTest, cmap=plt.cm.Blues)
plt.show()

#%% c) Political Orientation vs. Dark Personality Traits

predictors = X_data[:,21:33]
yOutcomes = new_user_political

# Do PCA
zscoredData = stats.zscore(predictors)   # Z-score the predictors data
pca = PCA().fit(zscoredData)   # Run PCA
eigVals = pca.explained_variance_   #Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
loadings = pca.components_*-1     # Loadings (eigenvectors): Weights per factor in terms of the original data.
predictors_c = pca.fit_transform(zscoredData)*-1

# Classifying political orientation with random forests 

# First: Self image
# Mixing in the labels Y:
X = np.column_stack((predictors_c[:,0],predictors_c[:,1],predictors_c[:,2]))
plt.plot(X[np.argwhere(yOutcomes==0),0],X[np.argwhere(yOutcomes==0),1],'o',markersize=2,color='green')
plt.plot(X[np.argwhere(yOutcomes==1),0],X[np.argwhere(yOutcomes==1),1],'o',markersize=2,color='blue')
plt.xlabel('1st factor')
plt.ylabel('2nd factor')
plt.legend(['Non Left','Left'])
plt.show()
# Note: From the graph, it is hard to see a trend, which means the outcomes are not 
# determined by these factor. The variability may be large. This means that there won't be a linear 
# solution. The dataset is not linearly separable. There is no line we could draw to separate the 
# dataset cleanly into left political orientation and non-left orientation.

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Split the dataset (80% Train, 20% Test)
XTrain, XTest , yTrain, yTest = train_test_split(X, yOutcomes, test_size=0.2, random_state=N_num)

numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(XTrain,yTrain) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(XTest) 

# Assess model accuracy:
modelAccuracy = accuracy_score(yTest,predictions)
print('Random forest model accuracy:',modelAccuracy)
# The random forest model accuracy: 0.44642857142857145
confusion = confusion_matrix(yTest,predictions)
plot_confusion_matrix(clf, XTest, yTest, cmap=plt.cm.Blues)
plt.show()











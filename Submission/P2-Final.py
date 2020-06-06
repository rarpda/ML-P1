# Code for machine learning practical 2.

# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import json
import time

# Test size for split
TEST_SIZE = 0.3

###
# Binary classification task
###

# Load data and assign headers
sensorData = pd.read_csv("binary/X.csv", header=None, names=range(0, 768))
classifiedData = pd.read_csv("binary/y.csv", header=None, names="Y")
dataToClassify = pd.read_csv("binary/XToClassify.csv", header=None, names=range(0, 768))

with open("binary/key.txt") as file:
    headers = json.load(file)
# Check for null values. Implement cleaning data if there are any.
print(sensorData.isnull().values.any())
print(classifiedData.isnull().values.any())
print(dataToClassify.isnull().values.any())

# Split the data and ignore test data for the rest of the assignment until testing
train_input, test_input, train_output, test_output = train_test_split(sensorData, classifiedData, test_size=TEST_SIZE,
                                                                      random_state=111, stratify=classifiedData)


### Plotting and analyzing data

# function to plot by sensor
def getSensorDataByClassification(inputData, classificationData, classificationLabel):
    # Find all
    index = classificationData.index[classificationData['Y'] == classificationLabel]
    dataset = inputData.loc[index]
    return dataset


# Create figure and add title
fig = plt.figure(figsize=(20, 10))


def plotAllTheData(training_input, output_data, classficationLabel, colour, label):
    # Get all data for a particular label
    data = getSensorDataByClassification(training_input, output_data, classficationLabel)
    for row in data.values:
        # Plot means, min and max.
        plt.scatter(y=row, x=range(0, 768), c=colour, marker='.', linewidths=0.1, label=label)
    plt.title("All the data provided", fontsize=18)
    plt.xlabel("Feature Number", fontsize=18)
    plt.ylabel("Amplitude", fontsize=18)
    plt.grid(True)


# Colours for each classification label
colours = ['blue', 'red']
array_of_patches = []
# Iterate through all labels
for classIndex in range(0, len(headers)):
    # Plot all data
    color = colours[classIndex]
    label = headers[str(classIndex)]
    plotAllTheData(train_input, train_output, classIndex, color, label)
    # Add patches to the holding array.
    array_of_patches.append(mpatches.Patch(color=color, label=label))

# Add legends and graph information
plt.legend(handles=array_of_patches)
plt.show()


# plot mean average radar signal(sum all the components) and each separate channel together
def plotAveragesData(input_data, output_data, headers, colours):
    fig = plt.figure(figsize=(25, 15))
    # Iterate through each class
    for classID in range(0, len(headers)):
        # Get data for each classification
        classData = getSensorDataByClassification(input_data, output_data, classID)
        # Average across the feature
        classData = classData.mean(axis=0)
        colour = colours[classID]
        # Iterate through each channel
        for channelIndex in range(0, 4):
            plt.subplot(2, 2, channelIndex + 1)
            # Get mean values - first 256 features.
            meanAverageValues = classData[0:256]
            # Plot for each channel
            meanAverageValues = meanAverageValues[channelIndex * 64:(channelIndex + 1) * 64]
            label = headers[str(classID)] + " mean"
            plt.plot(range(0, len(meanAverageValues)), meanAverageValues, label=label, c=colour)

            # Get min values - next 256 features.
            minAverageValues = classData[256:512]
            # Plot for each channel
            minAverageValues = minAverageValues[channelIndex * 64:(channelIndex + 1) * 64]
            label = headers[str(classID)] + " min"
            plt.plot(range(0, len(minAverageValues)), minAverageValues, label=label, c=colour, linestyle='dashed')

            # Get max values - next 256 components
            maxAverageValues = classData[512:768]
            # Plot for each channel
            maxAverageValues = maxAverageValues[channelIndex * 64:(channelIndex + 1) * 64]
            label = headers[str(classID)] + " max"
            plt.plot(range(0, len(maxAverageValues)), maxAverageValues, label=label, c=colour, linestyle='dotted')
            # Graph information
            plt.ylabel("Amplitude", fontsize=18)
            plt.title("Data for Channel " + str(channelIndex + 1), fontsize=18)
            plt.xlabel("Component Number", fontsize=18)
            plt.grid(True)
            plt.legend()
    plt.tight_layout()
    plt.show()


# Plot all averaged data.
plotAveragesData(train_input, train_output, headers, colours)


# get correlation of mean to output
def plotMeanCorrelationLinegraph(inputData, outputData):
    plt.figure(figsize=(25, 15))
    # Get means features
    mean_values = inputData.iloc[:, 0:256]
    # Join output dataframe
    holder_df = mean_values.join(outputData)
    # Generate the correlation matrix
    corrMat = holder_df[holder_df.columns[1:]].corr()['Y'][:]
    # Plot correalation matrix
    plt.plot(range(0, len(corrMat)), corrMat)
    # Graph information
    plt.ylabel("Correlation Value", fontsize=18)
    plt.title("Pearson correlation between components and labels", fontsize=18)
    plt.xlabel("Feature Number", fontsize=18)
    plt.grid(True)
    plt.show()


# Plots the correlation linegraph for the mean features
plotMeanCorrelationLinegraph(train_input, train_output)

# Create correlation heat map with Pearson Rank Correlation Coefficients for means of channel 1
# Mean feature for channel 1
corrData = train_input.iloc[:, 0:64]
# Calculate correlation matrices
correlationMap = corrData.corr(method='pearson')
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1)
# Show correlation matrix and color bar.
im = ax.imshow(correlationMap, interpolation='nearest', cmap='Spectral')
ax.figure.colorbar(im, ax=ax)
# Set axis information
ax.set(xticks=np.arange(correlationMap.shape[1]),
       yticks=np.arange(correlationMap.shape[0]),
       xticklabels=range(1, 64), yticklabels=range(1, 64),
       title="Mean- Correlation matrix for Channel 1 with Pearson coefficient",
       ylabel='Component number',
       xlabel='Component number')
plt.show()


### Principal Component Analysis

# PCA 2 component analysis.

# Function calculate PCA
def producePCAData(inputData, classificationData):
    # Create scalar to standardise data and fit data to using it.
    standardisedSensorData = StandardScaler().fit_transform(inputData)
    # set pca to use 2 components
    pca = PCA(n_components=2)
    # Conduct analysis.
    principalComponents = pca.fit_transform(standardisedSensorData)
    # Get the principal dataframe with indices
    principalDf = pd.DataFrame(data=principalComponents, index=classificationData.index
                               , columns=['PCA1', 'PCA2'])
    # Getting the explained variance ratio for analysis.
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    return principalDf


# Method to conduct PCA analysis and plot the data
def plotPCAData(principalDf, classificationData, headers, colours):
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # populate with data. Iterate through each classification
    for classIndex in range(0, len(headers)):
        pcaData = getSensorDataByClassification(principalDf, classificationData, classIndex)
        # Plot the data
        plt.scatter(x=pcaData['PCA1'], y=pcaData['PCA2'], linewidths=1, c=colours[classIndex])
    # Setting axes information
    plt.xlabel("Principal Component 1", fontsize=18)
    plt.ylabel("Principal Component 2", fontsize=18)
    plt.title("PCA Analysis with 2 components for sensor data", fontsize=18)
    ax.legend(headers.values())
    plt.grid(True)
    plt.show()


# Plot PCA data
pcaData = producePCAData(train_input, train_output)
plotPCAData(pcaData, train_output, headers, colours)


### Training models

# Train binary SVM model.
def trainSVMBinaryModel(trainingInput, trainingOutput):
    # Create scaler and fit it to training input data
    scaler = StandardScaler().fit(trainingInput)
    # Create a pipeline for Polynomial regression. Add scaler to scale input data.
    pipeline = Pipeline([("scaler", scaler), ("model", svm.SVC(kernel='linear'))])
    # Create regularizatiosn hyperparameter space
    C = [0.01, 0.1, 1, 10]
    # Create hyperparameter options
    hyperparameters = {'model__C': C}
    # Time training time
    start = time.time()
    # 10-fold Cross validation
    model = GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=10, refit=True, n_jobs=1, scoring='roc_auc',
                         return_train_score=True)
    # fit model to training data
    model.fit(trainingInput.values, trainingOutput.values.ravel())
    end = time.time()
    print("Training Time: " + str(end - start))
    print(model.best_params_)
    # Print the best parameters
    print("Best score: %0.3f" % model.best_score_)
    return model


# Train binary Logistic Classification model.
def trainLogisticBinaryModel(trainingInput, trainingOutput):
    # Create scaler and fit it to training input data
    scaler = StandardScaler().fit(trainingInput)
    # Create a pipeline for Polynomial regression. Add scaler to scale input data.
    pipeline = Pipeline([("scaler", scaler), ("model", LogisticRegression())])
    # Create hyperparameter options
    penalty = ['l1'];
    C = [0.01, 0.1, 1]
    hyperparameters = {"model__penalty": penalty, "model__C": C}
    # Time training time
    start = time.time()
    # 10-fold Cross validation
    model = GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=10, refit=True, n_jobs=1, scoring='roc_auc',
                         return_train_score=True)
    # fit model to training data
    model.fit(trainingInput.values, trainingOutput.values.ravel())
    end = time.time()
    print("Training Time: " + str(end - start))
    # Print the best parameters
    print(model.best_params_)
    print("Best score: %0.3f" % model.best_score_)
    return model


# Method to test model develops
def testModel(model, test_input, test_output, headers):
    # Check if best_score is sufficient enough to use test data.
    if (model.best_score_ > 0.9):
        # Predict labels
        start = time.time()
        predictedData = model.predict(test_input)
        # Measure testing time
        end = time.time()
        print("Testing Time: " + str(end - start))
        print(classification_report(y_true=test_output, y_pred=predictedData, target_names=headers.values()))
        # Generate and show confusion matrix
        plotConfusionMatrix(test_output, predictedData, headers)
    else:
        print("The models score is not good enough:")


# Method to create confusion matrix heatmap
def plotConfusionMatrix(test_output, predictedData, headers):
    confMat = confusion_matrix(test_output.values, predictedData)
    fig, ax = plt.subplots()
    im = ax.imshow(confMat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confMat.shape[1]),
           yticks=np.arange(confMat.shape[0]),
           xticklabels=headers, yticklabels=headers,
           title="Confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')
    # Removed from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    # Loop over data dimensions and create text annotations.
    for i in range(len(headers)):
        for j in range(len(headers)):
            ax.text(j, i, round(confMat[i, j], 3), ha="center", va="center")
    plt.show()


# Only use mean.
train_reduced = train_input.iloc[:, 0:256]
test_reduced = test_input.iloc[:, 0:256]

# Train and test Logistic Regression Model model
model = trainLogisticBinaryModel(train_input, train_output)
testModel(model, test_input, test_output, headers)

# Train and test Logistic Regression Model model with reduced set
model = trainLogisticBinaryModel(train_reduced, train_output)
testModel(model, test_reduced, test_output, headers)

# Train and test Support Vector Machine model
model = trainSVMBinaryModel(train_input, train_output)
testModel(model, test_input, test_output, headers)

# Train and test Support Vector Machine model with reduced set
model = trainSVMBinaryModel(train_reduced, train_output)
testModel(model, test_reduced, test_output, headers)

### Retrain the model and predict unknown labels

# Retrain model with the entire data set means
train_reduced = sensorData.iloc[:, 0:256]
model = trainSVMBinaryModel(train_reduced, classifiedData)

# Classify data for task
classifiedMeans = dataToClassify.iloc[:, 0:256]
predictedUnknownData = model.predict(classifiedMeans)

# Convert predicted data and
predictedUnknownData = predictedUnknownData.astype(int)
np.savetxt("binaryTask/PredictedClasses.csv", predictedUnknownData, delimiter=",")

###
# Multiclass classification task
###

###Load, clean and split data.

# Load data and assign headers
sensorData = pd.read_csv("multiclass/X.csv", header=None, names=range(0, 768))
classifiedData = pd.read_csv("multiclass/y.csv", header=None, names="Y")
dataToClassify = pd.read_csv("multiclass/XToClassify.csv", header=None, names=range(0, 768))

with open("multiclass/key.txt") as file:
    headers = json.load(file)

# Check for null values. Implement cleaning data if there are any.
print(sensorData.isnull().values.any())
print(classifiedData.isnull().values.any())
print(dataToClassify.isnull().values.any())

# Split the data and ignore test data for the rest of the assignment until testing
train_input, test_input, train_output, test_output = train_test_split(sensorData, classifiedData, test_size=TEST_SIZE,
                                                                      random_state=111, stratify=classifiedData)

###Ploting and analyzing data

# Plot all the data
fig = plt.figure(figsize=(20, 10))
# Generate patches for all data
array_of_patches = []
colours = ['green', 'black', 'red', 'blue', 'orange']
# Iterate through classes
for classIndex in range(0, 5):
    label = headers[str(classIndex)]
    color = colours[classIndex]
    plotAllTheData(train_input, train_output, classIndex, color, label)
    array_of_patches.append(mpatches.Patch(color=color, label=label))

# Add legends and graph information
plt.legend(handles=array_of_patches)
plt.show()

# Plot the averaged data
plotAveragesData(train_input, train_output, headers, colours)

# Plot the correlation line for mean components
plotMeanCorrelationLinegraph(train_input, train_output)

### Correlation Heatmap

# Create correlation heat map with Pearson Rank Correlation Coefficients for means
corrData = train_input.iloc[:, 0:64]
correlationMap = corrData.corr(method='pearson')
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(1, 1, 1)
# Show correlation matrix and coloar bar.
im = ax.imshow(correlationMap, interpolation='nearest', cmap='Spectral')
ax.figure.colorbar(im, ax=ax)
# Set axis information
ax.set(xticks=np.arange(correlationMap.shape[1]),
       yticks=np.arange(correlationMap.shape[0]),
       xticklabels=range(1, 64), yticklabels=range(1, 64),
       title="Mean values - Correlation matrix with Pearson coefficient with Channel 1",
       ylabel='Feature number',
       xlabel='Feature number')
plt.show()

### PCA

# PCA 2 component analysis.
# Plot PCA data
pcaData = producePCAData(train_input, train_output)
plotPCAData(pcaData, train_output, headers, colours)


### Train models

# Train svm for multiclass model.
def trainSVMMultiClassModel(trainingInput, trainingOutput):
    # Create scaler and fit it to training input data
    scaler = StandardScaler().fit(trainingInput)
    # Create a pipeline for support vector machine with poly kernel . Add scaler to scale input data.
    pipeline = Pipeline([("scaler", scaler), ("model", svm.SVC(kernel='poly', decision_function_shape='ovr'))])
    # Create regularizatiosn hyperparameter space
    C = [0.01, 0.1, 1, 10]
    gamma = [0.1, 1]
    degree = range(1, 4)
    # Create hyperparameter options
    hyperparameters = {'model__C': C, 'model__gamma': gamma, 'model__degree': degree}
    # 10-fold Cross validation
    start = time.time()
    model = GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=10, refit=True, n_jobs=1,
                         scoring='f1_micro',
                         return_train_score=True)
    # fit model to training data
    model.fit(trainingInput.values, trainingOutput.values.ravel())
    end = time.time()
    print("Training Time: " + str(end - start))
    print(model.best_params_)
    # Print the best parameters
    print("Best score: %0.3f" % model.best_score_)
    return model


# Train Softmax Logistic Regression model for multiclass.
def trainLogisticMulticlassModel(trainingInput, trainingOutput):
    # Create scaler and fit it to training input data
    scaler = StandardScaler().fit(trainingInput)
    # Create a pipeline for Polynomial regression. Add scaler to scale input data.
    pipeline = Pipeline([("scaler", scaler), ("model", LogisticRegression(multi_class='multinomial', solver='lbfgs'))])
    # Create hyperparameter options
    penalty = ['l2'];
    C = [0.01, 0.1, 1]
    hyperparameters = {"model__penalty": penalty, 'model__C': C}
    start = time.time()
    # 10-fold Cross validation
    model = GridSearchCV(estimator=pipeline, param_grid=hyperparameters, cv=10, refit=True, n_jobs=1,
                         scoring='f1_micro',
                         return_train_score=True)

    # fit model to training data
    model.fit(trainingInput.values, trainingOutput.values.ravel())
    end = time.time()
    print("Training Time: " + str(end - start))
    # Print the best parameters
    print(model.best_params_)
    print("Best score: %0.3f" % model.best_score_)
    return model


# Only use mean.
train_reduced = train_input.iloc[:, 0:512]
test_reduced = test_input.iloc[:, 0:512]

### Testing model

# Train and test SVM Regression Model model
model = trainSVMMultiClassModel(train_input, train_output)
testModel(model, test_input, test_output, headers)

# Train and test SVM with smaller set
model = trainSVMMultiClassModel(train_reduced, train_output)
testModel(model, test_reduced, test_output, headers)

# Train and test Logistic Regression Model model
model = trainLogisticMulticlassModel(train_input, train_output)
testModel(model, test_input, test_output, headers)

# Train and test Logistic Regression Model model
model = trainLogisticMulticlassModel(train_reduced, train_output)
testModel(model, test_reduced, test_output, headers)

### Retrain model

# Retrain model with the entire data set
model = trainSVMMultiClassModel(sensorData, classifiedData)

# Classify data for task
predictedUnknownData = model.predict(dataToClassify)
# Convert predicted data and
predictedUnknownData = predictedUnknownData.astype(int)
np.savetxt("multiClassTask/PredictedClasses.csv", predictedUnknownData, delimiter=",")

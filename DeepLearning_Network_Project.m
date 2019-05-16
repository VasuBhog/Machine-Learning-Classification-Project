%Classification using a deep learning network for leaves

clear all; close all; clc;

%%%4.1 Load and Explore Data

%Loading the handwritten data
imds = imageDatastore('Deep_Learning_Project/Leaf', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Counting the number of images
tbl = countEachLabel(imds)

% tbl has uneven number of images per category

%%% find the minimum amount of images in a category
minSetCount = min(tbl{:,2}); % all rows, col 2

%%% Limit number of images to reduce the time it takes
%%% ONLY FOR TESTING - we can use entire leaf
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

%%% Use splitEachLabel to trim the set
%%imds = splitEachLabel(imds,minSetCount,'randomize');

countEachLabel(imds);

numInputFiles = readimage(imds,1);
size(numInputFiles); %Each Image is 175x175x1 pixels

%%% Divide the data into training and validation sets
%%% numTrainFiles can either set a value or list a percent
%%% [TrainingSet, TestSet] = splitEachLabel(imds,numTrainFiles or Percent,'randomize');
[trainingImages, testImages] = splitEachLabel(imds,.7,'randomize');

%%
%%%4.3 Define Network architecture

%Define the Image Input Layer 175x175x3
inputLayer = imageInputLayer([175 175 3]); %specify the image size


middleLayers = [
    
    convolution2dLayer(8,32,'Padding',2)    %conv(filtersize(3x3),#'s of filters,padding,1(spatial output size is same as the input size))
    batchNormalizationLayer                 %Normalize the activations and gradients propagating
    reluLayer()                             %Nonlinear activation function
    maxPooling2dLayer(3,'Stride',2)         %Down-sampling reduces the spatial size of the feature map and removes redundant spatial information
    
    convolution2dLayer(3,64,'Padding',2)
    batchNormalizationLayer
    reluLayer()
    maxPooling2dLayer(2,'Stride',2) 
    
    convolution2dLayer(5,128,'Padding',2)
    batchNormalizationLayer
    reluLayer()
    maxPooling2dLayer(3,'Stride',2) 
    
    convolution2dLayer(5,256,'Padding',2)
    batchNormalizationLayer
    reluLayer()
    maxPooling2dLayer(3,'Stride',2) 
    
    convolution2dLayer(5,512,'Padding',2)
    batchNormalizationLayer
    reluLayer()
    maxPooling2dLayer(3,'Stride',2) 
    
    convolution2dLayer(5,1024,'Padding',2)
    batchNormalizationLayer
    reluLayer()
    maxPooling2dLayer(3,'Stride',2) 
    
];

finalLayers = [

fullyConnectedLayer(4)      %Layer in which neurons connect to all nuerons in preceding layer (combines all featuers learned in previous layers)
softmaxLayer                %Normalizes the output of the fully connected layer
classificationLayer         %Layer uses the probabilites returned by softmax activation function
];


%Define Convolutional Neural Network

layers = [
    inputLayer
    middleLayers
    finalLayers
];


options = trainingOptions('sgdm', ...   %Train Network using stochastic gradient descent with momentum
    'InitialLearnRate',0.01, ...
    'MaxEpochs',6, ...                  %how many times to repeat the training sets (a full training cycle on the entire training data set)
    'Shuffle','every-epoch', ...
    'ValidationData',testImages, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(trainingImages,layers,options);

YPred = classify(net,testImages);
YValidation = testImages.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
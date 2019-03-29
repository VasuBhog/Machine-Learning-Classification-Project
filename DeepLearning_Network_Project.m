%Classification using a simple deep learning network for 
%handwritten numbers

close all; clear all;

%%%4.1 Load and Explore Data

%Loading the handwritten data
imds = imageDatastore('Leaf', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


%Displaying the 20 handwritten numbers
figure;
perm = randperm(200,20); %random permutation of 20 numbers from 1 to 10000
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)})
end

%sgtitle('20 Random Handwritten Numbers');

%Counting the number of images
labelnumbers = countEachLabel(imds);
%Each number(0-9) contain 1000 images each

numInputFiles = readimage(imds,1);
size(numInputFiles) %Each Image is 28x28x1 pixels

%%%4.2 Specify training and validation sets

%Divide the data into training and validation sets
numTrainFiles = 750;
% [imdsTrain, imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
[imdsTrain, imdsValidation] = splitEachLabel(imds,.7,'randomize');

%%
%%%4.3 Define Network architecture

%Define Convolutional Neural Network
layers = [
imageInputLayer([175 175 3]) %specify the image size
convolution2dLayer(3,8,'Padding','same') %conv(filtersize(3x3),#'s of filters,padding,1(spatial output size is same as the input size))
batchNormalizationLayer %Normalize the activations and gradients propagating
reluLayer               %Nonlinear activation function
maxPooling2dLayer(2,'Stride',2) %Down-sampling reduces the spatial size of the feature map and removes redundant spatial information
...
fullyConnectedLayer(10) %Layer in which neurons connect to all nuerons in preceding layer (combines all featuers learned in previous layers)
softmaxLayer            %Normalizes the output of the fully connected layer
classificationLayer];   %Layer uses the probabilites returned by softmax activation function

options = trainingOptions('sgdm', ... %Train Network using stochastic gradient descent with momentum
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...  %how many times to repeat the training sets (a full training cycle on the entire training data set)
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
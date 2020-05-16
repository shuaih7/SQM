function f = LoadKerasModel

% format long;
%************** Part 1: Loading the Tensorflow-Keras Model **************
modelfile = 'model_load.h5';
% net = importKerasLayers(modelfile,'ImportWeights',true);
net = importKerasNetwork(modelfile);
%************************************************************************


%***************** Part 2: Loading the Necessary Files ******************
testfile = 'test.txt';
scalefile = 'scale.txt';
testlabelfile = 'testlabel.txt';

delimiterIn = ' ';
headerlinesIn = 0;
scale = importdata(scalefile,delimiterIn,headerlinesIn);
testlabel = importdata(testlabelfile,delimiterIn,headerlinesIn);
testset0 = importdata(testfile,delimiterIn,headerlinesIn);  % Original 2D Array
%***********************************************************************


%*************** Part 3: Data Rearranging and Preparing ****************
testset = zeros(1,length(testset0(1,:)),1,length(testset0(:,1)));  % 4D Array
for i = 1:1:length(testset0(1,:))
   for j = 1:1:length(testset0(:,1))
      testset(1,i,1,j) = testset0(j,i); 
   end
end
%***********************************************************************


%************** Part 4: Prediction and Error Calculation ***************
test_predictions = predict(net,testset);
test_rearranged = zeros(size(testset));
Error = zeros(1,length(test_predictions(1,:)));
for i = 1:1:length(testset0(1,:))
   for j = 1:1:length(testset0(:,1))
       test_rearranged(j,i) = test_predictions(j,i)*scale(2,i)+scale(1,i);
       Error(i) = abs(test_rearranged(j,i)-testlabel(j,i)); 
   end
   Error(i) = Error(i)/sum(abs(testlabel(:,i)));
end
%***********************************************************************

fprintf('The mean relative error is:');

f = Error;

end
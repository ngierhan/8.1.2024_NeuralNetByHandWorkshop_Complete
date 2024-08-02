% Run Multilayer Perceptron Neural Network with 2 total neurons
%%%%% INPUTS %%%%%
% InputValue: NN input
% w1: NN weight
% b1: NN bias
%%%%% OUTPUTS %%%%%
% z1: NN output

function [z1] = RunNeuralNet_LQR(InputValue,w1, b1)

%% Forward Propogation
z1 = w1*InputValue + b1;
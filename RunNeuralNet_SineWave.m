% Run Multilayer Perceptron Neural Network
%%%%% INPUTS %%%%%
% InputValue: First node of Neural Net
% W: Cell array with NN weights
% B: Cell array with NN biases
%%%%% OUTPUTS %%%%%
% o: NN output

function [o] = RunNeuralNet_SineWave(InputValue,W,B)
w1 = W{1};
w2 = W{2};
w3 = W{3};
b1 = B{1};
b2 = B{2};
b3 = B{3};
%% Forward Propogation
o = [];
for i = 1:length(InputValue)
z1 = w1.*InputValue(i) + b1;
H1 = max(0,z1);
z2 = w2*H1 + b2;
H2 = max(0,z2);
z3 = w3'*H2 + b3;
o(i) = z3;
end

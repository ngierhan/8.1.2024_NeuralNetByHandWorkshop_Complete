% Create Multilayer Perceptron Neural Network
%%%%% INPUTS %%%%%
% InputData: NN Input training data 
% OutputData: NN Output training data 
% h: # neurons in both of 2 hidden layers
% TrainingIters: # NN Training iterations 
%%%%% OUTPUTS %%%%%
% W: Cell array with NN weights
% B: Cell array with NN biases

function [W, B] = CreateNeuralNet_SineWave(InputData, OutputData, h,TrainingIters)

% Initialize weights and biases
w1 = rand(h,1) - 0.5;
b1 = rand(h,1) - 0.5;
w2 = rand(h,h) - 0.5;
b2 = rand(h,1) - 0.5;
w3 = rand(h,1) - 0.5;
b3 = rand(1,1) - 0.5;

for i = 1:TrainingIters
for k = 1:length(InputData)
%% 1. Forward Propogation
z1 = w1.*InputData(k) + b1;
H1 = max(0,z1);
z2 = w2*H1 + b2;
H2 = max(0,z2);
z3 = w3'*H2 + b3;
o = z3;

%% 2. Backward Propogation
% Error = (o - OutputValue)^2
dC_dw3 = 2*(o-OutputData(k))*H2;

diffH1 = [];
diffH2 = [];
for j = 1:length(H1)
if H1(j) == 0
    diffH1(j,1) = 0;
else 
    diffH1(j,1) = 1;
end
if H2(j) == 0
    diffH2(j,1) = 0;
else 
    diffH2(j,1) = 1;
end
end
dC_dw2 = 2*(o-OutputData(k))*(w3.*diffH2)*H1';
dC_dw1 = 2*(o-OutputData(k))*((w3.*diffH2)'*w2)'.*diffH1*InputData(k);

dC_db3 = 2*(o-OutputData(k));
dC_db2 = 2*(o-OutputData(k))*w3.*diffH2;
dC_db1 = 2*(o-OutputData(k))*((w3.*diffH2)'*w2)'.*diffH1;

%% 3. Update Weights and Biases
alpha = .01; % learning rate

w1 = w1 - alpha*dC_dw1;
w2 = w2 - alpha*dC_dw2;
w3 = w3 - alpha*dC_dw3;
b1 = b1 - alpha*dC_db1;
b2 = b2 - alpha*dC_db2;
b3 = b3 - alpha*dC_db3;

end
end
W = {w1,w2,w3};
B = {b1,b2,b3};


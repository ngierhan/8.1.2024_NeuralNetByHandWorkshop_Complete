% main.m
% 1. Neural Network based on LQR data - No Hidden Layer
% 2. Neural Network to fit Sine Wave - 2 Hidden Layers

%% 1. Neural Network based on LQR data - No Hidden Layer
clc
clear
close all

% Gather Data from LQR controller to balance inverted pendulum on cart
[th_LQR, F_LQR] = LQR_pendulum();

TrainingIters = [100;500;1000;5000]; % Compare various training iterations
figure % Figure to compare LQR data and Neural Net

% Plot LQR data
plot(th_LQR,F_LQR,'Linewidth',5) 
for i =1:length(TrainingIters)
[W, B] = CreateNeuralNet_LQR(th_LQR, F_LQR,TrainingIters(i)); % Create NN
hold on

% Plot NN Results
plot(th_LQR,RunNeuralNet_LQR(th_LQR, W, B),'Linewidth',1.5) % Run NN
end
title('Data vs Neural Net')
legend('Input/Output Data LQR', ...
    ['Neural Net: ', num2str(TrainingIters(1)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(2)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(3)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(4)),' iterations'])
ylabel('F')
xlabel('\theta')

%% 2. Neural Network to fit Sine Wave - 2 Hidden Layers
clc
clear
close all

% Generate Input trajectory: sine wave
thvec_data = 0:.01:2*pi;
Fvec_data = sin(thvec_data);

TrainingIters = [100;500;1000;5000]; % Compare various training iterations
figure % Figure to compare input data and Neural Net

% Plot Input trajectory
plot(thvec_data,Fvec_data,'Linewidth',5)
for i =1:length(TrainingIters)
[W,B] = CreateNeuralNet_SineWave(thvec_data, Fvec_data,100,TrainingIters(i)); % Create NN
hold on

% Plot NN Results
plot(thvec_data,RunNeuralNet_SineWave(thvec_data,W,B),'Linewidth',1.5) % Run NN
end
title('Data vs Neural Net')
legend('Input/Output Data LQR', ...
    ['Neural Net: ', num2str(TrainingIters(1)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(2)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(3)),' iterations'], ...
    ['Neural Net: ', num2str(TrainingIters(4)),' iterations'])
ylabel('F')
xlabel('\theta')

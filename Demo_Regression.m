% Demo Data regression of SCN
clear;
clc;
close all;
format long;
tic;

%%  Prepare the training (X, T) and test data (X2, T2) 
% X: each row vector represents one sample input.
% T: each row vector represents one sample target.
% same to the X2 and T2.
% Note: Data preprocessing (normalization) should be done before running the program.
filename = 'newdatajune10-1-2.csv';
Originaldata1 = csvread(filename,0,0,[0,0,4318,3]);
X = Originaldata1(1:2880,:);
X2 = Originaldata1(2880:4319,:);
Originaldata2 = csvread(filename,0,4,[0,4,4318,4]);
T = Originaldata2(1:2880);
T2 = Originaldata2(2880:4319);

%% Parameter Setting
L_max =250;                    % maximum hidden node number
tol = 0.0001;                    % training tolerance
T_max = 100;                    % maximun candidate nodes number
Lambdas = [0.5, 1, 5, 10, ...
    30, 50, 100, 150, 200, 250];% scope sequence
r =  [ 0.9, 0.99, 0.999, ...
    0.9999, 0.99999, 0.999999]; % 1-r contraction sequence
nB = 1;       % batch size

%% Model Initialization
M = SCN(L_max, T_max, tol, Lambdas, r , nB);
disp(M);

%% Model Training
% M is the trained model
% per contains the training error with respect to the increasing L
[M, per] = M.Regression(X, T);
disp(M);

%% Training error demo
figure;
plot(per.Error, 'r.-');
xlabel('L');
ylabel('RMSE');
legend('Training RMSE');

%% Model output vs target on training dataset
O1 = M.GetOutput(X);
figure;
plot(1:2880, T, 'r.-',1:2880, O1, 'b.-');  
xlabel('X');
ylabel('Y');
legend('Training Target', 'Model Output');
E1 = T -  O1; 
MSE1=mse(E1);
RMSE1=sqrt(MSE1);
MAE1=mae(E1);
MAPE1=100*mae(E1./O1);
VAR1=var(E1);

%% Model output vs target on test dataset
O2 = M.GetOutput(X2);
figure;
plot(1:1440, T2, 'r.-',1:1440, O2, 'b.-'); 
xlabel('X');
ylabel('Y');
legend('Test Target', 'Model Output');
E2 = T2 -  O2; 
MSE2=mse(E2);
RMSE2=sqrt(MSE2);
MAE2=mae(E2);
MAPE2=100*mae(E2./O2);
VAR2=var(E2);
toc;
disp(['运行时间: ',num2str(toc)]);
save 20200802.mat
% The End 



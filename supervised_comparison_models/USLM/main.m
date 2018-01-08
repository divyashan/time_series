clear;clc;
Data=read_file('CBF_TRAIN');
[mD,nD]=size(Data);
Y_true=Data(:,1);
DT=Data(:,2:end);
DT=z_regularization(DT);% regularize time series data;
T=[(nD-1)*ones(mD,1),DT]; % Each element in first column of time series matrix is the length of the time series in that row;
N=mD; % Number of time series

%% Parameters ilitialization
Parameter.Lmin=5; % the minimum length of shapelets we plan to learn
Parameter.k=1;% the number of shapelets in equal length 
Parameter.R=3;% the number of scales of shapelets length 
Parameter.C=3;% the number of clusters
Parameter.alpha=-100; % parameter in Soft Minimum Function
Parameter.sigma=1; % parameter in RBF kernel
Parameter.lambda_1=1; % regularization parameter
Parameter.lambda_2=1;% regularization parameter
Parameter.lambda_3=1; % regularization parameter
Parameter.Imax=100; % the number of internal iterations
Parameter.eta=0.01; % learning rate
Parameter.epsilon=0.1; % internal convergence parameter


%Unsupervised Shapelets Learning Algorithm
tic;
[W_star, Y_star, S_star,S_0,F_tp1,wh_time]=USLA(T,Parameter);
%Y_star is the ultimate pseudo-class label matrix under the specific parameters;
%S_star is the shapelets learned by USLA under the specific parameters;
%W_star is the pseudo-classifier under the specific parameters.
time=toc; % time record

Y_true_matrix=reshape_y_ture(Y_true,Parameter.C);
RI = RandIndex(Y_star,Y_true_matrix) % Calculate Rand Index
W_star
S_star
Y_star
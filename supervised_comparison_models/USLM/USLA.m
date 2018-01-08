function [W_star, Y_star, S_star,S_0,F_t,F_tp1,wh_time]=USLA(T,Parameter)

S_0=initialization_s(T,Parameter); % initialize S_0;
[X_0 Xkj_0_skl]=distance_timeseries_shapelet(T,S_0,Parameter.alpha); % calculate the distance matrix X_0 between time series and shapelets S_0.
[Centroid,Y_0]=kmeans(X_0,Parameter.C); % initialize Y_0;
W_0=[-Centroid(1,:);Centroid(2:end,:)]; % initialize W_0;

W_tp1=W_0;
S_tp1=S_0;
Y_tp1=Y_0;
gap=100;
F_tp1=10000;
F_t=F_tp1+10^10;

wh_time=0;
while gap>Parameter.epsilon 

%%%%%%Calculation matrix
[X_tp1 Xkj_tp1_skl]=distance_timeseries_shapelet(T,S_tp1,Parameter.alpha);% update X_tp1;
[L_G_tp1 G_tp1]=Spectral_timeseries_similarity(X_tp1,Parameter.sigma);% update L_G_tp1, the Laplacian matrix of similarity matrix G of time series. 
[SS_tp1 XS_tp1 SSij_tp1_sil]=shapelet_similarity(S_tp1,Parameter.alpha,Parameter.sigma);% update shapelets similarity matrix H_tp1;
F_tp1=function_value(X_tp1,Y_tp1,L_G_tp1,SS_tp1,W_tp1,Parameter); % calculate the value of objective function;
gap=F_t-F_tp1; 

if isnan(F_tp1)
    break;
end

W_tp1=update_W(X_tp1,Y_tp1,Parameter); %update W_tp1;
W_tp1=z_regularization(W_tp1); % regularize W_tp1;
Y_tp1=update_Y(W_tp1,X_tp1,L_G_tp1,Parameter); %update Y_tp1;
S_tp1=update_S(Y_tp1,X_tp1,W_tp1,G_tp1,S_tp1,Xkj_tp1_skl,SSij_tp1_sil,SS_tp1,Parameter); % update S_tp1;
S_tp1=[S_tp1(:,1), z_regularization(S_tp1(:,2:end))]; % regularize S_tp1; The first column is the length of the shapelet in the corresponding row;

F_t=F_tp1;
wh_time=wh_time+1;
if wh_time==15
    break;
end
end

W_star=W_tp1; % the ultimate W_star;
S_star=S_tp1; % the ultimate S_star;

[mY,nY]=size(Y_tp1);
Y_star=zeros(mY,nY);
y_max=max(Y_tp1);

for j=1:nY
    y_index=find(Y_tp1(:,j)==y_max(j));
    Y_star(y_index,j)=1; % the ultimate Y_star;
end


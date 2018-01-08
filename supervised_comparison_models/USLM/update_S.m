function S_tp1=update_S(Y,X,W,ST_t,S_t,Xkj_skl,SSij_sil,SS,Parameter)
DS_t=S_t(:,2:end);
i=1;
while i<Parameter.Imax+1 
    deri_of_S=derivation_of_S(Y,X,W,ST_t,S_t,Xkj_skl,SSij_sil,SS,Parameter);
    DS_t=S_t(:,2:end);
    DS_t=DS_t-Parameter.eta*deri_of_S;
    S_t=[S_t(:,1) DS_t];
    i=i+1;
end
S_tp1=S_t;

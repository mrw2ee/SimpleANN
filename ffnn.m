function [loss,gradient]=ffnn(Ws,xTr,yTr,wst)
% function w=ffnn(Ws,xTr,yTr,wst)
%
% Feed forward neural network with sigmoid transition function
%
% INPUT:
% W weights
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% ws weight structure (e.g. [1 10 5] 1 output node, 10 hidden, 5 input)
%
% if yTr=[] then loss=prediction of the data xTr
%
% OUTPUTS:
%
% loss = the total loss obtained with w on xTr and yTr, or the prediction of yTr is not passed on
% gradient = the gradient at w
%
% Copyright (C) 2016 - 2019 by Michael R. Walker II

% Definition of sigma and its derivative
sig=@(z) 1./(1+exp(-z));
sigp=@(sz) sz.*(1-sz);

% Determine the number of weigts for each state + 'b' parameters
% Each stage: n outputs x m inputs + b : (n x (m+1)) Weights
% Where: n = wst(i), m = wst(i+1)

% reformat the data from one vector to a cell-array of matrices
entry=cumsum(wst(1:end-1).*wst(2:end)+wst(1:end-1));
if isempty(Ws)
    Ws=randn(entry(end),1)./100;
end
W={};
e=1;
for i=1:length(entry)
    W{i}=reshape(Ws(e:entry(i)),[wst(i),wst(i+1)+1]);
    e=entry(i)+1;
end

[~,n]=size(xTr);

% first, we add the constant weight
zs{length(W)+1}=[xTr;ones(1,n)];
% Do the forward process here:
for i=length(W):-1:2
    alpha = W{i}*zs{i+1};
    zs{i} = [sig(alpha);ones(1,n)];
end
% last one is special, no transition function
i = 1;
zs{i} = W{i}*zs{i+1};

% If [] is passed on as yTr, return the prediction as loss and exit
if isempty(yTr)
    loss=zs{1};
    return;
end

% otherwise compute loss
delta=zs{1}-yTr;
loss=0.5*sum(delta(:).^2);

if nargout>1
    % compute gradient with back-prop
    gradient=zeros(size(Ws));
    e=1;
    for i=1:length(W)
        gradient(e:entry(i)) = delta*zs{i+1}.';
        delta = (W{i}(:,1:end-1).'*delta).*sigp(zs{i+1}(1:end-1,:));
        e=entry(i)+1;
    end
end

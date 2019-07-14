function [xTr,xTe,u,m]=preprocess(xTr,xTe)
% function [xTr,xTe,u,m]=preprocess(xTr,xTe);
%
% Preproces the data to make the training features have zero-mean and
% standard-deviation 1
%
% output:
% xTr - pre-processed training data 
% xTe - pre-processed testing data
% 
% u,m - any other data should be pre-processed by x-> u*(x-m)
%

m = mean(xTr,2);
u = sqrt(var(xTr,1,2));

xTr = (xTr - repmat(m,1,size(xTr,2))).*repmat(1./u,1,size(xTr,2));
xTe = (xTe - repmat(m,1,size(xTe,2))).*repmat(1./u,1,size(xTe,2));

%xTr = (xTr - repmat(m,1,size(xTr,2)));
%u = pcacov(xTr.');
%xTr = u.'*xTr;
%xTe = u.'*(xTe - repmat(m,1,size(xTe,2)));


Simple artificial neural network implementation for MATLAB

Tools provided for the training and testing of a feed-forward neural network using the sigmoid transition function.

The template for this code was provided by Kilian Q. Weinberger as part of the Machine Learning course at at Washington University in St. Louis: CSE517a, Spring 2016. We also include code by Carl Edward Rasmussen.

For reference we provide example code. We assume available variables
 xTr - training features, M x Ntr matrix
 yTr - training labels, P x Ntr  matrix
 xTe - testing features, M x Nte matrix
 yTe - testing labels, P x Nte matrix
Where Ntr and Nte are the number of samples for training and testing, respectively. M and P are the number of features and labels, respectively.

% -------------------- START OF SOURCE CODE --------------------

wst=[1 12 size(xTr,1)];
w=initweights(wst);

[xTr,xTe]=preprocess(xTr,xTe);
graderr=checkgrad('ffnn',w,1e-05,xTr,yTr,wst)
[w,l]=minimize(w,'ffnn',100,xTr,yTr,wst);


if graderr<1e-05
        disp('Gradient looks good. Implementation probably correct!');
else
        disp('Error! Gradient incorrect!!!');
end

trainerr=sqrt(mean((ffnn(w,xTr,[],wst)-yTr).^2));
testerr=sqrt(mean((ffnn(w,xTe,[],wst)-yTe).^2));

% --------------------- END OF SOURCE CODE ---------------------

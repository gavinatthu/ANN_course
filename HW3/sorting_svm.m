clear;
clc;
close all;


[x,t] = wine_dataset;
x = x.';
c = t(2,:) + 2 * t(3,:);

mdl = fitcecoc(x, c);
a = predict(mdl,x);
mdl.ClassNames;
CodingMat = mdl.CodingMatrix;
mdl.BinaryLearners{1};

error = resubLoss(mdl)

%view(net)
%y = net(x);

%classed = vec2ind(y);

figure(1)

scatter(x(:,1),x(:,2),25,a,'filled')

%figure(2)
%plotconfusion(t, y)
%plotsomtop(net)
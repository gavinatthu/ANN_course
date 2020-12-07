clear;
clc;
close all;


[x,t] = wine_dataset;
net = selforgmap([1 3] ,'coverSteps',100,'initNeighbor',10);
[net, tr] = train(net,x);

%view(net)
y = net(x);

classed = vec2ind(y);

figure(1)

scatter(x(1,:),x(2,:),25,classed,'filled')

figure(2)
%plotconfusion(t, y)
%plotsomtop(net)

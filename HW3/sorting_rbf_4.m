clear;
clc;
close all;


[x,t] = simpleclass_dataset;

net = newrb(x,t,1e-3,1,500,10);
Y = sim(net,x);

view(net)

y = net(x);
classed = vec2ind(y);

figure(1)

scatter(x(1,:),x(2,:),25,classed,'filled')

figure(2)
plotconfusion(t, y)


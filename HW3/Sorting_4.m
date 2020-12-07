clear;
clc;
close all;


[x,t] = wine_dataset;

net = patternnet(5);
[net,tr] = train(net,x,t);
view(net)
nntraintool
%nntraintool('close')
y = net(x);
classed = vec2ind(y);

figure(1)

scatter(x(1,:),x(2,:),25,classed,'filled')

figure(2)
plotperform(tr)

figure(3)
plotconfusion(t, y)


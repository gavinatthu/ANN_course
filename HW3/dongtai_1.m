clear;
clc;
[x,t] = maglev_dataset;
net = narnet(1:2,10);
%view(net)
[Xs,Xi,Ai,Ts] = preparets(net,{},{},t);
[net,tr] = train(net,Xs,Ts,Xi,Ai);
nntraintool
nntraintool('close')
figure(1)
plotperform(tr)
Y = net(Xs,Xi,Ai);
perf = mse(net,Ts,Y);
figure(2)
plotresponse(Ts,Y)
clear;
clc;
f = load('HMW3-DATA.MAT');
temp = num2cell(f.random3600_1_temp.');
time = num2cell(f.random3600_1_time.');
volt = num2cell(f.random3600_1_volt.');

temp_t = num2cell(f.sin3600_1_temp.');
time_t = num2cell(f.sin3600_1_time.');
volt_t = num2cell(f.sin3600_1_volt.');

net = narxnet(1:2,1:2,15);
%view(net)
[Xs,Xi,Ai,Ts] = preparets(net,volt,{},temp);
[Xs_t,Xi_t,Ai_t,Ts_t] = preparets(net,volt_t,{},temp_t);
[net,tr] = train(net,Xs,Ts,Xi,Ai);
nntraintool
nntraintool('close')
figure(1)
plotperform(tr)
Y = net(Xs,Xi,Ai);
perf = mse(net,Ts,Y);
Y_t = net(Xs_t,Xi_t,Ai_t);
perf_t = mse(net,Ts_t,Y);
figure(2)
plotresponse(Ts,Y)
figure(3)
plotresponse(Ts_t,Y_t)
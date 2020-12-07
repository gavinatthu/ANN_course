clear;
clc;
f = load('RotateData.MAT');
rot = num2cell(f.rotspeed);
ang = mat2cell(f.angledata,2,ones(1,2000));
err = normrnd(0,10,2,2000);
rot_t = num2cell(f.rotspeed + err(1,:));
ang_t = mat2cell(f.angledata + err,2,ones(1,2000));
net = narxnet(1:2,1:2,30);
%view(net)
[Xs,Xi,Ai,Ts] = preparets(net,ang,{},rot);
[Xs_t,Xi_t,Ai_t,Ts_t] = preparets(net,ang_t,{},rot);
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
figure(4)
plot(f.angledata(1,:) + err(1,:))
hold on 
plot(f.angledata(2,:) + err(2,:))
hold on 
plot(f.rotspeed+ err(1,:))
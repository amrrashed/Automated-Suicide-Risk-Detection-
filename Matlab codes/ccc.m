clc;clear;close all;clf
fs = 10000;
t = 0:1/fs:1;
fm=sin(2*pi*5*t);%message signal
%fm = sin(2*pi*t) + sin(4*pi*t);
fc=10;%carrier signal freq
y=cos(2*pi*fc*t);%carrier signal
phasedev=pi/2; %phase deviation in radian
phasemod=pmmod(fm,fc,fs,phasedev);%modulated signal
%Pass the signal through an AWGN channel.
rx = awgn(phasemod,25,'measured');
phasedemod=pmdemod(rx,fc,fs,phasedev);% demolated signal
figure(1)
subplot(511)
plot(t,fm)
title('message signal')
subplot(512)
plot(t,y)
title('carrier signal')
subplot(513)
plot(t,phasemod)
title('modulated signal')
subplot(514)
plot(t,rx)
title('modulated signal after adding noise')
subplot(515)
plot(t,phasedemod)
title('demodulated signal')
figure(2)
subplot(3,1,1)
plot(fftshift(abs(fft(fm))));
subplot(3,1,2)
plot(fftshift(abs(fft(phasemod))));
subplot(3,1,3)
plot(fftshift(abs(fft(phasedemod))));
figure(3); 
plot(t,fm,'k');
hold on;
plot(t,phasedemod,'y');
legend('Original signal','Recovered signal');
xlabel('Time (s)')
ylabel('Amplitude (V)')
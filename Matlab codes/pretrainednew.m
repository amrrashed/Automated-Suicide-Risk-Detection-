clc;clear all;close all
%x=cell(4,5);
[t,x]=nasnetmobilecv('D:\new researches\SUICIDE PAPER\Dataset\our DB1\crop224new','rmsprop',0,5);
t=ceil(t.*10000)/10000;



function stop=savetrainingplot(info)
stop=false;  %prevents this function from ending trainNetwork prematurely
if info.State=='done'   %check if all iterations have completed
% if true
x=fix(clock);
       % saveas(gcf,'training_process.png')  % save figure as .png, you can change this
       NNET_CNN= findall(groot, 'Type', 'Figure');
   %  print(NNET_CNN(end),strcat('training_noaug_dr_mahdi_model_dpi600',num2str(x(5))),'-dpng','-r600');
      %saveas(NNET_CNN(end),strcat('training_noaug_xceptionsgdm_',num2str(x(5)),'.png'));
      %saveas(NNET_CNN(end),strcat('training_noaug_xceptionsgdm_',num2str(x(5)),'.eps'));
      print(NNET_CNN(end),strcat('training_alexcustom_dpi1000',num2str(x(5))),'-dpng','-r1000');
end
end
% function stop=savetrainingplot(info)
% stop=false;  %prevents this function from ending trainNetwork prematurely
% if info.State=='done'   %check if all iterations have completed
% % if true
%       saveas(findall(groot, 'Type', 'Figure'),'training_process.jpg')  % save figure as .png, you can change this
%       
%       saveas(findall(groot, 'Type', 'Figure'),'training_process.fig')
%       
% %       savefig(findall(groot, 'Type', 'Figure'),'training_process.fig')
% end
% end
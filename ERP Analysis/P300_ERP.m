clear all

VS=textread(strcat('s_oddball.txt'));
VT=textread(strcat('t_oddball.txt'));

Fsamp=500;
[b,a]=butter(10, 20/250, 'low');
f_VS=filter(b,a,VS');
f_VT=filter(b,a,VT');

b_f_VS=baseline_1ch(f_VS);
b_f_VT=baseline_1ch(f_VT);
   
a_b_f_VS=artifact(b_f_VS);
a_b_f_VT=artifact(b_f_VT);
   
av_a_b_f_VS=average(a_b_f_VS);
av_a_b_f_VT=average(a_b_f_VT);
   
createfigure_ERP(av_a_b_f_VS, av_a_b_f_VT);

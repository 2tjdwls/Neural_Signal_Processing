%% Data read
clear all

VS=textread(strcat('s_oddball.txt'));
VT=textread(strcat('t_oddball.txt'));

%% Variable def

sp = 750; % sampling point
t_VS = size(VS,2)/sp; % VS trial 216
t_VT = size(VT,2)/sp; % VT trial 54
Fsamp=500;

a = VS';
b = VT';

figure();
subplot(2,1,1), plot(a), title("Standard EEG"), xlabel("Sampling point (162000)"), ylabel("Amp(μV)");
subplot(2,1,2), plot(b), title("Target EEG"), xlabel("Sampling ploint (40500)"), ylabel("Amp(μV)");
%% Filtering

[b,a]=butter(10, 20/250, 'low'); % Butterworth lowpass filter, Wn : cutoff frequency, n : 10차

f_VS=filter(b,a,VS'); % Transfer function coefficient = b,a / Rational Transfer Function -> VS' filtering / ' : ctranspose
f_VT=filter(b,a,VT'); % "

figure();
subplot(2,1,1), plot(f_VS), title("Filtered standard EEG"), xlabel("Sampling point (162000)"), ylabel("Amp(μV)");
subplot(2,1,2), plot(f_VT), title("Filtered target EEG"), xlabel("Sampling ploint (40500)"), ylabel("Amp(μV)");

%% Baseline correction

for i_t = 1:t_VS % trial : 1~216
    for i_c = 1:4 % 1ch~4ch
        dvs(i_t,i_c) = sum(f_VS(sp*(i_t-1)+201:sp*(i_t-1)+250 , i_c))/50;
    end
end

for i_t = 1:t_VS
    for i_c = 1:4
        b_f_VS(sp*(i_t-1)+1:sp*(i_t-1)+sp , i_c) = f_VS(sp*(i_t-1)+1:sp*(i_t-1)+sp , i_c) - dvs(i_t , i_c);
    end
end

for i_t = 1:t_VT % 1~54
    for i_c = 1:4
        dvt(i_t,i_c) = sum(f_VT(sp*(i_t-1)+201 : sp*(i_t-1)+250 , i_c))/50;
    end
end

for i_t = 1:t_VT
    for i_c = 1:4
        b_f_VT(sp*(i_t-1)+1:sp*(i_t-1)+sp , i_c) = f_VT(sp*(i_t-1)+1:sp*(i_t-1)+sp , i_c) - dvt(i_t , i_c);
    end
end

%% Averaging

t = -0.5:1/500:1-1/500;

av_b_f_VS = 0;
av_b_f_VT = 0;

for i_t = 1:t_VS
    a_b_f_VS = b_f_VS((sp*(i_t-1))+1:sp*i_t, :);
    av_b_f_VS = av_b_f_VS + a_b_f_VS;
end

for i_t = 1:t_VT
    a_b_f_VT = b_f_VT((sp*(i_t-1))+1:sp*i_t, :);
    av_b_f_VT = av_b_f_VT + a_b_f_VT;
end

av_b_f_VS = av_b_f_VS/t_VS;
av_b_f_VT = av_b_f_VT/t_VT;

%% Plot

figure();
subplot(2,2,1), p_VS_1 = plot(t(251:end),av_b_f_VS(251:end,1)); hold on; p_VT_1 = plot(t(251:end),av_b_f_VT(251:end,1)); hold off; xlabel('Time(ms)'), ylabel('Amp(μV)'); title("Channel 1")
subplot(2,2,2), p_VS_2 = plot(t(251:end),av_b_f_VS(251:end,2)); hold on; p_VT_2 = plot(t(251:end),av_b_f_VT(251:end,2)); hold off; xlabel('Time(ms)'), ylabel('Amp(μV)'); title("Channel 2")
subplot(2,2,3), p_VS_3 = plot(t(251:end),av_b_f_VS(251:end,3)); hold on; p_VT_3 = plot(t(251:end),av_b_f_VT(251:end,3)); hold off; xlabel('Time(ms)'), ylabel('Amp(μV)'); title("Channel 3")
subplot(2,2,4), p_VS_4 = plot(t(251:end),av_b_f_VS(251:end,4)); hold on; p_VT_4 = plot(t(251:end),av_b_f_VT(251:end,4)); hold off; xlabel('Time(ms)'), ylabel('Amp(μV)'); title("Channel 4")

set(p_VS_1,'LineStyle',':','Color',[1 0 0],'DisplayName','EASY');
set(p_VS_2,'LineStyle',':','Color',[1 0 0],'DisplayName','EASY');
set(p_VS_3,'LineStyle',':','Color',[1 0 0],'DisplayName','EASY');
set(p_VS_4,'LineStyle',':','Color',[1 0 0],'DisplayName','EASY');
set(p_VT_1,'Color',[0 0 1],'DisplayName','DIFFICULT');
set(p_VT_2,'Color',[0 0 1],'DisplayName','DIFFICULT');
set(p_VT_3,'Color',[0 0 1],'DisplayName','DIFFICULT');
set(p_VT_4,'Color',[0 0 1],'DisplayName','DIFFICULT');
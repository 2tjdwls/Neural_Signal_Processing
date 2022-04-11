%% PSD estimation
cd('C:\Users\nelab_under\Desktop\뇌신경공학 연구실\캡스톤디자인\eeg');
%% loading LSJ KSS data
KSS_LSJ = readmatrix('KSS.xlsx', 'Sheet', 'LSJ', 'range', 'C2:CN22', 'OutputType', 'double');
%% loading KJY KSS data
KSS_KJY = readmatrix('KSS.xlsx', 'Sheet', 'KJY', 'range', 'C2:CN22', 'OutputType', 'double');
%% averaging KSS
KSS_aver = (KSS_LSJ + KSS_KJY)/2;
%% figure KSS
time = readmatrix('KSS.xlsx', 'Sheet', 'LSJ', 'range', 'C1:CN1', 'OutputType', 'double');

threshold = repmat(6.5,1,90);

for sub = 1:size(KSS_aver,1)
    LSJ_data = KSS_LSJ(sub,:);
    KJY_data = KSS_KJY(sub,:);
    total_data = KSS_aver(sub,:);

    sub_name = strjoin(["subject", num2str(sub)], '');
    figure("Name", sub_name);
    plot(time, threshold, 'b--', time, total_data, 'r', time, LSJ_data, 'c-.', time, KJY_data, 'g-.')
    legend('threshold', 'total', 'LSJ', 'KJY', 'Location', 'southeast')
end
%% KSS labeling
KSS_label = zeros(1,1);

for i = 1:size(KSS_aver,1)
    for j = 1:size(KSS_aver,2)
        if KSS_aver(i,j) >= 6.5
            KSS_label(i,j) = "1";
        else
            KSS_label(i,j) = "0";
        end
    end
end
save('KSS.mat','KSS_label')

%% loading EEG data
filepath = dir('*_*.mat');
filename = {filepath.name};
Fs = 200; %sampling rate
min_length = 90; %min
window = Fs*60; %1 min window

%% alert & fatigue period in KSS 
sub_alt_period = zeros(1,1);
sub_ftg_period = zeros(1,1);
sub_alt_ftg_period = zeros(1,1);

for i = 1:21 % subjects 1~21
    k=1;
    l=1;
    for j = 1:90 % 1~90 min
        if KSS_label(i,j) == 0        
            sub_alt_period(i,k) = j;
            k = k+1;
        else KSS_label(i,j) == 1
            sub_ftg_period(i,l) = j;
            l = l+1;
        end
    end
end

for i = 1:21
    for j =1:nnz(sub_alt_period(i,:))
        sub_alt_ftg_period(i,j) = sub_alt_period(i,j);
    end

    for k = 1:nnz(sub_ftg_period(i,:))
        sub_alt_ftg_period(i,nnz(sub_alt_period(i,:))+k) = sub_ftg_period(i,k);
    end
end

sub_alt_ftg_len = zeros(1,1);

for i=1:21
    sub_alt_ftg_len(1,i) = nnz(sub_alt_period(i,:));
    sub_alt_ftg_len(2,i) = nnz(sub_ftg_period(i,:));
end

%% EEG PSD estimation
filepath = dir('*_psd.mat');
filename = {filepath.name};

for subject = 1:21
    load(filename{subject});
    eeg = double(data); clear data
    count = 1;
    for ch = 1 : 64
        for t = 1 : 95
            count = count+1;
            disp(['  subject:  ',num2str(subject), '  channel: ',num2str(ch),'  time: ',num2str(t)]);
            sample_point = (t-1)*window+1:t*window; % first 5min = baseline
            [psd, freq] = pwelch(eeg(ch,sample_point),Fs,0.5*Fs,Fs,Fs); % 400 points of 1:Fs Hz = 1 Hz resolution
            psd_est(ch,t,1:100) = psd(2:101);
        end
    end
    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_psd.mat'],'psd_est');clear psd_est;
end

%% Baseline correction in frequency domain
for subject = 1:21
    load(filename{subject});
    baseline = 1:5; %min
    psd_base = repmat(mean(psd_est(:,baseline,:),2),[1,95,1]);

    psd_corrected = (psd_est - psd_base)./psd_base;

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_corrected.mat'],'psd_corrected'); clear psd_corrected;
end

%% Alert, Fatigue division
clc;clear;
filepath = dir('*corrected*.mat');
filename = {filepath.name};
load('KSS.mat')

for subject = 1:21
    load(filename{subject});
    psd_corrected(:,1:5,:) = [];
    KSS_ind = KSS_label(subject,:);

    psd_fatigue = psd_corrected(:,logical(KSS_ind),:);
    psd_alert = psd_corrected(:,~logical(KSS_ind),:);

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    
    save([fname_new,'_div.mat'],'psd_fatigue','psd_alert'); clear psd*;
end

%% Band power
% delta : 0.5 ~ 4Hz -> 1 ~ 4
% theta : 4 ~ 8Hz   -> 5 ~ 8
% alpha : 8 ~ 13Hz  -> 9 ~ 13
% beta  : 13 ~ 30Hz -> 14 ~ 30
% gamma : 30 ~ 50Hz -> 31 ~ 50
clc;clear;
filepath = dir('*div.mat');
filename = {filepath.name};

psd_alert_full = [];
psd_fatigue_full = [];
for subject = 1:21
    load(filename{subject});

    mean_psd_alert = squeeze(mean(psd_alert,2));
    mean_psd_fatigue = squeeze(mean(psd_fatigue,2));

    psd_alert_full = cat(3,psd_alert_full,mean_psd_alert);
    psd_fatigue_full = cat(3,psd_fatigue_full,mean_psd_fatigue);
end
save('PSD_average.mat','psd_alert_full','psd_fatigue_full');
%%
clc;clear;
load('PSD_average.mat')
band = [1,4;5,8;9,13;14,30;31,50];

for freq = 1:5
    psd_alert_band(:,freq,:) = mean(psd_alert_full(:,band(freq,1):band(freq,2),:),2);
    psd_fatigue_band(:,freq,:) = mean(psd_fatigue_full(:,band(freq,1):band(freq,2),:),2);
end

load('chanlocs.mat')
IOR = [24,26,29,31]; %P3, P4, O1, O2
IOR_name = {'P3','P4','O1','O2'};
alert_plot = mean(psd_alert_band(IOR,:,:),3);
fatigue_plot = mean(psd_fatigue_band(IOR,:,:),3);

for sp = 1:4
    subplot(2,2,sp);
    bar([alert_plot(sp,:)',fatigue_plot(sp,:)']);
    xlabel(IOR_name{sp})
end


%% Statistics analysis

for freq = 1:5
    [h,p] = ttest(squeeze(psd_alert_band(:,freq,:))', squeeze(psd_fatigue_band(:,freq,:))');

    p_val(:,freq) = p;
    h_val(:,freq) = h;
end

[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(p_val);

%% Topography
alert_topo = mean(psd_alert_band,3);
fatigue_topo = mean(psd_fatigue_band,3);
diff_topo = fatigue_topo - alert_topo;

alert_band_name = {'Alert state in Delta band', 'Alert state in Theta band', 'Alert state in Alpha band', 'Alert state in Beta band', 'Alert state in Gamma band'};
fatigue_band_name = {'Fatigue state in Delta band', 'Fatigue state in Theta band', 'Fatigue state in Alpha band', 'Fatigue state in Beta band', 'Fatigue state in Gamma band'};
diff_name = {'Fatigue - Alert in Delta band', 'P-value distribution in Theta band', 'P-value distribution in Alpha band', 'P-value distribution in Beta band', 'P-value distribution in Gamma band'};

marker_ch = find(adj_p(:,3)<0.05)';

figure
for freq = 1:5
    subplot(3,5,freq); topoplot(alert_topo(:,freq),chanlocs, 'maplimits', [mean(alert_topo(:,freq))-3*std(alert_topo(:,freq)) mean(alert_topo(:,freq))+5*std(alert_topo(:,freq))]); title (alert_band_name{freq});
    subplot(3,5,freq+5); topoplot(fatigue_topo(:,freq),chanlocs, 'maplimits', [mean(alert_topo(:,freq))-3*std(alert_topo(:,freq)) mean(alert_topo(:,freq))+5*std(alert_topo(:,freq))]); title (fatigue_band_name{freq});
    subplot(3,5,freq+10); topoplot(diff_topo(:,freq),chanlocs, 'maplimits', [0 0.1]); title(diff_name{freq});
end

figure
for freq = 1:5
    subplot(3,5,freq); topoplot(alert_topo(:,freq),chanlocs,'maplimits', [-0.07 0.25]); title (alert_band_name{freq});
    subplot(3,5,freq+5); topoplot(fatigue_topo(:,freq),chanlocs, 'maplimits', [-0.05 0.37]); title (fatigue_band_name{freq});
    subplot(3,5,freq+10); topoplot(diff_topo(:,freq),chanlocs, 'maplimits', [0 0.1]); title(diff_name{freq});
end

subplot(3,5,13); topoplot(diff_topo(:,3),chanlocs, 'maplimits', [0 0.065], 'emarker2', {marker_ch, 'o', 'k'}); title(diff_name{3});


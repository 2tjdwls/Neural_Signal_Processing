
%% EEG PSD estimation
clc; clear;
filepath = dir('*.mat');
filename = {filepath.name};

Fs = 200; %sampling rate
min_length = 90; %min
window = Fs*60; %1 min window

count = 1;
sec = 60;

for subject = 1:21
    load(filename{subject});
    eeg = double(data); clear data
    count = 1;
    for ch = 1 : 64
        for s = 1 : 121
                sec = 2*(s-1);
                count = count+1;
                disp([num2str(round(100*count/(21*64*2821))),'%, subject:',num2str(subject), '  channel: ',num2str(ch),'  epoch: ',num2str(s)]);
                sample_point = Fs*sec + 1 : Fs*(sec+60); % first 5min = baseline
                [psd, freq] = pwelch(eeg(ch,sample_point),Fs,0.5*Fs,Fs,Fs); % 400 points of 1:Fs Hz = 1 Hz resolution
                psd_est(ch,s,1:100) = psd(2:101);
        end

        for s = 151 : 2821
                sec = 2*(s-1);
                count = count+1;
                disp([num2str(round(count/(21*64*2821))),'%, subject:',num2str(subject), '  channel: ',num2str(ch),'  epoch: ',num2str(s)]);
                sample_point = Fs*sec+1 : Fs*(sec+60); % first 5min = baseline
                [psd, freq] = pwelch(eeg(ch,sample_point),Fs,0.5*Fs,Fs,Fs); % 400 points of 1:Fs Hz = 1 Hz resolution
                psd_est(ch,s-29,1:100) = psd(2:101);
        end
    end

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_psd.mat'],'psd_est');clear psd_est;
end

%% Baseline correction in frequency domain
clc; clear;
filepath = dir('*_psd.mat');
filename = {filepath.name};

for subject = 1:21
    load(filename{subject});
    baseline = 1:121;
    psd_base = repmat(mean(psd_est(:,baseline,:),2),[1,2792,1]);

    psd_corrected = (psd_est - psd_base)./psd_base;

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_corrected.mat'],'psd_corrected'); clear psd_corrected;
end

filepath = dir('*_corrected.mat');
filename = {filepath.name};

%% KSS augmentation

load('KSS_aver.mat')

for subject = 1:21
    KSS_rep(subject,:) = repelem(KSS_aver(subject,:),60);
end

KSS_smooth = movmean(KSS_rep, 60, 2);

for subject = 1:21
        for s = 1 : 2671
                sec = 2*(s-1)+1;
                KSS_aug(subject,s) = mean(KSS_smooth(subject,sec : sec+59),2);
        end
end

KSS_label = zeros(1,1);

for i = 1:size(KSS_aug,1)
    for j = 1:size(KSS_aug,2)
        if KSS_aug(i,j) >= 6.5
            KSS_label(i,j) = "1";
        else
            KSS_label(i,j) = "0";
        end
    end
end
save('KSS_aug.mat','KSS_label')
%% Alert, Fatigue division
clc;clear;
filepath = dir('*_corrected.mat');
filename = {filepath.name};
load('KSS_aug.mat')

for subject = 1:21
    load(filename{subject});
    psd_corrected(:,1:61,:) = [];
    KSS_ind = KSS_label(subject,:);

    psd_fatigue = psd_corrected(:,logical(KSS_ind),:);
    psd_alert = psd_corrected(:,~logical(KSS_ind),:);

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    
    save([fname_new,'_div.mat'],'psd_fatigue','psd_alert'); clear psd*;
end

%% Band power
% delta : 0.5 ~ 4Hz -> 1 ~ 4
% theta : 4 ~ 8Hz   -> 4 ~ 8
% alpha : 8 ~ 13Hz  -> 8 ~ 13
% beta  : 13 ~ 30Hz -> 13 ~ 30
% gamma : 30 ~ 50Hz -> 30 ~ 50

clc;clear;
filepath = dir('*div*.mat');
filename = {filepath.name};

band = [1,4;5,8;9,13;14,30;31,50];

for subject = 1:21
    load(filename{subject});
    for freq = 1:5
        psd_alert_band(:,:,freq) = mean(psd_alert(:,:,band(freq,1):band(freq,2)),3);
        psd_fatigue_band(:,:,freq) = mean(psd_fatigue(:,:,band(freq,1):band(freq,2)),3);
    end

    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_band.mat'],'psd_alert_band','psd_fatigue_band');clear psd_alert_band; clear psd_fatigue_band;
end

%% Channel selection
clc;clear;
filepath = dir('*_band.mat');
filename = {filepath.name};

load('FDR_h.mat');
ch_sel = find(h(:,3));


for subject = 1:21
    load(filename{subject});
    for ch = ch_sel(1:53,1)
        psd_alert_band_ch = psd_alert_band(ch,:,:);
        psd_alert_band_ch = permute(psd_alert_band_ch, [2 3 1]);

        psd_fatigue_band_ch= psd_fatigue_band(ch,:,:);
        psd_fatigue_band_ch = permute(psd_fatigue_band_ch, [2 3 1]);
    end
    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_ch.mat'],'psd_alert_band_ch','psd_fatigue_band_ch');clear psd_alert_band_ch; clear psd_fatigue_band_ch;
end


%% Data set making (csv file)

clc;clear;
filepath = dir('*_ch.mat');
filename = {filepath.name};

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(psd_alert_band_ch(:,1,1)),1);
    label_fatigue = ones(numel(psd_fatigue_band_ch(:,1,1)),1);

    psd_alert_dataset = reshape(psd_alert_band_ch, [numel(psd_alert_band_ch(:,1,1)), 5*53]);
    psd_fatigue_dataset = reshape(psd_fatigue_band_ch, [numel(psd_fatigue_band_ch(:,1,1)), 5*53]);

    psd_alert_dataset = horzcat(label_alert,psd_alert_dataset);
    psd_fatigue_dataset = horzcat(label_fatigue,psd_fatigue_dataset);


    fname_new = filename{subject};
    fname_new(end-3:end)=[];
    save([fname_new,'_dataset.mat'],'psd_alert_dataset','psd_fatigue_dataset');clear psd_alert_dataset; clear psd_fatigue_dataset;
end

clc; clear;
filepath = dir('*_dataset.mat');
filename = {filepath.name};


for subject = 1:21
    load(filename{subject});
    writematrix(psd_alert_dataset, 'train.csv','WriteMode', 'append');
    writematrix(psd_fatigue_dataset, 'train.csv', 'WriteMode', 'append');
end

% 21명 -> 20명 / 1명 (trrain / test)

%% Topography
% Topography input data 생성
clc, clear;

eeglab; clear;
load('chanlocs.mat')

ch_del = [7,10,14,22,30,31,41,48,51,58,62];
chanlocs(ch_del) = [];
chanlocs_sel = chanlocs;
save('chanlocs_sel.mat', 'chanlocs_sel');

load('chanlocs_sel.mat')

filepath = dir('*_ch.mat');
filename = {filepath.name};

topo_alert = [];
topo_fatigue = [];

for subject = 1:21
    for band = 1:5
        load(filename{subject});

        for sample = 1 : numel(psd_alert_band_ch(:,1,1)) % alert
            [psd_plot_alert ,topo_alert_value] = topoplot(psd_alert_band_ch(sample,band,:), chanlocs_sel,'conv','on','noplot','on');
            topo_alert(sample,:,:) = topo_alert_value;
            disp(['subject:',num2str(subject), '  band: ',num2str(band),'  sample: ',num2str(sample), ' state: alert']);

        end

        for sample = 1 : numel(psd_fatigue_band_ch(:,1,1)) % fatigue
            [psd_plot_fatigue, topo_fatigue_value] = topoplot(psd_fatigue_band_ch(sample,band,:), chanlocs_sel,'conv','on','noplot','on');
            topo_fatigue(sample,:,:) = topo_fatigue_value;
            disp(['subject:',num2str(subject), '  band: ',num2str(band),'  sample: ',num2str(sample), ' state: fatigue']);
        end

    fname_new = filename{subject};
    fname_new(end-29:end)=[];

        if band == 1
            save([fname_new, '_delta_dataset.mat'], 'topo_alert', 'topo_fatigue'); clear topo_alert; clear topo_fatigue;

        elseif band == 2
            save([fname_new, '_theta_dataset.mat'], 'topo_alert', 'topo_fatigue'); clear topo_alert; clear topo_fatigue;
    
        elseif band == 3
            save([fname_new, '_alpha_dataset.mat'], 'topo_alert', 'topo_fatigue'); clear topo_alert; clear topo_fatigue;
            
        elseif band == 4
            save([fname_new, '_beta_dataset.mat'], 'topo_alert', 'topo_fatigue'); clear topo_alert; clear topo_fatigue;
        else
            save([fname_new, '_gamma_dataset.mat'], 'topo_alert', 'topo_fatigue'); clear topo_alert; clear topo_fatigue;

        end
    end
end

%% Delta

clc;clear;
filepath = dir('*_delta*.mat');
filename = {filepath.name};

title = string(missing);

for sample = 1 : 67*67
    title(1) = 'label';
    title(sample+1) = strcat('pixel', num2str(sample-1));
end

writematrix(title, 'delta_dataset.csv');

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(topo_alert(:,1,1)),1);
    label_fatigue = ones(numel(topo_fatigue(:,1,1)),1);

    alert_delta_1D = reshape(topo_alert, [numel(topo_alert(:,1,1)), 67*67]);
    fatigue_delta_1D = reshape(topo_fatigue, [numel(topo_fatigue(:,1,1)), 67*67]);

    topo_alert_dataset = horzcat(label_alert, alert_delta_1D);
    topo_fatigue_dataset = horzcat(label_fatigue, fatigue_delta_1D);
    
    writematrix(topo_alert_dataset, 'delta_dataset.csv','WriteMode', 'append');
    writematrix(topo_fatigue_dataset, 'delta_dataset.csv', 'WriteMode', 'append');
end
%% Theta
clc;clear;
filepath = dir('*_theta*.mat');
filename = {filepath.name};

title = string(missing);

for sample = 1 : 67*67
    title(1) = 'label';
    title(sample+1) = strcat('pixel', num2str(sample-1));
end

writematrix(title, 'theta_dataset.csv');

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(topo_alert(:,1,1)),1);
    label_fatigue = ones(numel(topo_fatigue(:,1,1)),1);

    alert_theta_1D = reshape(topo_alert, [numel(topo_alert(:,1,1)), 67*67]);
    fatigue_theta_1D = reshape(topo_fatigue, [numel(topo_fatigue(:,1,1)), 67*67]);

    topo_alert_dataset = horzcat(label_alert, alert_theta_1D);
    topo_fatigue_dataset = horzcat(label_fatigue, fatigue_theta_1D);
    
    writematrix(topo_alert_dataset, 'theta_dataset.csv','WriteMode', 'append');
    writematrix(topo_fatigue_dataset, 'theta_dataset.csv', 'WriteMode', 'append');
end
%% Alpha
clc;clear;
filepath = dir('*_alpha*.mat');
filename = {filepath.name};

title = string(missing);

for sample = 1 : 67*67
    title(1) = 'label';
    title(sample+1) = strcat('pixel', num2str(sample-1));
end

writematrix(title, 'alpha_dataset.csv');

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(topo_alert(:,1,1)),1);
    label_fatigue = ones(numel(topo_fatigue(:,1,1)),1);

    alert_alpha_1D = reshape(topo_alert, [numel(topo_alert(:,1,1)), 67*67]);
    fatigue_alpha_1D = reshape(topo_fatigue, [numel(topo_fatigue(:,1,1)), 67*67]);

    topo_alert_dataset = horzcat(label_alert, alert_alpha_1D);
    topo_fatigue_dataset = horzcat(label_fatigue, fatigue_alpha_1D);
    
    writematrix(topo_alert_dataset, 'alpha_dataset.csv','WriteMode', 'append');
    writematrix(topo_fatigue_dataset, 'alpha_dataset.csv', 'WriteMode', 'append');
end
%% Beta
clc;clear;
filepath = dir('*_beta*.mat');
filename = {filepath.name};

title = string(missing);

for sample = 1 : 67*67
    title(1) = 'label';
    title(sample+1) = strcat('pixel', num2str(sample-1));
end

writematrix(title, 'beta_dataset.csv');

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(topo_alert(:,1,1)),1);
    label_fatigue = ones(numel(topo_fatigue(:,1,1)),1);

    alert_beta_1D = reshape(topo_alert, [numel(topo_alert(:,1,1)), 67*67]);
    fatigue_beta_1D = reshape(topo_fatigue, [numel(topo_fatigue(:,1,1)), 67*67]);

    topo_alert_dataset = horzcat(label_alert, alert_beta_1D);
    topo_fatigue_dataset = horzcat(label_fatigue, fatigue_beta_1D);
    
    writematrix(topo_alert_dataset, 'beta_dataset.csv','WriteMode', 'append');
    writematrix(topo_fatigue_dataset, 'beta_dataset.csv', 'WriteMode', 'append');
end
%% Gamma
clc;clear;
filepath = dir('*_gamma*.mat');
filename = {filepath.name};

title = string(missing);

for sample = 1 : 67*67
    title(1) = 'label';
    title(sample+1) = strcat('pixel', num2str(sample-1));
end

writematrix(title, 'gamma_dataset.csv');

for subject = 1:21
    load(filename{subject});
    
    label_alert = zeros(numel(topo_alert(:,1,1)),1);
    label_fatigue = ones(numel(topo_fatigue(:,1,1)),1);

    alert_gamma_1D = reshape(topo_alert, [numel(topo_alert(:,1,1)), 67*67]);
    fatigue_gamma_1D = reshape(topo_fatigue, [numel(topo_fatigue(:,1,1)), 67*67]);

    topo_alert_dataset = horzcat(label_alert, alert_gamma_1D);
    topo_fatigue_dataset = horzcat(label_fatigue, fatigue_gamma_1D);
    
    writematrix(topo_alert_dataset, 'gamma_dataset.csv','WriteMode', 'append');
    writematrix(topo_fatigue_dataset, 'gamma_dataset.csv', 'WriteMode', 'append');
end
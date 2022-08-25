%% Data load
clc; clear;

cd('/home/user/Documents/python/LSJ/LSJ_MATLAB/Memory_Network/01 jungchaeho/SourceData')

T1 = load('T1.mat').data;
T2 = load('T2.mat').data;
T3 = load('T3.mat').data;

F1 = load('F1.mat').data;
F2 = load('F2.mat').data;
F3 = load('F3.mat').data;

sfreq = 1000;
trial = 5;

%% Epoch

T1 = reshape(T1, size(T1,1), size(T1,2), sfreq, trial);
T2 = reshape(T2, size(T2,1), size(T2,2), sfreq, trial);
T3 = reshape(T3, size(T3,1), size(T3,2), sfreq, trial);

F1 = reshape(F1, size(F1,1), size(F1,2), sfreq, trial);
F2 = reshape(F2, size(F2,1), size(F2,2), sfreq, trial);
F3 = reshape(F3, size(F3,1), size(F3,2), sfreq, trial);

%% Labeling

Mem = [21,22,29:34,39,40,45:48,51,52,73,74];
Att = [9,10,23,24,75,76];
Vis = [7,8,11,12,15,16,49,50];
Mem_Att = [57,58,69,70,79,80,81,82];
Mem_Vis = [17,18,53,54,77,78];

Len = [length(Mem), length(Att), length(Vis), length(Mem_Att), length(Mem_Vis)];

T1_Mem = T1(:,Mem,:,:);
T1_Att = T1(:,Att,:,:);
T1_Vis = T1(:,Vis,:,:);
T1_Mem_Att = T1(:,Mem_Att,:,:);
T1_Mem_Vis = T1(:,Mem_Vis,:,:);

T2_Mem = T2(:,Mem,:,:);
T2_Att = T2(:,Att,:,:);
T2_Vis = T2(:,Vis,:,:);
T2_Mem_Att = T2(:,Mem_Att,:,:);
T2_Mem_Vis = T2(:,Mem_Vis,:,:);

T3_Mem = T3(:,Mem,:,:);
T3_Att = T3(:,Att,:,:);
T3_Vis = T3(:,Vis,:,:);
T3_Mem_Att = T3(:,Mem_Att,:,:);
T3_Mem_Vis = T3(:,Mem_Vis,:,:);

F1_Mem = F1(:,Mem,:,:);
F1_Att = F1(:,Att,:,:);
F1_Vis = F1(:,Vis,:,:);
F1_Mem_Att = F1(:,Mem_Att,:,:);
F1_Mem_Vis = F1(:,Mem_Vis,:,:);

F2_Mem = F2(:,Mem,:,:);
F2_Att = F2(:,Att,:,:);
F2_Vis = F2(:,Vis,:,:);
F2_Mem_Att = F2(:,Mem_Att,:,:);
F2_Mem_Vis = F2(:,Mem_Vis,:,:);

F3_Mem = F3(:,Mem,:,:);
F3_Att = F3(:,Att,:,:);
F3_Vis = F3(:,Vis,:,:);
F3_Mem_Att = F3(:,Mem_Att,:,:);
F3_Mem_Vis = F3(:,Mem_Vis,:,:);

clear T1 T2 T3 F1 F2 F3 Mem Att Vis Mem_Att Mem_Vis
%% Compute Coherence
% data                       : an input data matrix [Nsamples x 2 x trials]
% Memory region              : 18
% Attention region           : 6
% Visual region              : 8
% Memory, Attention region   : 8
% Memory, Visual region      : 6

% Memory - Attention
n = 0;
coh_1 = zeros(501, size(T3_Mem,2)*size(T3_Att,2), trial);
frq_1 = zeros(501, size(T3_Mem,2)*size(T3_Att,2), trial);
confidence_1 = zeros(1, size(T3_Mem,2)*size(T3_Att,2), trial);

for i = 1 : size(T3_Mem, 2)
   for j = 1 : size(T3_Att, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T3_Mem(:,i,:,:);
         coh_1_the_mean = T3_Att(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_1(:,n,k) = coh;
         frq_1(:,n,k) = frq;
         confidence_1(1,n,k) = confidence;
      end
   end
end

disp('*');

% Memory - Visual
n = 0;
coh_2 = zeros(501, size(T2_Mem,2)*size(T2_Vis,2), trial);
frq_2 = zeros(501, size(T2_Mem,2)*size(T2_Vis,2), trial);
confidence_2 = zeros(1, size(T2_Mem,2)*size(T2_Vis,2), trial);

for i = 1 : size(T2_Mem, 2)
   for j = 1 : size(T2_Vis, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Mem(:,i,:,:);
         coh_1_the_mean = T2_Vis(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_2(:,n,k) = coh;
         frq_2(:,n,k) = frq;
         confidence_2(1,n,k) = confidence;
      end
   end
end

disp('*');

% Memory - Memory, Attention
n = 0;
coh_3 = zeros(501, size(T2_Mem,2)*size(T2_Mem_Att,2), trial);
frq_3 = zeros(501, size(T2_Mem,2)*size(T2_Mem_Att,2), trial);
confidence_3 = zeros(1, size(T2_Mem,2)*size(T2_Mem_Att,2), trial);

for i = 1 : size(T2_Mem, 2)
   for j = 1 : size(T2_Mem_Att, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Mem(:,i,:,:);
         coh_1_the_mean = T2_Mem_Att(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_3(:,n,k) = coh;
         frq_3(:,n,k) = frq;
         confidence_3(1,n,k) = confidence;
      end
   end
end

disp('*');

% Memory - Memory, Visual
n = 0;
coh_4 = zeros(501, size(T2_Mem,2)*size(T2_Mem_Vis,2), trial);
frq_4 = zeros(501, size(T2_Mem,2)*size(T2_Mem_Vis,2), trial);
confidence_4 = zeros(1, size(T2_Mem,2)*size(T2_Mem_Vis,2), trial);

for i = 1 : size(T2_Mem, 2)
   for j = 1 : size(T2_Mem_Vis, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Mem(:,i,:,:);
         coh_1_the_mean = T2_Mem_Vis(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_4(:,n,k) = coh;
         frq_4(:,n,k) = frq;
         confidence_4(1,n,k) = confidence;
      end
   end
end

disp('*');

% Attention - Visual
n = 0;
coh_5 = zeros(501, size(T2_Att,2)*size(T2_Vis,2), trial);
frq_5 = zeros(501, size(T2_Att,2)*size(T2_Vis,2), trial);
confidence_5 = zeros(1, size(T2_Att,2)*size(T2_Vis,2), trial);

for i = 1 : size(T2_Att, 2)
   for j = 1 : size(T2_Vis, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Att(:,i,:,:);
         coh_1_the_mean = T2_Vis(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_5(:,n,k) = coh;
         frq_5(:,n,k) = frq;
         confidence_5(1,n,k) = confidence;
      end
   end
end

disp('*');

% Attention - Memory, Attention
n = 0;
coh_6 = zeros(501, size(T2_Att,2)*size(T2_Mem_Att,2), trial);
frq_6 = zeros(501, size(T2_Att,2)*size(T2_Mem_Att,2), trial);
confidence_6 = zeros(1, size(T2_Att,2)*size(T2_Mem_Att,2), trial);

for i = 1 : size(T2_Att, 2)
   for j = 1 : size(T2_Mem_Att, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Att(:,i,:,:);
         coh_1_the_mean = T2_Mem_Att(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_6(:,n,k) = coh;
         frq_6(:,n,k) = frq;
         confidence_6(1,n,k) = confidence;
      end
   end
end

disp('*');

% Attention - Memory, Visual
n = 0;
coh_7 = zeros(501, size(T2_Att,2)*size(T2_Mem_Vis,2), trial);
frq_7 = zeros(501, size(T2_Att,2)*size(T2_Mem_Vis,2), trial);
confidence_7 = zeros(1, size(T2_Att,2)*size(T2_Mem_Vis,2), trial);

for i = 1 : size(T2_Att, 2)
   for j = 1 : size(T2_Mem_Vis, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Att(:,i,:,:);
         coh_1_the_mean = T2_Mem_Vis(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_7(:,n,k) = coh;
         frq_7(:,n,k) = frq;
         confidence_7(1,n,k) = confidence;
      end
   end
end

disp('*');

% Visual - Memory, Attention
n = 0;
coh_8 = zeros(501, size(T2_Vis,2)*size(T2_Mem_Att,2), trial);
frq_8 = zeros(501, size(T2_Vis,2)*size(T2_Mem_Att,2), trial);
confidence_8 = zeros(1, size(T2_Vis,2)*size(T2_Mem_Att,2), trial);

for i = 1 : size(T2_Vis, 2)
   for j = 1 : size(T2_Mem_Att, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Vis(:,i,:,:);
         coh_1_the_mean = T2_Mem_Att(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_8(:,n,k) = coh;
         frq_8(:,n,k) = frq;
         confidence_8(1,n,k) = confidence;
      end
   end
end

disp('*');

% Visual - Memory, Visual
n = 0;
coh_9 = zeros(501, size(T2_Vis,2)*size(T2_Mem_Vis,2), trial);
frq_9 = zeros(501, size(T2_Vis,2)*size(T2_Mem_Vis,2), trial);
confidence_9 = zeros(1, size(T2_Vis,2)*size(T2_Mem_Vis,2), trial);

for i = 1 : size(T2_Vis, 2)
   for j = 1 : size(T2_Mem_Vis, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Vis(:,i,:,:);
         coh_1_the_mean = T2_Mem_Vis(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_9(:,n,k) = coh;
         frq_9(:,n,k) = frq;
         confidence_9(1,n,k) = confidence;
      end
   end
end

disp('*');

% Memory, Visual - Memory, Attention
n = 0;
coh_10 = zeros(501, size(T2_Mem_Vis,2)*size(T2_Mem_Att,2), trial);
frq_10 = zeros(501, size(T2_Mem_Vis,2)*size(T2_Mem_Att,2), trial);
confidence_10 = zeros(1, size(T2_Mem_Vis,2)*size(T2_Mem_Att,2), trial);

for i = 1 : size(T2_Mem_Vis, 2)
   for j = 1 : size(T2_Mem_Att, 2)
         n = n+1;
      for k = 1 : trial
         
         coh_1_del_mean = T2_Mem_Vis(:,i,:,:);
         coh_1_the_mean = T2_Mem_Att(:,j,:,:);
         data = cat(2, coh_1_del_mean, coh_1_the_mean);
         data = data(:,:,:,k);
         data = permute(data, [3,2,1]);
         [coh, frq, confidence] = cohere_blocks(data, sfreq);
         
         coh_10(:,n,k) = coh;
         frq_10(:,n,k) = frq;
         confidence_10(1,n,k) = confidence;
      end
   end
end

disp('*');

save('confidence.mat', 'confidence');

clear T1_Mem T1_Att T1_Vis T1_Mem_Att T1_Mem_Vis;
clear T2_Mem T2_Att T1_Vis T2_Mem_Att T2_Mem_Vis;
clear T3_Mem T3_Att T1_Vis T3_Mem_Att T3_Mem_Vis;
clear F1_Mem F1_Att F1_Vis F1_Mem_Att F1_Mem_Vis;
clear F2_Mem F2_Att F2_Vis F2_Mem_Att F2_Mem_Vis;
clear F3_Mem F3_Att F3_Vis F3_Mem_Att F3_Mem_Vis;
%% Coherence by Band
% Delta : 1~4 Hz
% Theta : 4~8 Hz
% Alpha : 8~13 Hz
% Beta  : 13~30 Hz
% Gamma : 30~50 Hz

f = 2:51;

coh_1 = coh_1(f,:,:);
coh_2 = coh_2(f,:,:);
coh_3 = coh_3(f,:,:);
coh_4 = coh_4(f,:,:);
coh_5 = coh_5(f,:,:);
coh_6 = coh_6(f,:,:);
coh_7 = coh_7(f,:,:);
coh_8 = coh_8(f,:,:);
coh_9 = coh_9(f,:,:);
coh_10 = coh_10(f,:,:);

Delta = 1:3;
Theta = 4:7;
Alpha = 8:12;
Beta  = 13:30;
Gamma = 31:50;

coh_1_del = coh_1(Delta,:,:);
coh_1_the = coh_1(Theta,:,:);
coh_1_alp = coh_1(Alpha,:,:);
coh_1_bet = coh_1(Beta,:,:);
coh_1_gam = coh_1(Gamma,:,:);

coh_2_del = coh_2(Delta,:,:);
coh_2_the = coh_2(Theta,:,:);
coh_2_alp = coh_2(Alpha,:,:);
coh_2_bet = coh_2(Beta,:,:);
coh_2_gam = coh_2(Gamma,:,:);

coh_3_del = coh_3(Delta,:,:);
coh_3_the = coh_3(Theta,:,:);
coh_3_alp = coh_3(Alpha,:,:);
coh_3_bet = coh_3(Beta,:,:);
coh_3_gam = coh_3(Gamma,:,:);

coh_4_del = coh_4(Delta,:,:);
coh_4_the = coh_4(Theta,:,:);
coh_4_alp = coh_4(Alpha,:,:);
coh_4_bet = coh_4(Beta,:,:);
coh_4_gam = coh_4(Gamma,:,:);

coh_5_del = coh_5(Delta,:,:);
coh_5_the = coh_5(Theta,:,:);
coh_5_alp = coh_5(Alpha,:,:);
coh_5_bet = coh_5(Beta,:,:);
coh_5_gam = coh_5(Gamma,:,:);

coh_6_del = coh_6(Delta,:,:);
coh_6_the = coh_6(Theta,:,:);
coh_6_alp = coh_6(Alpha,:,:);
coh_6_bet = coh_6(Beta,:,:);
coh_6_gam = coh_6(Gamma,:,:);

coh_7_del = coh_7(Delta,:,:);
coh_7_the = coh_7(Theta,:,:);
coh_7_alp = coh_7(Alpha,:,:);
coh_7_bet = coh_7(Beta,:,:);
coh_7_gam = coh_7(Gamma,:,:);

coh_8_del = coh_8(Delta,:,:);
coh_8_the = coh_8(Theta,:,:);
coh_8_alp = coh_8(Alpha,:,:);
coh_8_bet = coh_8(Beta,:,:);
coh_8_gam = coh_8(Gamma,:,:);

coh_9_del = coh_9(Delta,:,:);
coh_9_the = coh_9(Theta,:,:);
coh_9_alp = coh_9(Alpha,:,:);
coh_9_bet = coh_9(Beta,:,:);
coh_9_gam = coh_9(Gamma,:,:);

coh_10_del = coh_10(Delta,:,:);
coh_10_the = coh_10(Theta,:,:);
coh_10_alp = coh_10(Alpha,:,:);
coh_10_bet = coh_10(Beta,:,:);
coh_10_gam = coh_10(Gamma,:,:);

cd('/home/user/Documents/python/LSJ/LSJ_MATLAB/Memory_Network/01 jungchaeho/SourceData/True_Session/True_Session_2');

save('coh1_band.mat', 'coh_1_del', 'coh_1_the', 'coh_1_alp', 'coh_1_bet', 'coh_1_gam');
save('coh2_band.mat', 'coh_2_del', 'coh_2_the', 'coh_2_alp', 'coh_2_bet', 'coh_2_gam');
save('coh3_band.mat', 'coh_3_del', 'coh_3_the', 'coh_3_alp', 'coh_3_bet', 'coh_3_gam');
save('coh4_band.mat', 'coh_4_del', 'coh_4_the', 'coh_4_alp', 'coh_4_bet', 'coh_4_gam');
save('coh5_band.mat', 'coh_5_del', 'coh_5_the', 'coh_5_alp', 'coh_5_bet', 'coh_5_gam');
save('coh6_band.mat', 'coh_6_del', 'coh_6_the', 'coh_6_alp', 'coh_6_bet', 'coh_6_gam');
save('coh7_band.mat', 'coh_7_del', 'coh_7_the', 'coh_7_alp', 'coh_7_bet', 'coh_7_gam');
save('coh8_band.mat', 'coh_8_del', 'coh_8_the', 'coh_8_alp', 'coh_8_bet', 'coh_8_gam');
save('coh9_band.mat', 'coh_9_del', 'coh_9_the', 'coh_9_alp', 'coh_9_bet', 'coh_9_gam');
save('coh10_band.mat','coh_10_del','coh_10_the','coh_10_alp','coh_10_bet','coh_10_gam');
%% Memory - Attention
clc; clear;

load('coh1_band.mat');

figure(1), title('Memory - Attention');
coh_1_del_mean = squeeze(mean(coh_1_del,1));
coh_1_the_mean = squeeze(mean(coh_1_the,1));
coh_1_alp_mean = squeeze(mean(coh_1_alp,1));
coh_1_bet_mean = squeeze(mean(coh_1_bet,1));
coh_1_gam_mean = squeeze(mean(coh_1_gam,1));

subplot(5,5,1), plot(1:108, coh_1_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:108, coh_1_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:108, coh_1_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:108, coh_1_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:108, coh_1_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:108, coh_1_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:108, coh_1_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:108, coh_1_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:108, coh_1_the_mean(:,4)), title('Theta 4');
subplot(5,5,10), plot(1:108,coh_1_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:108, coh_1_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:108, coh_1_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:108, coh_1_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:108, coh_1_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:108,coh_1_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:108, coh_1_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:108, coh_1_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:108, coh_1_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:108, coh_1_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:108,coh_1_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:108, coh_1_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:108, coh_1_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:108, coh_1_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:108, coh_1_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:108,coh_1_gam_mean(:,5)), title('Gamma 5');

%% Memory - Visual

load('coh2_band.mat');

figure(2), title('Memory - Visual');
coh_2_del_mean = squeeze(mean(coh_2_del,1));
coh_2_the_mean = squeeze(mean(coh_2_the,1));
coh_2_alp_mean = squeeze(mean(coh_2_alp,1));
coh_2_bet_mean = squeeze(mean(coh_2_bet,1));
coh_2_gam_mean = squeeze(mean(coh_2_gam,1));

subplot(5,5,1), plot(1:size(coh_2_del_mean, 1), coh_2_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_2_del_mean, 1), coh_2_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_2_del_mean, 1), coh_2_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_2_del_mean, 1), coh_2_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_2_del_mean, 1), coh_2_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_2_the_mean, 1), coh_2_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_2_the_mean, 1), coh_2_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_2_the_mean, 1), coh_2_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_2_the_mean, 1), coh_2_the_mean(:,4)), title('Theta 4');
subplot(5,5,10), plot(1:size(coh_2_the_mean, 1),coh_2_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_2_alp_mean, 1), coh_2_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_2_alp_mean, 1), coh_2_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_2_alp_mean, 1), coh_2_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_2_alp_mean, 1), coh_2_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_2_alp_mean, 1), coh_2_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_2_bet_mean, 1), coh_2_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_2_bet_mean, 1), coh_2_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_2_bet_mean, 1), coh_2_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_2_bet_mean, 1), coh_2_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_2_bet_mean, 1), coh_2_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_2_gam_mean, 1), coh_2_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_2_gam_mean, 1), coh_2_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_2_gam_mean, 1), coh_2_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_2_gam_mean, 1), coh_2_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_2_gam_mean, 1), coh_2_gam_mean(:,5)), title('Gamma 5');

%% Memory - Memory, Attention

load('coh3_band.mat');

figure(3), title('Memory - Memory, Attention');
coh_3_del_mean = squeeze(mean(coh_3_del,1));
coh_3_the_mean = squeeze(mean(coh_3_the,1));
coh_3_alp_mean = squeeze(mean(coh_3_alp,1));
coh_3_bet_mean = squeeze(mean(coh_3_bet,1));
coh_3_gam_mean = squeeze(mean(coh_3_gam,1));

subplot(5,5,1), plot(1:size(coh_3_del_mean, 1), coh_3_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_3_del_mean, 1), coh_3_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_3_del_mean, 1), coh_3_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_3_del_mean, 1), coh_3_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_3_del_mean, 1), coh_3_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_3_the_mean, 1), coh_3_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_3_the_mean, 1), coh_3_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_3_the_mean, 1), coh_3_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_3_the_mean, 1), coh_3_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_3_the_mean, 1), coh_3_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_3_alp_mean, 1), coh_3_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_3_alp_mean, 1), coh_3_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_3_alp_mean, 1), coh_3_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_3_alp_mean, 1), coh_3_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_3_alp_mean, 1), coh_3_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_3_bet_mean, 1), coh_3_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_3_bet_mean, 1), coh_3_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_3_bet_mean, 1), coh_3_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_3_bet_mean, 1), coh_3_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_3_bet_mean, 1), coh_3_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_3_gam_mean, 1), coh_3_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_3_gam_mean, 1), coh_3_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_3_gam_mean, 1), coh_3_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_3_gam_mean, 1), coh_3_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_3_gam_mean, 1), coh_3_gam_mean(:,5)), title('Gamma 5');

%% Memory - Memory, Visual

load('coh4_band.mat');

figure(4), title('Memory - Memory, Visual');
coh_4_del_mean = squeeze(mean(coh_4_del,1));
coh_4_the_mean = squeeze(mean(coh_4_the,1));
coh_4_alp_mean = squeeze(mean(coh_4_alp,1));
coh_4_bet_mean = squeeze(mean(coh_4_bet,1));
coh_4_gam_mean = squeeze(mean(coh_4_gam,1));

subplot(5,5,1), plot(1:size(coh_4_del_mean, 1), coh_4_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_4_del_mean, 1), coh_4_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_4_del_mean, 1), coh_4_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_4_del_mean, 1), coh_4_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_4_del_mean, 1), coh_4_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_4_the_mean, 1), coh_4_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_4_the_mean, 1), coh_4_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_4_the_mean, 1), coh_4_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_4_the_mean, 1), coh_4_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_4_the_mean, 1), coh_4_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_4_alp_mean, 1), coh_4_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_4_alp_mean, 1), coh_4_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_4_alp_mean, 1), coh_4_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_4_alp_mean, 1), coh_4_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_4_alp_mean, 1), coh_4_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_4_bet_mean, 1), coh_4_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_4_bet_mean, 1), coh_4_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_4_bet_mean, 1), coh_4_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_4_bet_mean, 1), coh_4_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_4_bet_mean, 1), coh_4_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_4_gam_mean, 1), coh_4_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_4_gam_mean, 1), coh_4_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_4_gam_mean, 1), coh_4_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_4_gam_mean, 1), coh_4_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_4_gam_mean, 1), coh_4_gam_mean(:,5)), title('Gamma 5');

%% Attention - Visual

load('coh5_band.mat');

figure(5), title('Attention - Visual');
coh_5_del_mean = squeeze(mean(coh_5_del,1));
coh_5_the_mean = squeeze(mean(coh_5_the,1));
coh_5_alp_mean = squeeze(mean(coh_5_alp,1));
coh_5_bet_mean = squeeze(mean(coh_5_bet,1));
coh_5_gam_mean = squeeze(mean(coh_5_gam,1));

subplot(5,5,1), plot(1:size(coh_5_del_mean, 1), coh_5_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_5_del_mean, 1), coh_5_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_5_del_mean, 1), coh_5_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_5_del_mean, 1), coh_5_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_5_del_mean, 1), coh_5_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_5_the_mean, 1), coh_5_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_5_the_mean, 1), coh_5_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_5_the_mean, 1), coh_5_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_5_the_mean, 1), coh_5_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_5_the_mean, 1), coh_5_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_5_alp_mean, 1), coh_5_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_5_alp_mean, 1), coh_5_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_5_alp_mean, 1), coh_5_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_5_alp_mean, 1), coh_5_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_5_alp_mean, 1), coh_5_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_5_bet_mean, 1), coh_5_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_5_bet_mean, 1), coh_5_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_5_bet_mean, 1), coh_5_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_5_bet_mean, 1), coh_5_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_5_bet_mean, 1), coh_5_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_5_gam_mean, 1), coh_5_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_5_gam_mean, 1), coh_5_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_5_gam_mean, 1), coh_5_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_5_gam_mean, 1), coh_5_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_5_gam_mean, 1), coh_5_gam_mean(:,5)), title('Gamma 5');

%% Attention - Memory, Attention

load('coh6_band.mat');

figure(6), title('Attention - Memory, Attention');
coh_6_del_mean = squeeze(mean(coh_6_del,1));
coh_6_the_mean = squeeze(mean(coh_6_the,1));
coh_6_alp_mean = squeeze(mean(coh_6_alp,1));
coh_6_bet_mean = squeeze(mean(coh_6_bet,1));
coh_6_gam_mean = squeeze(mean(coh_6_gam,1));

subplot(5,5,1), plot(1:size(coh_6_del_mean, 1), coh_6_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_6_del_mean, 1), coh_6_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_6_del_mean, 1), coh_6_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_6_del_mean, 1), coh_6_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_6_del_mean, 1), coh_6_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_6_the_mean, 1), coh_6_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_6_the_mean, 1), coh_6_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_6_the_mean, 1), coh_6_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_6_the_mean, 1), coh_6_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_6_the_mean, 1), coh_6_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_6_alp_mean, 1), coh_6_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_6_alp_mean, 1), coh_6_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_6_alp_mean, 1), coh_6_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_6_alp_mean, 1), coh_6_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_6_alp_mean, 1), coh_6_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_6_bet_mean, 1), coh_6_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_6_bet_mean, 1), coh_6_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_6_bet_mean, 1), coh_6_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_6_bet_mean, 1), coh_6_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_6_bet_mean, 1), coh_6_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_6_gam_mean, 1), coh_6_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_6_gam_mean, 1), coh_6_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_6_gam_mean, 1), coh_6_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_6_gam_mean, 1), coh_6_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_6_gam_mean, 1), coh_6_gam_mean(:,5)), title('Gamma 5');

%% Attention - Memory, Visual

load('coh7_band.mat');

figure(7), title('Attention - Memory, Visual');
coh_7_del_mean = squeeze(mean(coh_7_del,1));
coh_7_the_mean = squeeze(mean(coh_7_the,1));
coh_7_alp_mean = squeeze(mean(coh_7_alp,1));
coh_7_bet_mean = squeeze(mean(coh_7_bet,1));
coh_7_gam_mean = squeeze(mean(coh_7_gam,1));

subplot(5,5,1), plot(1:size(coh_7_del_mean, 1), coh_7_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_7_del_mean, 1), coh_7_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_7_del_mean, 1), coh_7_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_7_del_mean, 1), coh_7_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_7_del_mean, 1), coh_7_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_7_the_mean, 1), coh_7_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_7_the_mean, 1), coh_7_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_7_the_mean, 1), coh_7_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_7_the_mean, 1), coh_7_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_7_the_mean, 1), coh_7_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_7_alp_mean, 1), coh_7_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_7_alp_mean, 1), coh_7_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_7_alp_mean, 1), coh_7_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_7_alp_mean, 1), coh_7_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_7_alp_mean, 1), coh_7_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_7_bet_mean, 1), coh_7_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_7_bet_mean, 1), coh_7_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_7_bet_mean, 1), coh_7_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_7_bet_mean, 1), coh_7_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_7_bet_mean, 1), coh_7_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_7_gam_mean, 1), coh_7_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_7_gam_mean, 1), coh_7_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_7_gam_mean, 1), coh_7_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_7_gam_mean, 1), coh_7_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_7_gam_mean, 1), coh_7_gam_mean(:,5)), title('Gamma 5');

%% Visual - Memory, Attention

load('coh8_band.mat');

figure(8), title('Visual - Memory, Attention');
coh_8_del_mean = squeeze(mean(coh_8_del,1));
coh_8_the_mean = squeeze(mean(coh_8_the,1));
coh_8_alp_mean = squeeze(mean(coh_8_alp,1));
coh_8_bet_mean = squeeze(mean(coh_8_bet,1));
coh_8_gam_mean = squeeze(mean(coh_8_gam,1));

subplot(5,5,1), plot(1:size(coh_8_del_mean, 1), coh_8_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_8_del_mean, 1), coh_8_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_8_del_mean, 1), coh_8_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_8_del_mean, 1), coh_8_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_8_del_mean, 1), coh_8_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_8_the_mean, 1), coh_8_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_8_the_mean, 1), coh_8_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_8_the_mean, 1), coh_8_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_8_the_mean, 1), coh_8_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_8_the_mean, 1), coh_8_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_8_alp_mean, 1), coh_8_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_8_alp_mean, 1), coh_8_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_8_alp_mean, 1), coh_8_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_8_alp_mean, 1), coh_8_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_8_alp_mean, 1), coh_8_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_8_bet_mean, 1), coh_8_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_8_bet_mean, 1), coh_8_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_8_bet_mean, 1), coh_8_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_8_bet_mean, 1), coh_8_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_8_bet_mean, 1), coh_8_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_8_gam_mean, 1), coh_8_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_8_gam_mean, 1), coh_8_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_8_gam_mean, 1), coh_8_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_8_gam_mean, 1), coh_8_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_8_gam_mean, 1), coh_8_gam_mean(:,5)), title('Gamma 5');

%% Visual - Memory, Visual

load('coh9_band.mat');

figure(9), title('Visual - Memory, Visual');
coh_9_del_mean = squeeze(mean(coh_9_del,1));
coh_9_the_mean = squeeze(mean(coh_9_the,1));
coh_9_alp_mean = squeeze(mean(coh_9_alp,1));
coh_9_bet_mean = squeeze(mean(coh_9_bet,1));
coh_9_gam_mean = squeeze(mean(coh_9_gam,1));

subplot(5,5,1), plot(1:size(coh_9_del_mean, 1), coh_9_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_9_del_mean, 1), coh_9_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_9_del_mean, 1), coh_9_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_9_del_mean, 1), coh_9_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_9_del_mean, 1), coh_9_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_9_the_mean, 1), coh_9_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_9_the_mean, 1), coh_9_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_9_the_mean, 1), coh_9_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_9_the_mean, 1), coh_9_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_9_the_mean, 1), coh_9_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_9_alp_mean, 1), coh_9_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_9_alp_mean, 1), coh_9_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_9_alp_mean, 1), coh_9_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_9_alp_mean, 1), coh_9_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_9_alp_mean, 1), coh_9_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_9_bet_mean, 1), coh_9_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_9_bet_mean, 1), coh_9_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_9_bet_mean, 1), coh_9_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_9_bet_mean, 1), coh_9_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_9_bet_mean, 1), coh_9_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_9_gam_mean, 1), coh_9_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_9_gam_mean, 1), coh_9_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_9_gam_mean, 1), coh_9_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_9_gam_mean, 1), coh_9_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_9_gam_mean, 1), coh_9_gam_mean(:,5)), title('Gamma 5');
%% Attention - Memory, Visual

load('coh10_band.mat');

figure(10), title('Attention - Memory, Visual');
coh_10_del_mean = squeeze(mean(coh_10_del,1));
coh_10_the_mean = squeeze(mean(coh_10_the,1));
coh_10_alp_mean = squeeze(mean(coh_10_alp,1));
coh_10_bet_mean = squeeze(mean(coh_10_bet,1));
coh_10_gam_mean = squeeze(mean(coh_10_gam,1));

subplot(5,5,1), plot(1:size(coh_10_del_mean, 1), coh_10_del_mean(:,1)), title('Delta 1');
subplot(5,5,2), plot(1:size(coh_10_del_mean, 1), coh_10_del_mean(:,2)), title('Delta 2');
subplot(5,5,3), plot(1:size(coh_10_del_mean, 1), coh_10_del_mean(:,3)), title('Delta 3');
subplot(5,5,4), plot(1:size(coh_10_del_mean, 1), coh_10_del_mean(:,4)), title('Delta 4');
subplot(5,5,5), plot(1:size(coh_10_del_mean, 1), coh_10_del_mean(:,5)), title('Delta 5');

subplot(5,5,6), plot(1:size(coh_10_the_mean, 1), coh_10_the_mean(:,1)), title('Theta 1');
subplot(5,5,7), plot(1:size(coh_10_the_mean, 1), coh_10_the_mean(:,2)), title('Theta 2');
subplot(5,5,8), plot(1:size(coh_10_the_mean, 1), coh_10_the_mean(:,3)), title('Theta 3');
subplot(5,5,9), plot(1:size(coh_10_the_mean, 1), coh_10_the_mean(:,4)), title('Theta 4');
subplot(5,5,10),plot(1:size(coh_10_the_mean, 1), coh_10_the_mean(:,5)), title('Theta 5');

subplot(5,5,11), plot(1:size(coh_10_alp_mean, 1), coh_10_alp_mean(:,1)), title('Alpha 1');
subplot(5,5,12), plot(1:size(coh_10_alp_mean, 1), coh_10_alp_mean(:,2)), title('Alpha 2');
subplot(5,5,13), plot(1:size(coh_10_alp_mean, 1), coh_10_alp_mean(:,3)), title('Alpha 3');
subplot(5,5,14), plot(1:size(coh_10_alp_mean, 1), coh_10_alp_mean(:,4)), title('Alpha 4');
subplot(5,5,15), plot(1:size(coh_10_alp_mean, 1), coh_10_alp_mean(:,5)), title('Alpha 5');

subplot(5,5,16), plot(1:size(coh_10_bet_mean, 1), coh_10_bet_mean(:,1)), title('Beta 1');
subplot(5,5,17), plot(1:size(coh_10_bet_mean, 1), coh_10_bet_mean(:,2)), title('Beta 2');
subplot(5,5,18), plot(1:size(coh_10_bet_mean, 1), coh_10_bet_mean(:,3)), title('Beta 3');
subplot(5,5,19), plot(1:size(coh_10_bet_mean, 1), coh_10_bet_mean(:,4)), title('Beta 4');
subplot(5,5,20), plot(1:size(coh_10_bet_mean, 1), coh_10_bet_mean(:,5)), title('Beta 5');

subplot(5,5,21), plot(1:size(coh_10_gam_mean, 1), coh_10_gam_mean(:,1)), title('Gamma 1');
subplot(5,5,22), plot(1:size(coh_10_gam_mean, 1), coh_10_gam_mean(:,2)), title('Gamma 2');
subplot(5,5,23), plot(1:size(coh_10_gam_mean, 1), coh_10_gam_mean(:,3)), title('Gamma 3');
subplot(5,5,24), plot(1:size(coh_10_gam_mean, 1), coh_10_gam_mean(:,4)), title('Gamma 4');
subplot(5,5,25), plot(1:size(coh_10_gam_mean, 1), coh_10_gam_mean(:,5)), title('Gamma 5');

save('coh1_band_mean.mat', 'coh_1_del_mean', 'coh_1_the_mean', 'coh_1_alp_mean', 'coh_1_bet_mean', 'coh_1_gam_mean');
save('coh2_band_mean.mat', 'coh_2_del_mean', 'coh_2_the_mean', 'coh_2_alp_mean', 'coh_2_bet_mean', 'coh_2_gam_mean');
save('coh3_band_mean.mat', 'coh_3_del_mean', 'coh_3_the_mean', 'coh_3_alp_mean', 'coh_3_bet_mean', 'coh_3_gam_mean');
save('coh4_band_mean.mat', 'coh_4_del_mean', 'coh_4_the_mean', 'coh_4_alp_mean', 'coh_4_bet_mean', 'coh_4_gam_mean');
save('coh5_band_mean.mat', 'coh_5_del_mean', 'coh_5_the_mean', 'coh_5_alp_mean', 'coh_5_bet_mean', 'coh_5_gam_mean');
save('coh6_band_mean.mat', 'coh_6_del_mean', 'coh_6_the_mean', 'coh_6_alp_mean', 'coh_6_bet_mean', 'coh_6_gam_mean');
save('coh7_band_mean.mat', 'coh_7_del_mean', 'coh_7_the_mean', 'coh_7_alp_mean', 'coh_7_bet_mean', 'coh_7_gam_mean');
save('coh8_band_mean.mat', 'coh_8_del_mean', 'coh_8_the_mean', 'coh_8_alp_mean', 'coh_8_bet_mean', 'coh_8_gam_mean');
save('coh9_band_mean.mat', 'coh_9_del_mean', 'coh_9_the_mean', 'coh_9_alp_mean', 'coh_9_bet_mean', 'coh_9_gam_mean');
save('coh10_band_mean.mat','coh_10_del_mean','coh_10_the_mean','coh_10_alp_mean','coh_10_bet_mean','coh_10_gam_mean');

%%
clc; clear;

load('coh1_band_mean.mat');
load('coh2_band_mean.mat');
load('coh3_band_mean.mat');
load('coh4_band_mean.mat');
load('coh5_band_mean.mat');
load('coh6_band_mean.mat');
load('coh7_band_mean.mat');
load('coh8_band_mean.mat');
load('coh9_band_mean.mat');
load('coh10_band_mean.mat');

Mem_label = {'Posterior cingulate cortex1 (left)', 'Posterior cingulate cortex2 (right)',
             'Posterior cingulate cortex1 (right)', 'Hippocampal area2 (left)',
             'Parahippocampal gyrus1 (left)', 'Hippocampal area2 (right)',
             'Parahippocampal gyrus1 (right)', 'Parahippocampal gyrus2 (left)',
             'Hippocampal area1 (left)', 'Parahippocampal gyrus2 (right)',
             'Hippocampal area1 (right)', 'Temporal pole (left)',
             'Retrosplenial cortex1 (left)', 'Temporal pole (right)',
             'Retrosplenial cortex1 (right)', 'Superior parietal sulcus (left)',
             'Posterior cingulate cortex2 (left)', 'Superior parietal sulcus (right)'};
             
Att_label = {'Secondary visual cortex (left)', 'Dorsal anterior cingulate cortex (right)',
             'Secondary visual cortex (right)','Supplementary motor area (left)',
             'Dorsal anterior cingulate cortex (left)', 'Supplementary motor area (right)'};
              
Vis_label = {'Primary visual cortex (left)', 'Inferior temporal gyrus (left)',
             'Primary visual cortex (right)','Inferior temporal gyrus (right)',
             'Cuneus (left)', 'Occipital-temporal cortex (left)',
             'Cuneus (right)','Occipital-temporal cortex (right)'};

Mem_Att_label = {'Intra-parietal sulcus (left)', 'Pre-supplementary motor area (left)',
                 'Intra-parietal sulcus (right)','Pre-supplementary motor area (right)',
                 'Medial prefrontal cortex (left)','Dorsolateral prefrontal cortex (left)',
                 'Medial prefrontal cortex (right)','Dorsolateral prefrontal cortex (right)'};
                 
Mem_Vis_label = {'Medial temporal gyrus (left)','Angular gyrus (right)',
                 'Medial temporal gyrus (right)','Superior parietal gyrus (left)',
                 'Angular gyrus (left)', 'Superior parietal gyrus (right)'};
                 
htm_coh_1_del = reshape(coh_1_del_mean, [18,6,5]);
htm_coh_1_the = reshape(coh_1_the_mean, [18,6,5]);
htm_coh_1_alp = reshape(coh_1_alp_mean, [18,6,5]);
htm_coh_1_bet = reshape(coh_1_bet_mean, [18,6,5]);
htm_coh_1_gam = reshape(coh_1_gam_mean, [18,6,5]);    

htm_coh_2_del = reshape(coh_2_del_mean, [18,8,5]);
htm_coh_2_the = reshape(coh_2_the_mean, [18,8,5]);
htm_coh_2_alp = reshape(coh_2_alp_mean, [18,8,5]);
htm_coh_2_bet = reshape(coh_2_bet_mean, [18,8,5]);
htm_coh_2_gam = reshape(coh_2_gam_mean, [18,8,5]);    

htm_coh_3_del = reshape(coh_3_del_mean, [18,8,5]);
htm_coh_3_the = reshape(coh_3_the_mean, [18,8,5]);
htm_coh_3_alp = reshape(coh_3_alp_mean, [18,8,5]);
htm_coh_3_bet = reshape(coh_3_bet_mean, [18,8,5]);
htm_coh_3_gam = reshape(coh_3_gam_mean, [18,8,5]);

htm_coh_4_del = reshape(coh_4_del_mean, [18,6,5]);
htm_coh_4_the = reshape(coh_4_the_mean, [18,6,5]);
htm_coh_4_alp = reshape(coh_4_alp_mean, [18,6,5]);
htm_coh_4_bet = reshape(coh_4_bet_mean, [18,6,5]);
htm_coh_4_gam = reshape(coh_4_gam_mean, [18,6,5]);    

htm_coh_5_del = reshape(coh_5_del_mean, [6,8,5]);
htm_coh_5_the = reshape(coh_5_the_mean, [6,8,5]);
htm_coh_5_alp = reshape(coh_5_alp_mean, [6,8,5]);
htm_coh_5_bet = reshape(coh_5_bet_mean, [6,8,5]);
htm_coh_5_gam = reshape(coh_5_gam_mean, [6,8,5]);

htm_coh_6_del = reshape(coh_6_del_mean, [6,8,5]);
htm_coh_6_the = reshape(coh_6_the_mean, [6,8,5]);
htm_coh_6_alp = reshape(coh_6_alp_mean, [6,8,5]);
htm_coh_6_bet = reshape(coh_6_bet_mean, [6,8,5]);
htm_coh_6_gam = reshape(coh_6_gam_mean, [6,8,5]);    

htm_coh_7_del = reshape(coh_7_del_mean, [6,6,5]);
htm_coh_7_the = reshape(coh_7_the_mean, [6,6,5]);
htm_coh_7_alp = reshape(coh_7_alp_mean, [6,6,5]);
htm_coh_7_bet = reshape(coh_7_bet_mean, [6,6,5]);
htm_coh_7_gam = reshape(coh_7_gam_mean, [6,6,5]);    

htm_coh_8_del = reshape(coh_8_del_mean, [8,8,5]);
htm_coh_8_the = reshape(coh_8_the_mean, [8,8,5]);
htm_coh_8_alp = reshape(coh_8_alp_mean, [8,8,5]);
htm_coh_8_bet = reshape(coh_8_bet_mean, [8,8,5]);
htm_coh_8_gam = reshape(coh_8_gam_mean, [8,8,5]);    

htm_coh_9_del = reshape(coh_9_del_mean, [8,6,5]);
htm_coh_9_the = reshape(coh_9_the_mean, [8,6,5]);
htm_coh_9_alp = reshape(coh_9_alp_mean, [8,6,5]);
htm_coh_9_bet = reshape(coh_9_bet_mean, [8,6,5]);
htm_coh_9_gam = reshape(coh_9_gam_mean, [8,6,5]);    

htm_coh_10_del = reshape(coh_10_del_mean, [8,6,5]);
htm_coh_10_the = reshape(coh_10_the_mean, [8,6,5]);
htm_coh_10_alp = reshape(coh_10_alp_mean, [8,6,5]);
htm_coh_10_bet = reshape(coh_10_bet_mean, [8,6,5]);
htm_coh_10_gam = reshape(coh_10_gam_mean, [8,6,5]);    

cd('/home/user/Documents/python/LSJ/LSJ_MATLAB/Memory_Network/01 jungchaeho/SourceData/True_Session/True_Session_3/Figure')

h_coh1_del_mean = heatmap(Att_label, Mem_label, mean(htm_coh_1_del,3)); h_coh1_del_mean.Title = 'Memory - Attention Network'; h_coh1_del_mean.XLabel = 'Attention region'; h_coh1_del_mean.YLabel = 'Memory region'; saveas(h_coh1_del_mean, 'h_coh1_del_mean', 'png');
h_coh1_the_mean = heatmap(Att_label, Mem_label, mean(htm_coh_1_the,3)); h_coh1_the_mean.Title = 'Memory - Attention Network'; h_coh1_the_mean.XLabel = 'Attention region'; h_coh1_the_mean.YLabel = 'Memory region'; saveas(h_coh1_the_mean, 'h_coh1_the_mean', 'png');
h_coh1_alp_mean = heatmap(Att_label, Mem_label, mean(htm_coh_1_alp,3)); h_coh1_alp_mean.Title = 'Memory - Attention Network'; h_coh1_alp_mean.XLabel = 'Attention region'; h_coh1_alp_mean.YLabel = 'Memory region'; saveas(h_coh1_alp_mean, 'h_coh1_alp_mean', 'png');
h_coh1_bet_mean = heatmap(Att_label, Mem_label, mean(htm_coh_1_bet,3)); h_coh1_bet_mean.Title = 'Memory - Attention Network'; h_coh1_bet_mean.XLabel = 'Attention region'; h_coh1_bet_mean.YLabel = 'Memory region'; saveas(h_coh1_bet_mean, 'h_coh1_bet_mean', 'png');
h_coh1_gam_mean = heatmap(Att_label, Mem_label, mean(htm_coh_1_gam,3)); h_coh1_gam_mean.Title = 'Memory - Attention Network'; h_coh1_gam_mean.XLabel = 'Attention region'; h_coh1_gam_mean.YLabel = 'Memory region'; saveas(h_coh1_gam_mean, 'h_coh1_gam_mean', 'png');

h_coh2_del_mean = heatmap(Vis_label, Mem_label, mean(htm_coh_2_del,3)); h_coh2_del_mean.Title = 'Memory - Visual Network'; h_coh2_del_mean.XLabel = 'Visual region'; h_coh2_del_mean.YLabel = 'Memory region'; saveas(h_coh2_del_mean, 'h_coh2_del_mean', 'png');
h_coh2_the_mean = heatmap(Vis_label, Mem_label, mean(htm_coh_2_the,3)); h_coh2_the_mean.Title = 'Memory - Visual Network'; h_coh2_the_mean.XLabel = 'Visual region'; h_coh2_the_mean.YLabel = 'Memory region'; saveas(h_coh2_the_mean, 'h_coh2_the_mean', 'png');
h_coh2_alp_mean = heatmap(Vis_label, Mem_label, mean(htm_coh_2_alp,3)); h_coh2_alp_mean.Title = 'Memory - Visual Network'; h_coh2_alp_mean.XLabel = 'Visual region'; h_coh2_alp_mean.YLabel = 'Memory region'; saveas(h_coh2_alp_mean, 'h_coh2_alp_mean', 'png');
h_coh2_bet_mean = heatmap(Vis_label, Mem_label, mean(htm_coh_2_bet,3)); h_coh2_bet_mean.Title = 'Memory - Visual Network'; h_coh2_bet_mean.XLabel = 'Visual region'; h_coh2_bet_mean.YLabel = 'Memory region'; saveas(h_coh2_bet_mean, 'h_coh2_bet_mean', 'png');
h_coh2_gam_mean = heatmap(Vis_label, Mem_label, mean(htm_coh_2_gam,3)); h_coh2_gam_mean.Title = 'Memory - Visual Network'; h_coh2_gam_mean.XLabel = 'Visual region'; h_coh2_gam_mean.YLabel = 'Memory region'; saveas(h_coh2_gam_mean, 'h_coh2_gam_mean', 'png');

h_coh3_del_mean = heatmap(Mem_Att_label, Mem_label, mean(htm_coh_3_del,3)); h_coh3_del_mean.Title = 'Memory - Memory, Attention Network'; h_coh3_del_mean.XLabel = 'Memory, Attention region'; h_coh3_del_mean.YLabel = 'Memory region'; saveas(h_coh3_del_mean, 'h_coh3_del_mean', 'png');
h_coh3_the_mean = heatmap(Mem_Att_label, Mem_label, mean(htm_coh_3_the,3)); h_coh3_the_mean.Title = 'Memory - Memory, Attention Network'; h_coh3_the_mean.XLabel = 'Memory, Attention region'; h_coh3_the_mean.YLabel = 'Memory region'; saveas(h_coh3_the_mean, 'h_coh3_the_mean', 'png');
h_coh3_alp_mean = heatmap(Mem_Att_label, Mem_label, mean(htm_coh_3_alp,3)); h_coh3_alp_mean.Title = 'Memory - Memory, Attention Network'; h_coh3_alp_mean.XLabel = 'Memory, Attention region'; h_coh3_alp_mean.YLabel = 'Memory region'; saveas(h_coh3_alp_mean, 'h_coh3_alp_mean', 'png');
h_coh3_bet_mean = heatmap(Mem_Att_label, Mem_label, mean(htm_coh_3_bet,3)); h_coh3_bet_mean.Title = 'Memory - Memory, Attention Network'; h_coh3_bet_mean.XLabel = 'Memory, Attention region'; h_coh3_bet_mean.YLabel = 'Memory region'; saveas(h_coh3_bet_mean, 'h_coh3_bet_mean', 'png');
h_coh3_gam_mean = heatmap(Mem_Att_label, Mem_label, mean(htm_coh_3_gam,3)); h_coh3_gam_mean.Title = 'Memory - Memory, Attention Network'; h_coh3_gam_mean.XLabel = 'Memory, Attention region'; h_coh3_gam_mean.YLabel = 'Memory region'; saveas(h_coh3_gam_mean, 'h_coh3_gam_mean', 'png');

h_coh4_del_mean = heatmap(Mem_Vis_label, Mem_label, mean(htm_coh_4_del,3)); h_coh4_del_mean.Title = 'Memory - Memory, Visual Network'; h_coh4_del_mean.XLabel = 'Memory, Visual region'; h_coh4_del_mean.YLabel = 'Memory region'; saveas(h_coh4_del_mean, 'h_coh4_del_mean', 'png');
h_coh4_the_mean = heatmap(Mem_Vis_label, Mem_label, mean(htm_coh_4_the,3)); h_coh4_the_mean.Title = 'Memory - Memory, Visual Network'; h_coh4_the_mean.XLabel = 'Memory, Visual region'; h_coh4_the_mean.YLabel = 'Memory region'; saveas(h_coh4_the_mean, 'h_coh4_the_mean', 'png');
h_coh4_alp_mean = heatmap(Mem_Vis_label, Mem_label, mean(htm_coh_4_alp,3)); h_coh4_alp_mean.Title = 'Memory - Memory, Visual Network'; h_coh4_alp_mean.XLabel = 'Memory, Visual region'; h_coh4_alp_mean.YLabel = 'Memory region'; saveas(h_coh4_alp_mean, 'h_coh4_alp_mean', 'png');
h_coh4_bet_mean = heatmap(Mem_Vis_label, Mem_label, mean(htm_coh_4_bet,3)); h_coh4_bet_mean.Title = 'Memory - Memory, Visual Network'; h_coh4_bet_mean.XLabel = 'Memory, Visual region'; h_coh4_bet_mean.YLabel = 'Memory region'; saveas(h_coh4_bet_mean, 'h_coh4_bet_mean', 'png');
h_coh4_gam_mean = heatmap(Mem_Vis_label, Mem_label, mean(htm_coh_4_gam,3)); h_coh4_gam_mean.Title = 'Memory - Memory, Visual Network'; h_coh4_gam_mean.XLabel = 'Memory, Visual region'; h_coh4_gam_mean.YLabel = 'Memory region'; saveas(h_coh4_gam_mean, 'h_coh4_gam_mean', 'png');

h_coh5_del_mean = heatmap(Vis_label, Att_label, mean(htm_coh_5_del,3)); h_coh5_del_mean.Title = 'Attention - Visual Network'; h_coh5_del_mean.XLabel = 'Visual region'; h_coh5_del_mean.YLabel = 'Attention region'; saveas(h_coh5_del_mean, 'h_coh5_del_mean', 'png');
h_coh5_the_mean = heatmap(Vis_label, Att_label, mean(htm_coh_5_the,3)); h_coh5_the_mean.Title = 'Attention - Visual Network'; h_coh5_the_mean.XLabel = 'Visual region'; h_coh5_the_mean.YLabel = 'Attention region'; saveas(h_coh5_the_mean, 'h_coh5_the_mean', 'png');
h_coh5_alp_mean = heatmap(Vis_label, Att_label, mean(htm_coh_5_alp,3)); h_coh5_alp_mean.Title = 'Attention - Visual Network'; h_coh5_alp_mean.XLabel = 'Visual region'; h_coh5_alp_mean.YLabel = 'Attention region'; saveas(h_coh5_alp_mean, 'h_coh5_alp_mean', 'png');
h_coh5_bet_mean = heatmap(Vis_label, Att_label, mean(htm_coh_5_bet,3)); h_coh5_bet_mean.Title = 'Attention - Visual Network'; h_coh5_bet_mean.XLabel = 'Visual region'; h_coh5_bet_mean.YLabel = 'Attention region'; saveas(h_coh5_bet_mean, 'h_coh5_bet_mean', 'png');
h_coh5_gam_mean = heatmap(Vis_label, Att_label, mean(htm_coh_5_gam,3)); h_coh5_gam_mean.Title = 'Attention - Visual Network'; h_coh5_gam_mean.XLabel = 'Visual region'; h_coh5_gam_mean.YLabel = 'Attention region'; saveas(h_coh5_gam_mean, 'h_coh5_gam_mean', 'png');

h_coh6_del_mean = heatmap(Mem_Att_label, Att_label, mean(htm_coh_6_del,3)); h_coh6_del_mean.Title = 'Attention - Memory, Attention Network'; h_coh6_del_mean.XLabel = 'Memory, Attention region'; h_coh6_del_mean.YLabel = 'Attention region'; saveas(h_coh6_del_mean, 'h_coh6_del_mean', 'png');
h_coh6_the_mean = heatmap(Mem_Att_label, Att_label, mean(htm_coh_6_the,3)); h_coh6_the_mean.Title = 'Attention - Memory, Attention Network'; h_coh6_the_mean.XLabel = 'Memory, Attention region'; h_coh6_the_mean.YLabel = 'Attention region'; saveas(h_coh6_the_mean, 'h_coh6_the_mean', 'png');
h_coh6_alp_mean = heatmap(Mem_Att_label, Att_label, mean(htm_coh_6_alp,3)); h_coh6_alp_mean.Title = 'Attention - Memory, Attention Network'; h_coh6_alp_mean.XLabel = 'Memory, Attention region'; h_coh6_alp_mean.YLabel = 'Attention region'; saveas(h_coh6_alp_mean, 'h_coh6_alp_mean', 'png');
h_coh6_bet_mean = heatmap(Mem_Att_label, Att_label, mean(htm_coh_6_bet,3)); h_coh6_bet_mean.Title = 'Attention - Memory, Attention Network'; h_coh6_bet_mean.XLabel = 'Memory, Attention region'; h_coh6_bet_mean.YLabel = 'Attention region'; saveas(h_coh6_bet_mean, 'h_coh6_bet_mean', 'png');
h_coh6_gam_mean = heatmap(Mem_Att_label, Att_label, mean(htm_coh_6_gam,3)); h_coh6_gam_mean.Title = 'Attention - Memory, Attention Network'; h_coh6_gam_mean.XLabel = 'Memory, Attention region'; h_coh6_gam_mean.YLabel = 'Attention region'; saveas(h_coh6_gam_mean, 'h_coh6_gam_mean', 'png');

h_coh7_del_mean = heatmap(Mem_Vis_label, Att_label, mean(htm_coh_7_del,3)); h_coh7_del_mean.Title = 'Attention - Memory, Visual Network'; h_coh7_del_mean.XLabel = 'Memory, Visual region'; h_coh7_del_mean.YLabel = 'Attention region'; saveas(h_coh7_del_mean, 'h_coh7_del_mean', 'png');
h_coh7_the_mean = heatmap(Mem_Vis_label, Att_label, mean(htm_coh_7_the,3)); h_coh7_the_mean.Title = 'Attention - Memory, Visual Network'; h_coh7_the_mean.XLabel = 'Memory, Visual region'; h_coh7_the_mean.YLabel = 'Attention region'; saveas(h_coh7_the_mean, 'h_coh7_the_mean', 'png');
h_coh7_alp_mean = heatmap(Mem_Vis_label, Att_label, mean(htm_coh_7_alp,3)); h_coh7_alp_mean.Title = 'Attention - Memory, Visual Network'; h_coh7_alp_mean.XLabel = 'Memory, Visual region'; h_coh7_alp_mean.YLabel = 'Attention region'; saveas(h_coh7_alp_mean, 'h_coh7_alp_mean', 'png');
h_coh7_bet_mean = heatmap(Mem_Vis_label, Att_label, mean(htm_coh_7_bet,3)); h_coh7_bet_mean.Title = 'Attention - Memory, Visual Network'; h_coh7_bet_mean.XLabel = 'Memory, Visual region'; h_coh7_bet_mean.YLabel = 'Attention region'; saveas(h_coh7_bet_mean, 'h_coh7_bet_mean', 'png');
h_coh7_gam_mean = heatmap(Mem_Vis_label, Att_label, mean(htm_coh_7_gam,3)); h_coh7_gam_mean.Title = 'Attention - Memory, Visual Network'; h_coh7_gam_mean.XLabel = 'Memory, Visual region'; h_coh7_gam_mean.YLabel = 'Attention region'; saveas(h_coh7_gam_mean, 'h_coh7_gam_mean', 'png');

h_coh8_del_mean = heatmap(Mem_Att_label, Vis_label, mean(htm_coh_8_del,3)); h_coh8_del_mean.Title = 'Visual - Memory, Attention Network'; h_coh8_del_mean.XLabel = 'Memory, Attention region'; h_coh8_del_mean.YLabel = 'Visual region'; saveas(h_coh8_del_mean, 'h_coh8_del_mean', 'png');
h_coh8_the_mean = heatmap(Mem_Att_label, Vis_label, mean(htm_coh_8_the,3)); h_coh8_the_mean.Title = 'Visual - Memory, Attention Network'; h_coh8_the_mean.XLabel = 'Memory, Attention region'; h_coh8_the_mean.YLabel = 'Visual region'; saveas(h_coh8_the_mean, 'h_coh8_the_mean', 'png');
h_coh8_alp_mean = heatmap(Mem_Att_label, Vis_label, mean(htm_coh_8_alp,3)); h_coh8_alp_mean.Title = 'Visual - Memory, Attention Network'; h_coh8_alp_mean.XLabel = 'Memory, Attention region'; h_coh8_alp_mean.YLabel = 'Visual region'; saveas(h_coh8_alp_mean, 'h_coh8_alp_mean', 'png');
h_coh8_bet_mean = heatmap(Mem_Att_label, Vis_label, mean(htm_coh_8_bet,3)); h_coh8_bet_mean.Title = 'Visual - Memory, Attention Network'; h_coh8_bet_mean.XLabel = 'Memory, Attention region'; h_coh8_bet_mean.YLabel = 'Visual region'; saveas(h_coh8_bet_mean, 'h_coh8_bet_mean', 'png');
h_coh8_gam_mean = heatmap(Mem_Att_label, Vis_label, mean(htm_coh_8_gam,3)); h_coh8_gam_mean.Title = 'Visual - Memory, Attention Network'; h_coh8_gam_mean.XLabel = 'Memory, Attention region'; h_coh8_gam_mean.YLabel = 'Visual region'; saveas(h_coh8_gam_mean, 'h_coh8_gam_mean', 'png');

h_coh9_del_mean = heatmap(Mem_Vis_label, Vis_label, mean(htm_coh_9_del,3)); h_coh9_del_mean.Title = 'Visual - Memory, Visual Network'; h_coh9_del_mean.XLabel = 'Memory, Visual region'; h_coh9_del_mean.YLabel = 'Visual region'; saveas(h_coh9_del_mean, 'h_coh9_del_mean', 'png');
h_coh9_the_mean = heatmap(Mem_Vis_label, Vis_label, mean(htm_coh_9_the,3)); h_coh9_the_mean.Title = 'Visual - Memory, Visual Network'; h_coh9_the_mean.XLabel = 'Memory, Visual region'; h_coh9_the_mean.YLabel = 'Visual region'; saveas(h_coh9_the_mean, 'h_coh9_the_mean', 'png');
h_coh9_alp_mean = heatmap(Mem_Vis_label, Vis_label, mean(htm_coh_9_alp,3)); h_coh9_alp_mean.Title = 'Visual - Memory, Visual Network'; h_coh9_alp_mean.XLabel = 'Memory, Visual region'; h_coh9_alp_mean.YLabel = 'Visual region'; saveas(h_coh9_alp_mean, 'h_coh9_alp_mean', 'png');
h_coh9_bet_mean = heatmap(Mem_Vis_label, Vis_label, mean(htm_coh_9_bet,3)); h_coh9_bet_mean.Title = 'Visual - Memory, Visual Network'; h_coh9_bet_mean.XLabel = 'Memory, Visual region'; h_coh9_bet_mean.YLabel = 'Visual region'; saveas(h_coh9_bet_mean, 'h_coh9_bet_mean', 'png');
h_coh9_gam_mean = heatmap(Mem_Vis_label, Vis_label, mean(htm_coh_9_gam,3)); h_coh9_gam_mean.Title = 'Visual - Memory, Visual Network'; h_coh9_gam_mean.XLabel = 'Memory, Visual region'; h_coh9_gam_mean.YLabel = 'Visual region'; saveas(h_coh9_gam_mean, 'h_coh9_gam_mean', 'png');

h_coh10_del_mean = heatmap(Mem_Vis_label, Mem_Att_label, mean(htm_coh_10_del,3)); h_coh10_del_mean.Title = 'Memory, Visual - Memory, Attention Network'; h_coh10_del_mean.XLabel = 'Memory, Visual region'; h_coh10_del_mean.YLabel = 'Memory, Attention region'; saveas(h_coh10_del_mean, 'h_coh10_del_mean', 'png');
h_coh10_the_mean = heatmap(Mem_Vis_label, Mem_Att_label, mean(htm_coh_10_the,3)); h_coh10_the_mean.Title = 'Memory, Visual - Memory, Attention Network'; h_coh10_the_mean.XLabel = 'Memory, Visual region'; h_coh10_the_mean.YLabel = 'Memory, Attention region'; saveas(h_coh10_the_mean, 'h_coh10_the_mean', 'png');
h_coh10_alp_mean = heatmap(Mem_Vis_label, Mem_Att_label, mean(htm_coh_10_alp,3)); h_coh10_alp_mean.Title = 'Memory, Visual - Memory, Attention Network'; h_coh10_alp_mean.XLabel = 'Memory, Visual region'; h_coh10_alp_mean.YLabel = 'Memory, Attention region'; saveas(h_coh10_alp_mean, 'h_coh10_alp_mean', 'png');
h_coh10_bet_mean = heatmap(Mem_Vis_label, Mem_Att_label, mean(htm_coh_10_bet,3)); h_coh10_bet_mean.Title = 'Memory, Visual - Memory, Attention Network'; h_coh10_bet_mean.XLabel = 'Memory, Visual region'; h_coh10_bet_mean.YLabel = 'Memory, Attention region'; saveas(h_coh10_bet_mean, 'h_coh10_bet_mean', 'png');
h_coh10_gam_mean = heatmap(Mem_Vis_label, Mem_Att_label, mean(htm_coh_10_gam,3)); h_coh10_gam_mean.Title = 'Memory, Visual - Memory, Attention Network'; h_coh10_gam_mean.XLabel = 'Memory, Visual region'; h_coh10_gam_mean.YLabel = 'Memory, Attention region'; saveas(h_coh10_gam_mean, 'h_coh10_gam_mean', 'png');

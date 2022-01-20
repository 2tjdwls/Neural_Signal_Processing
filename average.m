             %%%% channel number %%%%
%%%           1 = Fp1, 2 = Fpz, 3 = Fp2
%%%          10 = F3, 12 = Fz,  14 = F4,
%%% 26 = T7, 28 = C3, 30 = Cz, 32 = C4, 34 = T8,
%%%          46 = P3, 48 = Pz, 50 = P4,
%%%          57 = O1, 58 = Oz, 59= O2,

function y=average(data);
    %단순히 프로그램을 누적으로 더하였다.
    %size함수를 이용해 어떤 길이의 함수가 들어와도 전부 범위로 더할 수 있게 하였다.
    ea1=750; ea4=size(data,1)/ea1;
    A5=data; A7=0;
    for i=1:ea4
        A6=A5((ea1*(i-1))+1:ea1*i,:);
        A7=A6+A7;
    end
    A8=A7/ea4;
    y=A8;
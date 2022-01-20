function y=baseline(data);
    %각각의 베이스라인을 0~100ms사이의 평균값을 이용해 baseline을 선택하였다.
    ea1=750;
    
    for i=1:size(data,1)/ea1
        for j=1:1;
             D=sum(data(ea1*(i-1)+200:ea1*(i-1)+250,j))/50;
             V1(i,j)=D;
        end
    end
    
    for i=1:size(data,1)/ea1
        for j=1:1;
          A4((ea1*(i-1))+1:ea1*i,j)=data((ea1*(i-1))+1:ea1*i,j)-V1(i,j);
        end
    end
    
    
    y=A4;
    
    
    
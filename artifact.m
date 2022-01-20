function y=artifact(data);
    ea1=750;
    A1=zeros(size(data,1)/ea1,1);
    
    for i=1:size(data,1)/ea1
     for Y1=((i-1)*ea1+1):(i*ea1)
         if (data(Y1,:) < -50) | (data(Y1,:) > 70)
            A1(i,:)=i;
         end
        end
    end
    
    %조건에 맞는 범위를 찾아 그 segment를 삭제하였다.
    B1=A1; k=1; 
    for i=1:size(data,1)/ea1
     if   B1(i,:)==0
        C1(k,:)=i;
        k=k+1;
        else
        k=k+0;
      end
    end
    
    for i=1:max(length(C1))
        data2((ea1*(i-1))+1:ea1*i,:) = data(ea1*(C1(i)-1)+1:ea1*C1(i),:);
    end
    
    y=data2;
    end
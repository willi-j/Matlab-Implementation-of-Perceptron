function perceptron(table)

clc
%============================================
% Load in sonar data 
%===========================================
%table = table(randperm(size(table,1)),:);
x=table(:,1:60);
x = table2array(x);
y = table(:,61);
y = table2array(y);
z = zeros(length(y),1);
counter=0;
for i=1:length(y)
    if(y(i)=="M")
        z(i) = 1;
        
    elseif(y(i)=="R")
         z(i) = 0;
    end
        

end
%disp(z);
%disp(counter);
%disp(z);
%=====================================
% run pereptron
%=====================================
%[w,b,iterations,Error] = perceptrontrain(x(1:190,:),z(1:190,:));
%e=test(x(191:205,:),z(191:205,:),w,b);
%disp(['Test_Errors=' num2str(e) '     Test Data Size= ' num2str(15)])

[w,b,iterations,Error] = perceptrontrainbfgs(x,z);
%disp(w(1:60,1));
%disp(b);
disp(iterations);
%disp(Error);

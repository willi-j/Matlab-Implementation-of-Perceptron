function e=PerecptronTst(x,y,w,b)

tic
[l,p]=size(x);
new = ones(l,1);
%disp(new);
x = [x,new];
e=0; % number of test errors
for i=1:l          
    xx=x(i,:); % take one row
    ey=xx*w; % apply the classification rule
    if ey>=b
       ey=1;
    else
       ey=0;
    end
    if y(i)~=ey
       e=e+1;
    end
end
toc
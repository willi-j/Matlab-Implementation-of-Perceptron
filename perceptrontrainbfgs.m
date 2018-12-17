function [w,b,iterations,Error]=perceptrontrainbfgs(x,y)
%timerr
tic
%disp(x);
%disp(x(1,:));
[l,p]=size(x);
w=rand(p,1); % initialize weights
b=-.5;          % initialize bias
w = [w;b]; % initialize a misclassification indicator
new = ones(l,1);
%disp(new);
x = [x,new];
iterations=0;       % number of iterations%
%n=0.0012; 
n = .002;
%n=0.1; % learning rate
Error = [];
B = eye(61);
norm1=10e4; ftol=10e-4;

%disp(x);
%disp(w);

while (norm1>ftol) %repeat until no error
       w1 = w;
      
       err =0;
       diff = 0;
      % disp(norm1);
      bigg = 0;
      S = [];
      bigg1 = 0;
      S1 = [];
       for i=1:l  % a pass through x           
           xx=x(i,:);
           %disp(size(xx));
           %disp(size(w));
           %disp(xx*w);
          
           diff= xx*w; % estimated y
           %disp(size(diff));
           
           if diff>=0
              ey=1;
           else
              ey=0;
           end
           %disp(size(y));
          if y(i)~=ey
                Sati=y(i)-ey; % error difference
              %disp(Sati);
              grad = (Sati)*x(i,:);
              %disp(size(bigg));
              %disp(size(grad));
              %disp(size(w));
             
              bigg = bigg+ grad';
             % disp(bigg);
              %disp(err);
              err=err +abs(diff);
              %disp(size(abs(diff)));
             %disp(err);
          end
       end
        w2 =w + n*bigg;
        %disp(size(S));
        %disp(size(bigg));
       %disp(bigg);
      % disp(B);
       p = B\(bigg);
      % disp(size(B));
     %  disp(size(p));
       s = n*p;
       for i=1:l  % a pass through x           
           xx=x(i,:);
           %disp(size(xx));
           %disp(size(w));
           %disp(xx*w);
          
           diff= xx*(w2); % estimated y
           %disp(size(diff));
           
           if diff>=0
              ey=1;
           else
              ey=0;
           end
           %disp(size(y));
          if y(i)~=ey
                Sati=y(i)-ey; % error difference
              %disp(Sati);
              grad = (Sati)*x(i,:);
              %disp(grad);
              %disp(size(grad));
              %disp(size(w));
              bigg1 = bigg1+ grad';
             % disp(bigg);
              %disp(err);
             
             %disp(err);
          end
       end
      % disp(size(S1));
       %disp(size(p));
       y1 = bigg1-bigg;
        
        if(y1'*s > 0)
            Bn = B - (1/(s'*B*s))*B*s*s'*B + (1/(y1'*s))*(y1*y1');
            B= Bn;
        end
        
       %disp(size(err));
       if err~=0
         Error = [Error, err];
       end
       
       iterations=iterations+1; 
       if iterations==300
          disp("too many iterations");
          break;
       end
       if(iterations>1 && abs(err/Error(iterations-1))>1-ftol && abs(err/Error(iterations-1))<1+ftol)
           disp( err/Error(iterations-1));
           fprintf("error function does not decrease enough to justify keeping on");
           break;
       end
      % disp( size(w));
       %disp(size(s));
    
         w = w+n*p;
     norm1=norm(w1-w); 
     %disp(norm1);
end
b=w(61,1);
disp(['     Training data Size=' num2str(l)    '      iterations=' num2str(iterations)  '      w(1)=' num2str(w(1))  ]);
plot(Error);
toc
end

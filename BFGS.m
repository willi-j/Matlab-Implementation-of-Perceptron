function [] = BFGS(maxit,ftol)

     H = diag([5,1,0.8,0.5]);
     x = [-1 0 1 1]';
     c = [2 -1 2 -1]';
     B = eye(4);
     g =@(x)  H*x + c;
     f =@(x) c'*x + 0.5*x'*H*x;
     Q=@(alpha,p,x)  alpha*g(x)'*p + 0.5*alpha^2*p'*B*p;
     tol = 0.001;
     
     for i = 0: maxit
          
         
         p = B \ -g(x);
         alpha = 1;
         a = f(x + alpha*p)-f(x);
       %  disp(a);
         b = tol * Q(alpha,p,x);
         %disp(b);
         while(a /b< tol)
             alpha = alpha*0.5;
             %disp(alpha);
             a = f(x + alpha*p)-f(x);
             b = tol * Q(alpha,p,x);
             
         end
         
            fprintf("Iteration: %d",i );
            %X:[%e; %e;%e;%e] f: %e alpha: %e \nnorm of g(x): %e norm of (H-B):%e\n",... 
             %i,x(1),x(2),x(3),x(4), f(x), alpha, norm(g(x)),norm(H - B));
         
        
         s = alpha*p;
         y = g(x+alpha*p) - g(x);
        
        if(y'*s > 0)
            Bn = B - (1/(s'*B*s))*B*s*s'*B + (1/(y'*s))*(y*y');
            B= Bn;
        end
        
       
        fprintf("%e %e %e %e\n",[B(:,:)]');
       %  disp(eig(B));
        if(norm(g(x)) <= ftol || norm(g(x)) == 0)
            break;
        end
        
        x = x+alpha*p;
     
     end
end


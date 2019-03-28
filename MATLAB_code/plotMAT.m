function f = plotMAT(MAT1)

% OPT = 0;

eta = 1.0;
a = 1.0;
A1 = 3.3322;
A2 = 12.829;

XDEV = 38;
YDEV = 40;
XMIN = 0.01;
YMIN = 0.2;
XMAX = 0.2;
YMAX = 0.6;
% MAT1 = table2array(MAT);

DX = (XMAX-XMIN)/XDEV;
DY = (YMAX-YMIN)/YDEV;

% if OPT == 0
    X = zeros(XDEV,1);
    Y = zeros(YDEV,1);
    F = zeros(YDEV,XDEV);

    for i=1:1:XDEV
        X(i) = XMIN + (i-1)*DX + DX/2;
    end

    for i=1:1:YDEV
        Y(i) = YMIN + (i-1)*DY + DY/2;
    end
    
    for i = 1:1:YDEV
        for j = 1:1:XDEV
%             F(i,j) = -eta/2*Y(i)*(A1*(2*a/X(j))^(3/2) + A2*(2*a/X(j))^(1/2));
            F(i,j) = 1-X(j)^2-Y(i)^2; %% Real solution
        end
    end
    
    Z = abs(F-MAT1);   %% Absolute error
    ERR = zeros(YDEV,XDEV);
    
    for i=1:1:YDEV
        for j=1:1:XDEV
            ERR(i,j) = abs(Z(i,j)/F(i,j));
        end
    end
    
    
% else
%     X = XMIN:DX:XMAX;
%     Y = YMIN:DY:YMAX;
% end

%%%%%%%%%%%% Apply an avergae to the given set %%%%%%%%%%%%
% sstart = 2;
% num = 9;
% vstart = 40;
% vend = 60;
% ploterr = zeros(1,num);
% 
% for n = 1:1:num
%     for v = 1:1:(vend-vstart)
%         ploterr(n)  = ploterr(n) + Z(vstart+v-1,sstart+n-1);
%     end
% end
% ploterr = ploterr/(vend-vstart);
%**********************************************************
figure(2);
surf(X,Y',F);
shading interp;
% xlim([0.01 0.1]);
% ylim([0.2 0.8]);
% zlim([-3500 4]);
xlabel('s')
ylabel('Vab')
title('Original Function')

figure(3);
surf(X,Y',MAT1);
shading interp;
% xlim([0.02 0.1]);
% ylim([0.4 0.6]);
% zlim([0 0.1]);
xlabel('s')
ylabel('Vab')
% zlabel('Error')
title('Learned Function')

% plotx = 1:1:num;
% figure(3);
% plot(plotx,ploterr,'b-')

% f = sum(ploterr)/length(ploterr);

f = sum(sum(ERR))/(XDEV*YDEV);

end
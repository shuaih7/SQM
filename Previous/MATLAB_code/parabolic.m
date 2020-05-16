function f = parabolic(MAT0)

MAT0 = table2array(MAT0);

XMIN = -1;
XMAX = 1;
YMIN = -1;
YMAX = 1;
XDEV = 50;
YDEV = 50;
DX = (XMAX-XMIN)/XDEV;
DY = (YMAX-YMIN)/YDEV;

MAT = zeros(XDEV,YDEV);
for i = 1:1:XDEV
    for j = 1:1:YDEV
       MAT(i,j) = MAT0((i-1)*YDEV+j);
    end
end

X = zeros(1,XDEV);
Y = zeros(1,YDEV);
F = zeros(XDEV,YDEV);

for i = 1:1:XDEV
    X(i) = XMIN + (i-1)*DX + DX/2;
end

for j = 1:1:YDEV
    Y(j) = YMIN + (j-1)*DY + DY/2;
end

for i = 1:1:XDEV
    for j = 1:1:YDEV
       F(i,j) = 1-X(i)^2-Y(j)^2;
    end
end


figure(1);
surf(Y,X,F);
shading interp;
xlim([XMIN XMAX]);
ylim([YMIN YMAX]);
zlim([-1 1]);
xlabel('X');
ylabel('Y');
zlabel('Z = 1-X^2-Y^2');
title('Original Function');

figure(2);
surf(Y,X,MAT);
shading interp;
xlim([XMIN XMAX]);
ylim([YMIN YMAX]);
zlim([-1 1]);
xlabel('X');
ylabel('Y');
zlabel('Z = 1-X^2-Y^2');
title('Learned Function');

figure(3);
surf(Y,X,abs(F-MAT));
shading interp;
% xlim([0 1]);
% ylim([0 1]);
% zlim([-2700 0]);
xlabel('X');
ylabel('Y');
zlabel('e');
title('Absolute Error Distribution');

f = sum(sum(abs(F-MAT)))/sum(sum(abs(F)))/(XDEV*YDEV);

end
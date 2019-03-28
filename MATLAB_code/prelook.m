function f = prelook

OPT = 0;

eta = 1.0;
a = 1.0;
A1 = 3.3322;
A2 = 12.829;

vmin = 0.2;
vmax = 0.6;
smin = 0.01;
smax = 0.2;
vdev = 40;
sdev = 38;

dv = (vmax-vmin)/vdev;
ds = (smax-smin)/sdev;


if OPT == 0

    v = zeros(1,vdev);
    s = zeros(1,sdev);
    F = zeros(vdev,sdev);

    for i = 1:1:vdev
        v(i) = vmin + dv*(i-1) + dv/2;
    end
    for i = 1:1:sdev
        s(i) = smin + ds*(i-1) + ds/2;
    end

    for i = 1:1:(vdev)
        for j = 1:1:(sdev)
            F(i,j) = -eta/2*v(i)*(A1*(2*a/s(j))^(3/2) + A2*(2*a/s(j))^(1/2));
        end
    end
else
    v = vmin:dv:vmax;
    s = smin:ds:smax;
    
end


figure(1);
surf(s,v,F);
shading interp;
% xlim([0 1]);
% ylim([0 1]);
% zlim([-2700 0]);
xlabel('s');
ylabel('Vab');
zlabel('F');
title('Original Function');


f = 'J';
end
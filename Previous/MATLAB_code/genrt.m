function f = genrt

%*************** This function is to generate input data ****************
%************************************************************************

%*************************** Initialization *****************************
% format long;
nVar = 2;

P1MIN = -1;
P1MAX = 1;
P1DEV = 40;
DP1 = (P1MAX-P1MIN)/P1DEV;

P2MIN = -1;
P2MAX = 1;
P2DEV = 40;
DP2 = (P2MAX-P2MIN)/P2DEV;

T1MIN = -1;
T1MAX = 1;
T1DEV = 100;
DT1 = (T1MAX-T1MIN)/T1DEV;

T2MIN = -1;
T2MAX = 1;
T2DEV = 100;
DT2 = (T2MAX-T2MIN)/T2DEV;

Train = zeros((P1DEV+1)*(P2DEV+1),nVar);
Label = zeros((P1DEV+1)*(P2DEV+1),1);
Test = zeros((T1DEV+1)*(T2DEV+1),nVar);
TLabel = zeros((T1DEV+1)*(T2DEV+1),1);

NTrain = (P1DEV+1)*(P2DEV+1);
NTest = P1DEV*P2DEV;

Path0 = '/home/shuai/Templates/deep_learning/Pracs/Parabolic/train.txt';
Path1 = '/home/shuai/Templates/deep_learning/Pracs/Parabolic/test.txt';
Path2 = '/home/shuai/Templates/deep_learning/Pracs/Parabolic/trainlabel.txt';
Path3 = '/home/shuai/Templates/deep_learning/Pracs/Parabolic/testlabel.txt';

%************************************************************************

%************************* Feeding the values ***************************
for i = 1:1:P1DEV+1
    for j = 1:1:P2DEV+1
        index = (i-1)*(P2DEV+1)+j;
        Train(index,1) = P1MIN + (i-1)*DP1;
        Train(index,2) = P2MIN + (j-1)*DP2;
        Label(index) = 1-Train(index,1)^2-Train(index,2)^2;
    end
end

for i = 1:1:T1DEV+1
    for j = 1:1:T2DEV+1
        index = (i-1)*(T2DEV+1)+j;
        Test(index,1) = T1MIN + (i-1)*DT1;
        Test(index,2) = T2MIN + (j-1)*DT2;
        TLabel(index) = 1-Test(index,1)^2-Test(index,2)^2;
    end
end

%************************************************************************

%*************************** Normalization ******************************
% TOT = Train;
% for i = 1:1:nVar
%     me = mean(TOT(:,i));
%     stddev = max(TOT(:,i))-min(TOT(:,i));
%     if stddev == 0
%         fprintf('Error: standard deviation equals to 0.\n')
%         return;
%     end
%     Train(:,i) = (Train(:,i) - me)/stddev;
%     Test(:,i) = (Test(:,i) - me)/stddev;
% end
%************************************************************************

%*********************** Writting data to file **************************
fid0 = fopen(Path0, 'wt');
for i = 1:1:(P1DEV+1)*(P2DEV+1)
    for j = 1:1:nVar
        fprintf(fid0,'%8.6f ',Train(i,j));
    end
    fprintf(fid0,'\n');
end
fclose(fid0);

fid1 = fopen(Path1, 'wt');
for i = 1:1:(T1DEV+1)*(T2DEV+1)
    for j = 1:1:nVar
        fprintf(fid1,'%8.6f ',Test(i,j));
    end
    fprintf(fid1,'\n');
end
fclose(fid1);

fid2 = fopen(Path2, 'wt');
for i = 1:1:(P1DEV+1)*(P2DEV+1)
    fprintf(fid2,'%8.6f\n',Label(i));
end
fclose(fid2);

fid3 = fopen(Path3, 'wt');
for i = 1:1:(T1DEV+1)*(T2DEV+1)
    fprintf(fid3,'%8.6f\n',TLabel(i));
end
fclose(fid3);

%************************************************************************

f = 'J';

end
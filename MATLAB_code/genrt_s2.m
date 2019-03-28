function f = genrt_s2(Array)

%*************************** Basic Parameters ***************************
SP = 0;
TS = 10;
Len = 150;
NORM = 1;
loss_weights = [1 1 1]; % Length must be in correspondence with the input dimension.

Path0 = '/home/shuai/Templates/deep_learning/Sequence/S3_1/train.txt';
Path1 = '/home/shuai/Templates/deep_learning/Sequence/S3_1/test.txt';
Path2 = '/home/shuai/Templates/deep_learning/Sequence/S3_1/trainlabel.txt';
Path3 = '/home/shuai/Templates/deep_learning/Sequence/S3_1/testlabel.txt';
Pathp = '/home/shuai/Templates/deep_learning/Sequence/S3_1/parameters.txt';
Paths = '/home/shuai/Templates/deep_learning/Sequence/S3_1/scale.txt';
%************************************************************************

%***************************** Checking Box *****************************
Array = Array';
[TOTS,DIM] = size(Array);

if length(loss_weights) ~= DIM
    fprintf('Invalid length for loss_weights.\n');
    f = 'Fault';
    return;
end

if Len > TOTS-1 || Len < 1
   fprintf('Invalid Len value.\n')
   return;
end

MAXSP = Len - TS + 1;
if SP > MAXSP
    fprintf('Invalid SP value, please double check.\n');
    f = 'Fault';
    return;
else
    if SP <= 0
        SP = MAXSP;
    end
end
%************************************************************************

%*************************** Normalization ******************************
Mvalue = zeros(1,DIM);
Svalue = zeros(1,DIM);
% Scale = zeros(2,DIM);
Array0 = Array;           % Saves the original value of Array
if NORM == 1              % 0-1 normalization
    for i = 1:1:DIM
        Mvalue(i) = min(Array(:,i));
        Svalue(i) = max(Array(:,i))-Mvalue(i);
        if Svalue(i) == 0
           fprintf('The scaling parameter is equal to zero.\n'); 
           f = 'Fault';
           return;
        end
        for j = 1:1:TOTS
            Array(j,i) = (Array(j,i)-Mvalue(i))/Svalue(i)*loss_weights(i);
        end
    end
else
    for i = 1:1:DIM
        Mvalue(i) = 0;
        Svalue(i) = 1;
    end
end

% for i = 1:1:DIM
%     Scale(1,i) = Mvalue(i);
%     Scale(2,i) = Svalue(i);
% end
%************************************************************************

%************************* Feeding the values ***************************
TLen = TOTS - Len -1;
Train = zeros(SP,TS,DIM);
Test = zeros(TLen,TS,DIM);
Label = zeros(SP,DIM);
TLabel = zeros(TLen,DIM);

for i = 1:1:SP
    for k = 1:1:DIM
       for j = 1:1:TS
           Train(i,j,k) = Array((i-1)+j,k);
       end
       Label(i,k) = Array(i+TS,k);
    end
end

PRE = SP;
for i = 1:1:TLen
    for k = 1:1:DIM
       for j = 1:1:TS
          Test(i,j,k) = Array((i-1)+j+PRE,k);
       end
       TLabel(i,k) = Array0(i+PRE+TS,k); % The test label remains original
    end
end
%************************************************************************

%*********************** Writting data to file **************************
fid0 = fopen(Path0, 'wt');
for i = 1:1:SP
    for j = 1:1:TS
        for k = 1:1:DIM
            fprintf(fid0,'%8.6f ',Train(i,j,k));
        end
        fprintf(fid0,'\n');
    end
end
fclose(fid0);

fid1 = fopen(Path1, 'wt');
for i = 1:1:TLen
    for j = 1:1:TS
        for k = 1:1:DIM
            fprintf(fid1,'%8.6f ',Test(i,j,k));
        end
        fprintf(fid1,'\n');
    end
end
fclose(fid1);

fid2 = fopen(Path2, 'wt');
for i = 1:1:SP
    for k = 1:1:DIM
        fprintf(fid2,'%8.6f ',Label(i,k));
    end
    fprintf(fid2,'\n');
end
fclose(fid2);

fid3 = fopen(Path3, 'wt');
for i = 1:1:TLen
    for k = 1:1:DIM
        fprintf(fid3,'%8.6f ',TLabel(i,k));
    end
    fprintf(fid3,'\n');
end
fclose(fid3);

%******************** Writting the basic parameters *********************
Params = [TS, DIM, Len, TOTS];
fidp = fopen(Pathp, 'wt');
for i = 1:1:length(Params)
    fprintf(fidp,'%d\n',Params(i));
end
fclose(fidp);

fids = fopen(Paths, 'wt');
for i = 1:1:DIM
    fprintf(fids,'%8.6f ',Mvalue(i));
end
fprintf(fids,'\n');
for i = 1:1:DIM
    fprintf(fids,'%8.6f ',Svalue(i)/loss_weights(i));
end
fclose(fids);

%************************************************************************

f = 'J';

end
function f = genrt_s1(Array)

%*************************** Basic Parameters ***************************
SP = 0;
TS = 10;
% DIM = 1;
Len = 100;
NORM = 0;

Path0 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/train.txt';
Path1 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/test.txt';
Path2 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/trainlabel.txt';
Path3 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/testlabel.txt';
%************************************************************************

%***************************** Checking Box *****************************
if Len > (length(Array)-1) || Len < 1
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
if NORM == 1    
    TOT = Array;
    me = mean(TOT);
    Range = abs(max(TOT) - min(TOT));
    % Array = (TOT-me)/Range;
else
    me = 0;
    Range = 1;
end
%************************************************************************

%************************* Feeding the values ***************************
TLen = length(Array) - Len -1;
Train = zeros(SP,TS);
Test = zeros(TLen,TS);
Label = zeros(SP,1);
TLabel = zeros(TLen,1);

for i = 1:1:SP
   for j = 1:1:TS
       Train(i,j) = (Array((i-1)+j)-me)/Range;
   end
   Label(i) = (Array(i+TS)-me)/Range;
end

PRE = SP;
for i = 1:1:TLen
   for j = 1:1:TS
      Test(i,j) = (Array((i-1)+j+PRE)-me)/Range;
   end
   TLabel(i) = (Array(i+PRE+TS)-me)/Range;
end
%************************************************************************

%*********************** Writting data to file **************************
fid0 = fopen(Path0, 'wt')%*************************** Basic Parameters ***************************
SP = 0;
TS = 10;
% DIM = 1;
Len = 100;
NORM = 0;

Path0 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/train.txt';
Path1 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/test.txt';
Path2 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/trainlabel.txt';
Path3 = '/home/shuai/Templates/deep_learning/Pracs/Sequence/S1/testlabel.txt';
%************************************************************************

%***************************** Checking Box *****************************
if Len > (length(Array)-1) || Len < 1
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
if NORM == 1    
    TOT = Array;
    me = mean(TOT);
    Range = abs(max(TOT) - min(TOT));
    % Array = (TOT-me)/Range;
else
    me = 0;
    Range = 1;
end
%************************************************************************

%************************* Feeding the values ***************************
TLen = length(Array) - Len -1;
Train = zeros(SP,TS);
Test = zeros(TLen,TS);
Label = zeros(SP,1);
TLabel = zeros(TLen,1);

for i = 1:1:SP
   for j = 1:1:TS
       Train(i,j) = (Array((i-1)+j)-me)/Range;
   end
   Label(i) = (Array(i+TS)-me)/Range;
end

PRE = SP;
for i = 1:1:TLen
   for j = 1:1:TS
      Test(i,j) = (Array((i-1)+j+PRE)-me)/Range;
   end
   TLabel(i) = (Array(i+PRE+TS)-me)/Range;
end
%************************************************************************

%*********************** Writting data to file **************************
fid0 = fopen(Path0, 'wt');
for i = 1:1:SP
    for j = 1:1:TS
        fprintf(fid0,'%8.6f ',Train(i,j));
    end
    fprintf(fid0,'\n');
end
fclose(fid0);

fid1 = fopen(Path1, 'wt');
for i = 1:1:TLen
    for j = 1:1:TS
        fprintf(fid1,'%8.6f ',Test(i,j));
    end
    fprintf(fid1,'\n');
end
fclose(fid1);

fid2 = fopen(Path2, 'wt');
for i = 1:1:SP
    fprintf(fid2,'%8.6f\n',Label(i));
end
fclose(fid2);

fid3 = fopen(Path3, 'wt');
for i = 1:1:TLen
    fprintf(fid3,'%8.6f\n',TLabel(i));
end
fclose(fid3);

%************************************************************************;
for i = 1:1:SP
    for j = 1:1:TS
        fprintf(fid0,'%8.6f ',Train(i,j));
    end
    fprintf(fid0,'\n');
end
fclose(fid0);

fid1 = fopen(Path1, 'wt');
for i = 1:1:TLen
    for j = 1:1:TS
        fprintf(fid1,'%8.6f ',Test(i,j));
    end
    fprintf(fid1,'\n');
end
fclose(fid1);

fid2 = fopen(Path2, 'wt');
for i = 1:1:SP
    fprintf(fid2,'%8.6f\n',Label(i));
end
fclose(fid2);

fid3 = fopen(Path3, 'wt');
for i = 1:1:TLen
    fprintf(fid3,'%8.6f\n',TLabel(i));
end
fclose(fid3);

%************************************************************************


f = [me, Range];

end
function f = distribution(MAT,MATT)

%************************ Primary Operations ****************************
X = MAT(:,1);
Y = MAT(:,2);
LEN = length(X);
%************************************************************************


%***************** Setting up the Basic Parameters **********************
VALID = 200;                     % How many points for the validation set
NORM = 1;
nVar = 2;

Path0 = '/home/shuai/Templates/deep_learning/Dense/D2_1/train.txt';
Path1 = '/home/shuai/Templates/deep_learning/Dense/D2_1/test.txt';
Path2 = '/home/shuai/Templates/deep_learning/Dense/D2_1/trainlabel.txt';
Path3 = '/home/shuai/Templates/deep_learning/Dense/D2_1/testlabel.txt';
Path4 = '/home/shuai/Templates/deep_learning/Dense/D2_1/trainlabel0.txt';
% Pathp = '/home/shuai/Templates/deep_learning/Dense/D2_1/parameters.txt';
Paths = '/home/shuai/Templates/deep_learning/Dense/D2_1/scale.txt';
%************************************************************************


%************* Randomly Setting up the Validation Set *******************
RANDP = randperm(LEN);
RANDID = RANDP(1:VALID);
VALSET = zeros(VALID,2);
VLABEL = zeros(VALID,2);
for i = 1:1:VALID
   VALSET(i,1) = X(RANDID(i));
   VALSET(i,2) = Y(RANDID(i));
   VLABEL(i,:) = MATT(RANDID(i),:);
end
%************************************************************************


%********************** Shuffle the Training Set ************************
PRI_TR = MAT;
INDX = min(X)-1;
TRAIN = ones(VALID,2)*INDX;
TLABEL = zeros(VALID,2);

for i = 1:1:VALID
    PRI_TR(RANDID(i),1) = INDX;
end

index = 1;
for n = 1:1:LEN
   if PRI_TR(n,1) >= min(X)
       TRAIN(index,:) = PRI_TR(n,:);
       TLABEL(index,:) = MATT(n,:);
       index = index + 1;
   end
end

% stp = 0;
% for j = LEN:-1:1 
%     if TRAIN(j,1) < min(X)
%         TRAIN(j,:) = [];
%         TLABEL(j,:) = [];
%     else
%         stp = stp + 1;
%     end
%     
%     if stp > 0
%        break; 
%     end
% end

fprintf('\n');
fprintf('The number of samples in the training set is %d.\n',length(TRAIN(:,1)));
fprintf('The number of samples in the test set is %d.\n',LEN-length(TRAIN(:,1)));
%************************************************************************


%************************* Data Normalization ***************************
DIM = nVar;
Mvalue = zeros(2,DIM);                          % First row for inputs
Svalue = ones(2,DIM);                           % Second row for outputs
TRAINN = zeros(size(TRAIN));
VALSETN = zeros(size(VALSET));
TLABELN = zeros(size(TLABEL));
VLABELN = zeros(size(VLABEL));

if NORM == 1              % 0-1 normalization
    for i = 1:1:DIM
        Mvalue(1,i) = min(MAT(:,i));
        Mvalue(2,i) = min(MATT(:,i));
        Svalue(1,i) = max(MAT(:,i))-Mvalue(1,i);
        Svalue(2,i) = max(MATT(:,i))-Mvalue(2,i);
        if Svalue(i) == 0
           fprintf('The scaling parameter is equal to zero.\n'); 
           f = 'Fault';
           return;
        end
        
        for j = 1:1:length(TRAIN(:,1))
            TRAINN(j,i) = (TRAIN(j,i)-Mvalue(1,i))/Svalue(1,i);
            TLABELN(j,i) = (TLABEL(j,i)-Mvalue(2,i))/Svalue(2,i);
        end
        
        for j = 1:1:VALID
            VALSETN(j,i) = (VALSET(j,i)-Mvalue(1,i))/Svalue(1,i);
            VLABELN(j,i) = (VLABEL(j,i)-Mvalue(2,i))/Svalue(2,i);
        end   
    end
else
    TRAINN = TRAIN;
    VALSETN = VALSET;
    TLABELN = TLABEL;
    VLABELN = VLABEL;
end
%************************************************************************


%************ Print out the Training set & Validation Set ***************
%*** Note: Be sure to comment the following four lines during operation
% TRAIN = TRAINN;
% VALSET = VALSETN;
% TLABEL = TLABELN;
% VLABEL = VLABELN;

figure(1);
plot(TRAIN(:,1),TRAIN(:,2),strcat('o'), 'Color', [0.2 0.3 0.7], ...
    'MarkerFaceColor',[0.2 0.3 0.7],'MarkerSize', 2, 'LineWidth', 1.5);
hold on;
plot(VALSET(:,1),VALSET(:,2),strcat('o'), 'Color', [1 0 0], ...
    'MarkerFaceColor',[1 0 0],'MarkerSize', 2, 'LineWidth', 1.5);
% axis([-4.5 4.5 -1 5.2]);
hYLabel=ylabel('Input Y');
hXLabel=xlabel('Input X');
hLegend = legend('Training Dataset','Validation Dataset');

x0=0;
y0=0;
width=420;
height=315;
set(gcf,'units','points','position',[x0,y0,width,height])
set(gca, 'FontName', 'arial' );
set(gca,'FontSize', 15);
set([hXLabel hYLabel], 'FontSize', 16);
set(hLegend,'FontSize', 12, 'Box', 'off', 'location', 'northeast');
box on;

figure(2);
plot(TLABEL(:,1),TLABEL(:,2),strcat('o'), 'Color', [0.2 0.3 0.7], ...
    'MarkerFaceColor',[0.2 0.3 0.7],'MarkerSize', 2, 'LineWidth', 1.5);
hold on;
plot(VLABEL(:,1),VLABEL(:,2),strcat('o'), 'Color', [1 0 0], ...
    'MarkerFaceColor',[1 0 0],'MarkerSize', 2, 'LineWidth', 1.5);
% axis([-4.5 4.5 -1 5.2]);
hYLabel=ylabel('Input Y');
hXLabel=xlabel('Input X');
hLegend = legend('Training Dataset','Validation Dataset');

x0=0;
y0=0;
width=420;
height=315;
set(gcf,'units','points','position',[x0,y0,width,height])
set(gca, 'FontName', 'arial' );
set(gca,'FontSize', 15);
set([hXLabel hYLabel], 'FontSize', 16);
set(hLegend,'FontSize', 12, 'Box', 'off', 'location', 'northeast');
box on;

%**************************************************2**********************


%**************** Writing up the Training & Test Data *******************
fid0 = fopen(Path0, 'wt');
for i = 1:1:length(TRAINN(:,1))
    for j = 1:1:nVar
        fprintf(fid0,'%10.8f ',TRAINN(i,j));
    end
    fprintf(fid0,'\n');
end
fclose(fid0);

fid1 = fopen(Path1, 'wt');
for i = 1:1:length(VALSETN(:,1))
    for j = 1:1:nVar
        fprintf(fid1,'%10.8f ',VALSETN(i,j));
    end
    fprintf(fid1,'\n');
end
fclose(fid1);

fid2 = fopen(Path2, 'wt');
for i = 1:1:length(TLABELN(:,1))
    for j = 1:1:nVar
        fprintf(fid2,'%10.8f ',TLABELN(i,j));
    end
    fprintf(fid2,'\n');
end
fclose(fid2);

%*** Testlabel and Trainlabel0 are those which are not scaled ***
fid3 = fopen(Path3, 'wt');
for i = 1:1:length(VLABEL(:,1))
    for j = 1:1:nVar
        fprintf(fid3,'%10.8f ',VLABEL(i,j));
    end
    fprintf(fid3,'\n');
end
fclose(fid3);

fid4 = fopen(Path4, 'wt');
for i = 1:1:length(TLABEL(:,1))
    for j = 1:1:nVar
        fprintf(fid4,'%10.8f ',TLABEL(i,j));
    end
    fprintf(fid4,'\n');
end
fclose(fid4);

outscale = [Mvalue(2,:); Svalue(2,:)];
fids = fopen(Paths, 'wt');
for j = 1:1:2
    for i = 1:1:nVar
        fprintf(fids,'%10.8f ',outscale(j,i));
    end
    fprintf(fids,'\n');
end
fclose(fids);
%************************************************************************

f = 'Done';

end
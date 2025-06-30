%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��ȡ����
res = xlsread('Data_Model.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_dim = size(res, 2) - 1;               % ����ά��
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.8;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
P_train =  double(reshape(P_train, num_dim, 1, 1, M));
P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  ���ݸ�ʽת��
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

disp('���㷨���н����������ĵȴ���')

%%  �Ż��㷨��������
SearchAgents_no = 6;                   % ����
Max_iteration = 5 ;                    % ����������
dim = 3;                               % �Ż���������
lb = [1e-5,4 ,1e-5];                   % ����ȡֵ�½�(ѧϰ�ʣ�������������ϵ��)
ub = [5e-2, 128,1e-2];                 % ����ȡֵ�Ͻ�(ѧϰ�ʣ�������������ϵ��)

fitness = @(x)fical(x);

[Best_score,Best_pos,curve]=CPO(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness);
Best_pos(1, 2) = round(Best_pos(1, 2));   
best_hd  = Best_pos(1, 2); % ���ز�ڵ����
best_lr= Best_pos(1, 1);% ��ѳ�ʼѧϰ��
best_l2 = Best_pos(1, 3);% ���L2����ϵ��   

%%  ����ģ��
lgraph = layerGraph();                                  % �����հ�����ṹ

tempLayers = [
    sequenceInputLayer([num_dim, 1, 1], "Name", "sequence")  % ��������㣬�������ݽṹΪ[num_dim, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                % ���������۵���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = [
    convolution2dLayer([2, 1], 32, "Name", "conv_1")    % ���������, ����˴�С[2, 1], 32������ͼ
    reluLayer("Name", "relu_1")                         % Relu �����
    convolution2dLayer([2, 1], 64, "Name", "conv_2")];  % ���������, ����˴�С[2, 1], 64������ͼ
lgraph = addLayers(lgraph, tempLayers);                 % ����������ṹ����հ׽ṹ��

tempLayers = TransposeLayer("tans_1");                  % ά�Ƚ�����, �Ӷ��ڿռ�ά�Ƚ���GAP, ������ͨ��ά��
lgraph = addLayers(lgraph, tempLayers);                 % ����������ṹ����հ׽ṹ��

tempLayers = globalAveragePooling2dLayer("Name", "gapool");  % ȫ��ƽ���ػ���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = globalMaxPooling2dLayer("Name", "gmpool");      % ȫ�����ػ���
lgraph = addLayers(lgraph, tempLayers);                      % ����������ṹ����հ׽ṹ��

tempLayers = [
    concatenationLayer(1, 2, "Name", "concat")                         % ƴ�� GAP �� GMP ��Ľ��
    TransposeLayer("tans_2")                                           % ά�Ƚ�����, �ָ�ԭʼά��
    convolution2dLayer([1, 1], 1, "Name", "conv_3", "Padding", "same") % ���������, ͨ����Ŀ�任
    sigmoidLayer("Name", "sigmoid")];                                  % sigmoid �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % ��˵Ŀռ�ע����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % �������з��۵���
    flattenLayer("Name", "flatten")                                    % ������ƽ��
    lstmLayer(best_hd, "Name", "BiLSTM", "OutputMode", "last")         % LSTM��
    fullyConnectedLayer(num_class, "Name", "fc")                       % ȫ���Ӳ�
    softmaxLayer("Name", "softmax")                                    % softmax�����
    classificationLayer("Name", "classification")];                    % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % �۵������ ���� ���������
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
                                                                       % �۵������ ���� ���۵�������
lgraph = connectLayers(lgraph, "conv_2", "tans_1");                    % �������� ���� ά�ȱ任��
lgraph = connectLayers(lgraph, "conv_2", "multiplication/in2");        % �������� ���� ��˲�(ע����)����2
lgraph = connectLayers(lgraph, "tans_1", "gapool");                    % ά�ȱ任�� ���� GAP
lgraph = connectLayers(lgraph, "tans_1", "gmpool");                    % ά�ȱ任�� ���� GMP
lgraph = connectLayers(lgraph, "gapool", "concat/in2");                % GAP ���� ƴ�Ӳ�1
lgraph = connectLayers(lgraph, "gmpool", "concat/in1");                % GMP ���� ƴ�Ӳ�2
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % sigmoid ���� ��˲�
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % ������

%% ��������
options = trainingOptions('adam', ...     % Adam �ݶ��½��㷨
    'MaxEpochs', 50,...                  % ���ѵ������
    'InitialLearnRate', best_lr,...       % ��ʼѧϰ��Ϊ0.001
    'L2Regularization', best_l2,...       % L2���򻯲���
    'LearnRateSchedule', 'piecewise',...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1,...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 40,...        % ����x��ѵ���� ѧϰ��Ϊ 0.001*0.1
    'Shuffle', 'every-epoch',...          % ÿ��ѵ���������ݼ�
    'ValidationPatience', Inf,...         % �ر���֤
    'Plots', 'training-progress',...                   % ��������
    'Verbose', false);

%% ѵ��
net = trainNetwork(p_train, t_train, lgraph, options);

%% Ԥ��
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%% ����һ��
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%% ��������
error1 = sum((T_sim1 == T_train))/M * 100 ;
error2 = sum((T_sim2 == T_test)) /N * 100 ;

%% ��Ӧ�����߻�ͼ
figure
plot(curve,'linewidth',1.5);
title('��Ӧ�ȱ仯����');
xlabel('��������');
ylabel('��Ӧ��ֵ');
grid off
set(gcf,'color','w')

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'CPO-CNN-LSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid
set(gcf,'color','w')

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'CPO-CNN-LSTM-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid
set(gcf,'color','w')

%% ��������

if flag_conusion == 1

    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = 'Confusion Matrix for Train Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    set(gcf,'color','w')
    
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    set(gcf,'color','w')

end



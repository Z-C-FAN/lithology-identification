function fitness = fical(x)
%%  ���������л�ȡѵ������
    num_dim = evalin('base', 'num_dim');
    num_class = evalin('base', 'num_class');
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    
    
best_hd  = round(x(1, 2)); % ���ز�ڵ���
best_lr= x(1, 1);% ��ѳ�ʼѧϰ��
best_l2 = x(1, 3);% ���L2����ϵ��

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

%%  ��������
options = trainingOptions('adam', ...     % Adam �ݶ��½��㷨
    'MaxEpochs', 100,...                  % ���ѵ������ 1000
    'InitialLearnRate', best_lr,...       % ��ʼѧϰ��Ϊ0.001
    'L2Regularization', best_l2,...       % L2���򻯲���
    'LearnRateSchedule', 'piecewise',...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1,...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 85,...         % ����800��ѵ���� ѧϰ��Ϊ 0.001*0.1
    'Shuffle', 'every-epoch',...          % ÿ��ѵ���������ݼ�
    'ValidationPatience', Inf,...         % �ر���֤
    'Plots', 'none',...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, lgraph, options);

%%  ����Ԥ��
t_sim = classify(net, p_train);

%%  ������Ӧ��
fitness = (1 - sum(t_sim == t_train) / length(t_sim)) * 100;

end
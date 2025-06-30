function fitness = fical(x)
%%  从主函数中获取训练数据
    num_dim = evalin('base', 'num_dim');
    num_class = evalin('base', 'num_class');
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    
    
best_hd  = round(x(1, 2)); % 隐藏层节点数
best_lr= x(1, 1);% 最佳初始学习率
best_l2 = x(1, 3);% 最佳L2正则化系数

%%  建立模型
lgraph = layerGraph();                                  % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([num_dim, 1, 1], "Name", "sequence")  % 建立输入层，输入数据结构为[num_dim, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = [
    convolution2dLayer([2, 1], 32, "Name", "conv_1")    % 建立卷积层, 卷积核大小[2, 1], 32个特征图
    reluLayer("Name", "relu_1")                         % Relu 激活层
    convolution2dLayer([2, 1], 64, "Name", "conv_2")];  % 建立卷积层, 卷积核大小[2, 1], 64个特征图
lgraph = addLayers(lgraph, tempLayers);                 % 将上述网络结构加入空白结构中

tempLayers = TransposeLayer("tans_1");                  % 维度交换层, 从而在空间维度进行GAP, 而不是通道维度
lgraph = addLayers(lgraph, tempLayers);                 % 将上述网络结构加入空白结构中

tempLayers = globalAveragePooling2dLayer("Name", "gapool");  % 全局平均池化层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = globalMaxPooling2dLayer("Name", "gmpool");      % 全局最大池化层
lgraph = addLayers(lgraph, tempLayers);                      % 将上述网络结构加入空白结构中

tempLayers = [
    concatenationLayer(1, 2, "Name", "concat")                         % 拼接 GAP 和 GMP 后的结果
    TransposeLayer("tans_2")                                           % 维度交换层, 恢复原始维度
    convolution2dLayer([1, 1], 1, "Name", "conv_3", "Padding", "same") % 建立卷积层, 通道数目变换
    sigmoidLayer("Name", "sigmoid")];                                  % sigmoid 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的空间注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    lstmLayer(best_hd, "Name", "BiLSTM", "OutputMode", "last")         % LSTM层
    fullyConnectedLayer(num_class, "Name", "fc")                       % 全连接层
    softmaxLayer("Name", "softmax")                                    % softmax激活层
    classificationLayer("Name", "classification")];                    % 分类层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
                                                                       % 折叠层输出 连接 反折叠层输入
lgraph = connectLayers(lgraph, "conv_2", "tans_1");                    % 卷积层输出 链接 维度变换层
lgraph = connectLayers(lgraph, "conv_2", "multiplication/in2");        % 卷积层输出 链接 点乘层(注意力)输入2
lgraph = connectLayers(lgraph, "tans_1", "gapool");                    % 维度变换层 链接 GAP
lgraph = connectLayers(lgraph, "tans_1", "gmpool");                    % 维度变换层 链接 GMP
lgraph = connectLayers(lgraph, "gapool", "concat/in2");                % GAP 链接 拼接层1
lgraph = connectLayers(lgraph, "gmpool", "concat/in1");                % GMP 链接 拼接层2
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % sigmoid 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出

%%  参数设置
options = trainingOptions('adam', ...     % Adam 梯度下降算法
    'MaxEpochs', 100,...                  % 最大训练次数 1000
    'InitialLearnRate', best_lr,...       % 初始学习率为0.001
    'L2Regularization', best_l2,...       % L2正则化参数
    'LearnRateSchedule', 'piecewise',...  % 学习率下降
    'LearnRateDropFactor', 0.1,...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 85,...         % 经过800次训练后 学习率为 0.001*0.1
    'Shuffle', 'every-epoch',...          % 每次训练打乱数据集
    'ValidationPatience', Inf,...         % 关闭验证
    'Plots', 'none',...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, lgraph, options);

%%  仿真预测
t_sim = classify(net, p_train);

%%  计算适应度
fitness = (1 - sum(t_sim == t_train) / length(t_sim)) * 100;

end
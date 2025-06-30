%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('Data_Model.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_dim = size(res, 2) - 1;               % 特征维度
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.8;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);

%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
P_train =  double(reshape(P_train, num_dim, 1, 1, M));
P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  数据格式转换
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

disp('该算法运行较慢，请耐心等待！')

%%  优化算法参数设置
SearchAgents_no = 6;                   % 数量
Max_iteration = 5 ;                    % 最大迭代次数
dim = 3;                               % 优化参数个数
lb = [1e-5,4 ,1e-5];                   % 参数取值下界(学习率，批量处理，正则化系数)
ub = [5e-2, 128,1e-2];                 % 参数取值上界(学习率，批量处理，正则化系数)

fitness = @(x)fical(x);

[Best_score,Best_pos,curve]=CPO(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness);
Best_pos(1, 2) = round(Best_pos(1, 2));   
best_hd  = Best_pos(1, 2); % 隐藏层节点个数
best_lr= Best_pos(1, 1);% 最佳初始学习率
best_l2 = Best_pos(1, 3);% 最佳L2正则化系数   

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

%% 参数设置
options = trainingOptions('adam', ...     % Adam 梯度下降算法
    'MaxEpochs', 50,...                  % 最大训练次数
    'InitialLearnRate', best_lr,...       % 初始学习率为0.001
    'L2Regularization', best_l2,...       % L2正则化参数
    'LearnRateSchedule', 'piecewise',...  % 学习率下降
    'LearnRateDropFactor', 0.1,...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 40,...        % 经过x次训练后 学习率为 0.001*0.1
    'Shuffle', 'every-epoch',...          % 每次训练打乱数据集
    'ValidationPatience', Inf,...         % 关闭验证
    'Plots', 'training-progress',...                   % 画出曲线
    'Verbose', false);

%% 训练
net = trainNetwork(p_train, t_train, lgraph, options);

%% 预测
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%% 反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%% 性能评价
error1 = sum((T_sim1 == T_train))/M * 100 ;
error2 = sum((T_sim2 == T_test)) /N * 100 ;

%% 适应度曲线绘图
figure
plot(curve,'linewidth',1.5);
title('适应度变化曲线');
xlabel('迭代次数');
ylabel('适应度值');
grid off
set(gcf,'color','w')

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', 'CPO-CNN-LSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid
set(gcf,'color','w')

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', 'CPO-CNN-LSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid
set(gcf,'color','w')

%% 混淆矩阵

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



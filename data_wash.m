%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('Rock_iden3.xlsx');
%res = res(5e3:1e4,:);
%res(:,10)=[];

%% 异常值处理
% 一行元素全部相同为-99 直接删除
same_pos = [];
for i = 1:size(res,1)
      if all(res(i,1:end) == res(i,1))
            same_pos = [same_pos;i];
      end
end
res(same_pos,:) =[]; 
%% 缺失值处理 采用前面的值替代
missing = isnan(res);
for i=1:size(res,1)
    for j=1:size(res,2)
        if missing(i,j) == 1
            res(i,j) =  res(i-1,j);
        end
    end
end
%% 离群值处理
[B,TF,L,U,C] = filloutliers(res(:,:),"clip","movmedian",15);%滑动窗线性插值
res_new = B;
%% 绘制处理前特征与标签相关性热力图
R1 = corrcoef(res_new(:,[1,2])); %风速与功率相关性强

figure;
pcolor(R1);
shading flat; % 去掉网格线
colorbar; % 显示颜色条

% 美化图形
title('处理前相关性热力图');
xlabel('变量X');
ylabel('变量Y');
axis equal tight; % 等比例轴，紧凑范围

%% 相关性极差的也定义为异常值
% 处理：采用Ransac拟合后替代
x = res_new(:,1);
y = res_new(:,2);
xyPoints = [x y];


% RANSAC直线拟合
sampleSize = 2; % 每次采样的点数，直线为2
maxDistance = 0.005; % 内点到模型的最大距离
fitLineFcn = @(xyPoints) polyfit(xyPoints(:,1),xyPoints(:,2),1); % 拟合方式采用 polyfit，这里不可以用x,y替换xyPoints(:,1)，xyPoints(:,2)
evalLineFcn =  @(model, xyPoints) sum((y - polyval(model, x)).^2,2);% 距离估算函数
[modelRANSAC, inlierIdx] = ransac(xyPoints,fitLineFcn,evalLineFcn,sampleSize,maxDistance);% 执行RANSAC直线拟合，提取内点索引
modelInliers = polyfit(xyPoints(inlierIdx,1),xyPoints(inlierIdx,2),1);% 对模型内点进行最小二乘直线拟合

figure;
plot(xyPoints(inlierIdx,1),xyPoints(inlierIdx,2),'.');		% 内点
hold on;
plot(xyPoints(~inlierIdx,1),xyPoints(~inlierIdx,2),'ro');	% 外点
hold on;

inlierPts = xyPoints(inlierIdx,:);
x2 = linspace(min(inlierPts(:,1)),max(inlierPts(:,1)));
y2 = polyval(modelInliers,x2);
plot(x2, y2, 'g-');											% RANSAC直线拟合结果
hold off;

title('最小二乘直线拟合 与 RANSAC直线拟合 对比');
xlabel('X(m)');
ylabel('Y(m)');
zlabel('Z(m)');
legend('内点','噪声点','RANSAC直线拟合','Location','NorthWest');
%% 残差-孤立森林
% 计算Ransac理论值
T_linear = (modelRANSAC(1)*res_new(:,1)+modelRANSAC(2));
for i = 1:size(T_linear,1)
    if T_linear(i,end)<0
        T_linear(i,end)=0;
    end
end
residual_power = abs(res_new(:,2) - T_linear);

% plot(T_linear)
% 孤立森林判断异常值
[error_pos2] = iso_forest([res_new(:,[1,2]) residual_power]);

%% 替代异常值
for i = 1:size(error_pos2,2)
    res_new(error_pos2{i,1},end) = T_linear(error_pos2{i,1});
end
res_new(~inlierIdx,end) = T_linear(~inlierIdx);
%% 绘制处理后特征与标签相关性热力图
R2 = corrcoef(res_new(:,[1,2]));

figure;
pcolor(R2);
shading flat; % 去掉网格线
colorbar; % 显示颜色条

% 美化图形
title('处理后相关性热力图');
xlabel('变量X');
ylabel('变量Y');
axis equal tight; % 等比例轴，紧凑范围
%% 保存清洗好的数据
save res res_new;



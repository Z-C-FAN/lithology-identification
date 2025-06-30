classdef TransposeLayer < nnet.layer.Layer
%%  数据翻转
    methods
        function layer = TransposeLayer(name)
            layer.Name = name;
        end
        function Y = predict(~, X)

            if ndims(X) > 3
                Y = permute(X, [3, 2, 1, 4]);
            else
                Y = permute(X, [3, 2, 1]);
            end
          
        end
    end
end
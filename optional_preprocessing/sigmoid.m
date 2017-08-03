function y = sigmoid(x, a)
% DESCR:
% Applies a sigmoid function on the data x in [0-1] range. Then rescales
% the result so 0.5 will be mapped to itself.

% Apply Sigmoid
y = 1./(1+exp(-a*x)) - 0.5;

% Re-scale
y05 = 1./(1+exp(-a*0.5)) - 0.5;
y = y*(0.5/y05);

end

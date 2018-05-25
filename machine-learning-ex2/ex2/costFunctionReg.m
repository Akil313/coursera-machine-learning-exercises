function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


g = sigmoid(X * theta);

cf = ((1/m) * -((y' * log(g)) + ((1 - y)' * log(1 - g))));

J = cf + ((lambda/(2*m)) * sum(theta(2:n).^2));

grad1 = (1/m) * (X(:,1)' * (g - y));

grad2 = ((X' * (g - y)) * 1/m) + ((lambda/m) .* theta);

[grad2_x, grad2_y] = size(grad2);

grad2(1) = grad1;

grad = grad2

grad = round(grad2, 4)
% =============================================================

end

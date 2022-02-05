function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% qInstructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h_theta = X * theta; %(12 x 2) x (2 x 1)

% 一种写法是用点乘来开平方然后sum求和，或者像下面求grad一样用直接用其中一个的transpose
J = 1/(2*m) * sum((X * theta - y) .^ 2) + (lambda/(2*m)) * sum(theta(2:length(theta)) .^ 2);

thetaZero = theta;
thetaZero(1) = 0;

grad = ((1 / m) * (h_theta - y)' * X) + lambda / m * thetaZero';



% =========================================================================

grad = grad(:);

end

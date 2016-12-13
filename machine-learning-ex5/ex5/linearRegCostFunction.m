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
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

reg_theta = theta;
reg_theta(1,1) = 0;
T = X*theta - y;
J = (T'*T)/(2*m) + lambda * (reg_theta' * reg_theta)/(2*m);

% grad = ((X*theta - y)'*X)'/m + lambda*theta/m;
% grad(1,1) = (((X*theta - y)'*X)'/m)(1,1);
grad = ((X*theta - y)'*X)'/m + lambda*reg_theta/m;











% =========================================================================

grad = grad(:);

end

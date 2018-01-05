function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
theta_len = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = sum(-y .* log(sigmoid(X * theta)) - (1-y) .* log(1- sigmoid(X * theta)));
J = (J / m) + ((sum(theta(2:theta_len) .^ 2)) * lambda) /(2 * m);

% Gradient Descent
grad(1) = X(:,1)' * (sigmoid(X * theta) - y) ./ m;
grad(2:theta_len) = X(:,2:theta_len)' * (sigmoid(X * theta) - y) ./ m + theta(2:theta_len).*(lambda/m);
% =============================================================

end

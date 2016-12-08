function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%size(nn_params)
%nn_params
%input_layer_size
%hidden_layer_size
%num_labels
%
%size(X)
%X
%size(y)
%y
%
%size(Theta1)
%Theta1
%size(Theta2)
%Theta2
%
%lambda

m = size(X, 1);
K = num_labels;

for i=1:size(y)(1)
	test_y(i,:) = eye(K)(y(i),:);
end

X = [ones(m, 1) X];

reg = 0;
for j=1:size(Theta1)(1)
	for k=1:size(Theta1)(2)
		if k >1
			reg += Theta1(j,k)^2;
	end
	end
end
for j=1:size(Theta2)(1)
	for k=1:size(Theta2)(2)
		if k >1
		reg += Theta2(j,k)^2;
	end
	end
end
reg = lambda*reg/(2*m);

for i=1:m
	a2 = sigmoid(Theta1*X(i,:)');
	a2 = [1; a2];
	a3 = sigmoid(Theta2*a2);
	J += (-test_y(i,:)*log(a3) - (1 - test_y(i,:))*log(1-a3));
	J += reg;
end

J /= m;


%J = sum(-y.*log(sigmoid(X*theta)) - (1 - y).*log(1-sigmoid(X*theta))/m;

%grad = ((sigmoid(X*theta) - y)'*X)'/m + lambda*theta/m;
%grad(1,1) = (((sigmoid(X*theta) - y)'*X)'/m)(1,1);

for t=1:m
	a1 = X(t,:);
	z2 = Theta1*a1';
	a2 = [1; sigmoid(z2)];
	z3 = Theta2*a2;
	a3 = sigmoid(z3)
	test_y(t,:)'
	% delta3(:,t) = a3 - test_y(t,:)';
	% delta2(:,t) = Theta2'*(delta3(:,t).*sigmoidGradient(z2));
	delta3 = a3 - test_y(t,:)'
	size(Theta2')
	size(delta3)
	size(z2)
	size(sigmoidGradient(z2))
	size(delta3.*sigmoidGradient(z2))
	delta2 = Theta2'*(delta3.*sigmoidGradient(z2));
	% delta2 = delta2(2:end)
	% size(delta3(:,t))
	% size(delta2(:,t))
	% size(delta2)
	% size(a2)
	% [0; delta3(:,t)]'
	% a2
	% [0; delta3(:,t)]'*a2
	% D2 += [0; delta3(:,t)]'*a2;
	% D1 += delta2(:,t)'*a1'
	delta2 = delta2(2:end);
	if t == 1
		D2 = delta3 * a2';
		D1 = delta2 * a1;
	else
		D2 += delta3 * a2';
		D1 += delta2 * a1;
end

Theta1_grad = D1/m;
Theta2_grad = D2/m;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

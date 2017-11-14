function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
%% depth here starts with input layer (layer 1)
%% ith value in hAct stores values activation result
%% for (i+1)th layer
for depth=1:numHidden+1
  if depth == 1
    hAct{depth} = data;
  else
    W = stack{depth-1}.W;
    b = stack{depth-1}.b;
    z = bsxfun(@plus, W*hAct{depth-1}, b);
    hAct(depth) = 1./(1+exp(-z));
  end
end;

m = size(data, 2);
z = bsxfun(@plus, stack{numHidden+1}.W*hAct{numHidden+1}, stack{numHidden+1}.b);
h = exp(z);
pred_prob = bsxfun(@rdivide, h, sum(h, 1));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
d=zeros(size(pred_prob));
I = sub2ind(size(pred_prob), labels', 1:m);
d(I)=1;
error = -(d - pred_prob); 
ceCost = -sum(log(pred_prob(:))(I));


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
%% depth here starts with last hidden layer,
%% back all the way to input layer
for depth=numHidden+1:-1:1
  hl = hAct{depth};
  gradStack{depth}.W = error*hl';
  gradStack{depth}.b = sum(error, 2);
  W = stack{depth}.W;
  b = stack{depth}.b;
  %fprintf('size of error: (%d, %d)\n', size(error));
  %fprintf('size of W: (%d, %d)\n', size(W));
  %fprintf('size of h: (%d, %d)\n', size(hl));
  error = W'*error.*(hl.*(1-hl));
end;

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for depth=1:numHidden+1
  wCost += 0.5*ei.lambda*sum(stack{depth}.W(:).^2);
end
cost = ceCost + wCost;

for depth=1:numHidden+1
  gradStack{depth}.W += ei.lambda * stack{depth}.W;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

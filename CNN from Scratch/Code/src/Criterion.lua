require 'torch'
class = require 'class'
math = require 'math'

Criterion = class('Criterion')

function softmax(inputs)
    local inputs=torch.exp(inputs)
    local sums=torch.sum(inputs,2)
    local sumsi = torch.pow(sums,-1)
    sumsi = torch.repeatTensor(sumsi,1,inputs:size()[2])
    local softmax = torch.cmul(inputs,sumsi)
    return softmax
end

function one_hot(targets,n_classes)
    local h = targets:size()[1]
    local zeros = torch.zeros(h, n_classes)
    local indices = targets:view(-1, 1):long()
    local one_hot = zeros:scatter(2, indices, 1)
    return one_hot
end

function calc_loss(sm,targets)
    local sm=torch.log(sm)
    local loss=torch.cmul(sm,targets)
    loss=loss*-1
    loss=torch.sum(loss)
    return loss/sm:size()[1]
end

function calc_grad(sm,targets)
    return (sm-targets)/sm:size()[1]
end

function Criterion:__init()
    self._type="cross_entropy"
end

function Criterion:forward(inputs,targets)
    local sm = softmax(inputs)
    local targets_oh=one_hot(targets,inputs:size()[2])
    return calc_loss(sm,targets_oh)
end

function Criterion:backward(inputs,targets)
    local sm = softmax(inputs)
    local targets_oh=one_hot(targets,inputs:size()[2])
    return calc_grad(sm,targets_oh)
end
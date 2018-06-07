require 'torch'
class = require 'class'
math = require 'math'

Criterion = class('Criterion')

function softmax(inputs)
    local inputs=torch.exp(inputs)
    -- print(inputs)
    local sums=torch.sum(inputs)
    local sumsi = torch.pow(sums,-1)
    -- print(sumsi)
    -- sumsi = torch.repeatTensor(sumsi,1,inputs:size()[1])
    -- local softmax = torch.cmul(inputs,sumsi)
    local softmax = sumsi * inputs
    -- print(softmax)
    return softmax
end

function one_hot(targets,n_classes)
    -- local h = targets
    local one_hot = torch.zeros(n_classes)
    if targets == 0 then
        one_hot[1] = 1
    else
        one_hot[2] = 1
    end
    -- print(one_hot)
    -- local indices = targets:view(-1, 1):long()
    -- local one_hot = zeros:scatter(2, indices, 1)
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
    local targets_oh=one_hot(targets,inputs:size())
    return calc_loss(sm,targets_oh)
end

function Criterion:backward(inputs,targets)
    local sm = softmax(inputs)
    local targets_oh=one_hot(targets,inputs:size())
    return calc_grad(sm,targets_oh)
end
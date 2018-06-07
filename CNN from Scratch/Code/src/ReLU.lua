require 'torch'
class = require 'class'
nn = require 'nn'
math = require 'math'
ReLU = class('ReLU')

-- ReLU Class


function ReLU:__init(name)
    self._name=name
    self._output=nil
    self._gradInput=nil
    self._n_node=0
    -- self._relu=nn.ReLU()
end

function ReLU:forward(inputs)

    self._output=torch.cmul(inputs,inputs:gt(0):double())
    -- self._output=self._relu:forward(inputs)
    self._n_node=self._output:size()[1]
    return self._output
end

function ReLU:gradZero()
    -- self._relu:zeroGradParameters()
end

function ReLU:update(learning_rate,batch_size)
end

function ReLU:backward(inputs,gradOutput)
    self._gradInput=torch.cmul(inputs:gt(0):double(),gradOutput)
    -- self._gradInput=self._relu:backward(inputs,gradOutput)
    return self._gradInput
end
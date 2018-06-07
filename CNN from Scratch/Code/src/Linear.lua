require 'torch'
class = require 'class'
math = require 'math'


Linear = class('Linear')

function Linear:__init(n_input,n_node,name)
    self._name=name
    self._n_input=n_input
    self._n_node=n_node
    self._output=nil
    self._W = (torch.randn(n_node,n_input))/math.sqrt(n_input)
    self._gradW = torch.zeros(n_node,n_input)
    self._B = torch.randn(n_node)/math.sqrt(n_input)
    self._gradB = torch.zeros(n_node)
    self._gradIn= nil
    self._prev_gradW=torch.zeros(self._gradW:size())
    self._prev_gradB=torch.zeros(self._gradB:size())
end

function Linear:forward(inputs)
    local batch_size=inputs:size()[1]
    self._output=torch.zeros(batch_size,self._n_node)
    local rep_B=torch.repeatTensor(self._B,batch_size):resize(batch_size,self._n_node)
    self._output:addmm(0, self._output, 1, inputs, self._W:t())
    self._output:add(rep_B)
    return self._output
end

function Linear:gradZero()
    self._gradW:zero()
    self._gradB:zero()
end

function Linear:update(learning_rate,momentum)
    momentum = momentum or 0
    self._W:add(-learning_rate,self._gradW)
    self._W:add(-momentum*learning_rate,self._prev_gradW)
    self._B:add(-learning_rate,self._gradB)
    self._B:add(-momentum*learning_rate,self._prev_gradB)
end

function Linear:backward(inputs2,gradOut2)
    local batch_size=inputs2:size()[1]
    self._gradIn= torch.zeros(batch_size,self._n_input)
    self._gradIn:addmm(0, 1, gradOut2, self._W)
    self._prev_gradW=self._gradW:clone()
    self._prev_gradB=self._gradB:clone()
    self._gradW:addmm(gradOut2:t(), inputs2)
    self._gradB:addmv(1,gradOut2:t(), torch.ones(batch_size))
    return self._gradIn
end



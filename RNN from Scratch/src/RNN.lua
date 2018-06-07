require 'torch'
class = require 'class'
nn = require 'nn'
math = require 'math'

RNN = class('RNN')

function RNN:__init(n_input,n_hidden,n_output)

	self._n_input = n_input
	self._n_hidden = n_hidden
    self._n_output = n_output
    self._hidden_states = nil

    self._Wh = (torch.randn(n_hidden,n_hidden))/math.sqrt(n_hidden)
    self._Wx = (torch.randn(n_input,n_hidden))/math.sqrt(n_input)
    self._Wy = (torch.randn(n_hidden,n_output))/math.sqrt(n_hidden)
    self._Bh = torch.randn(n_hidden)
    self._By = torch.randn(n_output)  

    self._hidden = torch.zeros(n_hidden)
    self._output = nil

    self._gradWh = torch.zeros(n_hidden,n_hidden)
    self._gradWx = torch.zeros(n_input,n_hidden)
    self._gradBh = torch.zeros(n_hidden)
    self._gradWy = torch.zeros(n_hidden,n_output)
    self._gradBy = torch.zeros(n_output)
    self._gradIn = nil

    self._prev_gradWh = torch.zeros(n_hidden,n_hidden)
    self._prev_gradWx = torch.zeros(n_input,n_hidden)
    self._prev_gradBh = torch.zeros(n_hidden)
    self._prev_gradWy = torch.zeros(n_hidden,n_output)
    self._prev_gradBy = torch.zeros(n_output)

end


function RNN:forward(input)
    local no_samples = input:size()[1]

    self._hidden_states = torch.Tensor(no_samples,self._n_hidden)

    for i = 1,no_samples do
        -- print(input[i])
        self._hidden = torch.tanh(self._Wh:t() * self._hidden+self._Wx:t()*input[i] + self._Bh)
        self._hidden_states[i] = self._hidden
    end

    self._output = self._Wy:t() * self._hidden + self._By

    return self._output   
end

function RNN:gradZero()

    self._gradWh:zero()
    self._gradWx:zero()
    self._gradBh:zero()
    self._gradWy:zero()
    self._gradBy:zero()

end

function RNN:update(learning_rate,momentum)

    momentum = momentum or 0
    
    self._Wh:add(-learning_rate,self._gradWh)
    self._Wx:add(-learning_rate,self._gradWx)
    -- self._Wh:add(-momentum*learning_rate,self._prev_gradWh)
    self._Bh:add(-learning_rate,self._gradBh)
    -- self._Bh:add(-momentum*learning_rate,self._prev_gradBh)

    self._Wy:add(-learning_rate,self._gradWy)
    -- self._Wy:add(-momentum*learning_rate,self._prev_gradWy)
    self._By:add(-learning_rate,self._gradBy)
    -- self._By:add(-momentum*learning_rate,self._prev_gradBy)

end

function RNN:backward(input, gradOutput)

    -- ############# For momentum ############
    self._prev_gradWy = self._gradWy:clone()
    self._prev_gradBy = self._gradBy:clone()
    self._prev_gradWh = self._gradWh:clone()
    self._prev_gradBh = self._gradBh:clone()
    self._prev_gradWx = self._gradWx:clone()

    self._gradWy = gradOutput:view(self._n_output,1) * self._hidden:view(self._n_hidden,1):t()
    self._gradBy = gradOutput

    local gradIn = torch.cmul(self._Wy * gradOutput,(1-self._hidden_states[-1]:pow(2)))

    self._gradIn = gradIn

    local no_samples = input:size()[1]
    local temp = nil
    
    for i = no_samples,1,-1 do

        local hidden_state = torch.zeros(self._n_hidden)
        if i ~= 1 then
            hidden_state = self._hidden_states[i-1]
        end

        temp = hidden_state:view(self._n_hidden, 1)*gradIn:view(self._n_hidden,1):t()
        self._gradWh = self._gradWh + temp
        temp = input[i]:view(self._n_input,1)*gradIn:view(self._n_hidden,1):t()
        self._gradWx = self._gradWx + temp

        self._gradBh = self._gradBh + gradIn

        temp = self._Wh:clone()
        gradIn = torch.cmul(temp * gradIn,1-hidden_state:pow(2))

        self._gradIn = gradIn
    end

    return self._gradIn
end




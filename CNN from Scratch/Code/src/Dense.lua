require 'torch'
class = require 'class'
nn = require 'nn'
math = require 'math'
image = require 'image'

Dense = class('Dense')

function Dense:__init()
	self._output=nil
	self._gradIn=nil
end

function Dense:gradZero()
	
end

function Dense:update(learning_rate,batch_size)
	
end

function Dense:forward(inputs)
	ins=inputs:size()
	local batch_size=ins[1]
	h=ins[2]
	w=ins[3]
	self._output=inputs:reshape(batch_size,h*w)
	return self._output
end

function Dense:backward(inputs,gradIn)
	gs=gradIn:size()
	self._gradIn=gradIn:reshape(gs[1],math.sqrt(gs[2]),math.sqrt(gs[2]))
	return self._gradIn
end

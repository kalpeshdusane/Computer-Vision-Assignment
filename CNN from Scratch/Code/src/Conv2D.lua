require 'torch'
class = require 'class'
nn = require 'nn'
math = require 'math'
image = require 'image'

Conv2D= class('Conv2D')

function Conv2D:__init(n_in_filter,n_out_filter,w,h)
	self._conv = nn.SpatialConvolution(n_in_filter,n_out_filter,w,h)
	self._conv:noBias()
	self._w = w
	self._h = h
	self._output = nil
	self._gradW = torch.zeros(self._conv.weight:size())
	self._gradIn = nil
	self._prev_gradW=torch.zeros(self._gradW:size())
end

function Conv2D:gradZero()

	self._gradW:zero()
end

function Conv2D:update(learning_rate,batch_size)
	-- learning_rate=0.1
	-- print('gw',self._gradW)
	-- print('cw_update',self._conv.weight)
	self._conv.weight:add(-learning_rate,self._gradW)
end

function Conv2D:forward(inputs)
	ins = inputs:size()
	inputs = inputs:reshape(ins[1],1,ins[2],ins[3])
	local output = self._conv:forward(inputs)
	self._output = output:reshape(ins[1],output:size()[3],output:size()[4])
	-- print('cw_forward',self._conv.weight)
	return self._output
end

function Conv2D:backward(inputs,gradIn)
	self._gradW=nil
	local gs = gradIn:size()
	local ins = inputs:size()
	local inputs = inputs:reshape(1,ins[1],ins[2],ins[3])
	local convback = nn.SpatialConvolutionMM(gs[1],1,gs[2],gs[3])
	convback:noBias()
	convback.weight:copy(gradIn:reshape(1,gs[1]*gs[2]*gs[3]))
	local gradWeight = convback:forward(inputs)
	self._gradW=gradWeight
	-- print('cw_gradW',self._conv.weight)
	self._gradIn=nil
	local convback = nn.SpatialConvolutionMM(1,gs[1],gs[2],gs[3])
	convback:noBias()
	local gradin_wt=gradIn:reshape(1,gs[1]*gs[2]*gs[3])
	local gradin_flipped = torch.zeros(gradin_wt:size())
	for i=1,gradin_flipped:size()[2] do
		gradin_flipped[1][i]=gradin_wt[1][gradin_flipped:size()[2]-i+1]
	end
	convback.weight:copy(gradin_flipped)
	local conv_wt=self._conv.weight:clone()
	local padded_wt=torch.zeros(1,1,2*gs[3]+self._w-2,2*gs[2]+self._w-2)
	padded_wt[1][1][{{gs[2],gs[2]+conv_wt:size(4)-1},{gs[3],gs[3]+conv_wt:size(4)-1}}]=conv_wt
	local gradInput = convback:forward(padded_wt)
	local new_gradInput = torch.zeros(gradInput:size())
	for i=1,gs[1] do
		new_gradInput[1][i]=gradInput[1][gs[1]-i+1]
	end
	self._gradIn=new_gradInput:reshape(gs[1],gs[2]+self._w-1,gs[3]+self._w-1)
	return self._gradIn


end

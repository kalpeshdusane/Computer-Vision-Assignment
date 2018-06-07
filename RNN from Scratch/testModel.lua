require 'torch'
require 'class'
require 'math'
require 'src.RNN'
require 'src.preprocessing'
require 'src.Criterion'
require 'src.Model'

cmd = torch.CmdLine()
cmd:option('-modelName', 'bestModel')
cmd:option('-data', 'train_data.txt') 
cmd:option('-target', 'train_labels.txt')
cmd:option('-test', 'test_data.txt') 
cmd:text('testModel')

opt = cmd:parse(arg or {})

-- preprocess test data
print('==========================================')
print('        Loading and Processing Data       ')
print('==========================================')

-- local p = preprocessing('train_data.txt','test_data.txt','train_labels.txt')
local p=preprocessing(opt['data'],opt['test'],opt['target'])

local input_size=153
--  flexible parameter
local hidden_size=32
local output_size=2


print('==========================================')
print('           Predicting on Data             ')
print('==========================================')

model = Model(1,hidden_size,input_size,input_size)
model:load(opt['modelName']..'/')

local output_labels = torch.zeros(#p.test_x)

for i = 1, #p.test_x do
	x = p:get_test()
	-- print(x)
	local predicted_y = model:forward(x)

	if predicted_y[1] < predicted_y[2] then
		output_labels[i] = 1
	end
end

s = 'id,label\n'
for i=1,#p.test_x do
    s=s..(i-1)..','..output_labels[i]..'\n'
end

fd=io.open('prediction.csv','w')
fd:write(s)
fd:close()

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
cmd:text('trainModel')

opt = cmd:parse(arg or {})

-- local p=preprocessing('train_data.txt','test_data.txt','train_labels.txt')
local p=preprocessing(opt['data'],opt['test'],opt['target'])

local input_size=153
--  flexible parameter
local hidden_size=32
local output_size=2
local learning_rate = 0.01
local momentum = 0.0

print("=============== start training ========================")
local model = Model(1,hidden_size,input_size,input_size)
-- local rnn = RNN(input_size,hidden_size,output_size)

local loss=Criterion()
local print_loss=100


for e = 1,6 do
	
	local total_acc = 0
	local accuracy = 0
	local bl=0

	for i = 1, #p.train_x do
		x,y = p:get_batch()
		-- predicted_y = rnn:forward(x)
		predicted_y = model:forward(x)
		
		local correct = 0
		if predicted_y[1] < predicted_y[2] then correct = 1 end

		if correct == y then accuracy = accuracy + 1 end

		-- rnn:gradZero()
		model:zeroGradParameters()
		bl= loss:forward(predicted_y,y) + bl
		if(i%print_loss == 0) then
			print("Epoch : ",e," Iter : ",i," Loss : ",bl/print_loss," Accuracy : ", accuracy/print_loss)
			bl=0
			total_acc = total_acc + accuracy
			accuracy = 0
		end
		local gradLoss = loss:backward(predicted_y,y)

		-- rnn:backward(x,gradLoss)
		model:backward(x,gradLoss)
		-- rnn:update(learning_rate,momentum)
		model:updateParameters(learning_rate,momentum)
	end
	print("Total Accuracy : ", total_acc/#p.train_x)

	p.idx = 1

end
print("=============== training Completed ======================")

model:save(opt['modelName']..'/')


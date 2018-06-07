require 'torch'
require 'class'
require 'math'
require 'image'
require 'src.Linear'
require 'src.Conv2D'
require 'src.Dense'
require 'src.Model'
require 'src.ReLU'
require 'src.Criterion'

cmd = torch.CmdLine()
cmd:option('-config', 'bestModel/modelConfig.txt')
cmd:option('-i', 'data/data.bin') 
cmd:option('-ig', 'data/gradIn.bin')
cmd:option('-o', 'checkModel/output.bin')
cmd:option('-ow', 'checkModel/gradWeight.bin')
cmd:option('-ob', 'checkModel/gradBias.bin')
cmd:option('-og', 'checkModel/gradOutput.bin')
cmd:text('trainModel')

opt = cmd:parse(arg or {})

-- load data

data=torch.load(opt['i'])
data=data:float()
data=(data/255.0)-.5
n_data=data:size()[1]
imr=16
new_data=torch.zeros(n_data,imr,imr)
for i=1,n_data do
    new_data[i]=image.scale(data[i],imr,imr)
end
data=new_data

-- comment out this line if using Dense()
data=data:reshape(data:size(1),imr*imr)
-- load gradient in
local gradIn=torch.load(opt['ig'])

-- load model
local  mlp = Model()
mlp:load(opt['config'],true)

-- generate output
local output=mlp:forward(data)
torch.save(opt['o'],output)

-- generate gradOutput
local gradO=mlp:backward(data,gradIn)
torch.save(opt['og'],gradO)

-- generate gradW
local gradW={}
for i=1,#mlp._layers do
	table.insert(gradW,mlp._layers[i]._gradW)
end
torch.save(opt['ow'],gradW)

-- generate gradB
local gradB={}
for i=1,#mlp._layers do
	table.insert(gradB,mlp._layers[i]._gradB)
end
torch.save(opt['ob'],gradB)

print("Successfuly Checked Model ")

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
cmd:option('-modelName', 'bestModel')
cmd:option('-data', 'data/data.bin')  
cmd:text('trainModel')

opt = cmd:parse(arg or {})

-- preprocess test data
print('==========================================')
print('        Loading and Processing Data       ')
print('==========================================')

test=torch.load(opt['data'])
test=test:float()
test=(test/255.0)-.5
n_data=test:size()[1]
imr=16
new_data=torch.zeros(n_data,imr,imr)
for i=1,n_data do
    new_data[i]=image.scale(test[i],imr,imr)
end
test=new_data

print('==========================================')
print('           Predicting on Data             ')
print('==========================================')
mlp=Model()
mlp:load(opt['modelName']..'/')
function predict(data)
    local n_data=data:size()[1]
    local predict = nil
    local response = nil
    response=torch.DoubleTensor(mlp:forward(data):double())
    xx,predict=torch.max(response,2)
    predict=predict-1
    return predict
end

labels=predict(test)
torch.save('testPrediction.bin',labels)

-- save in csv

s='id,label\n'
for i=1,test:size()[1] do
    s=s..(i-1)..','..labels[i][1]..'\n'
end

fd=io.open('prediction.csv','w')
fd:write(s)
fd:close()


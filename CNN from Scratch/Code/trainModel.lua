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
cmd:option('-target', 'data/labels.bin') 
cmd:text('trainModel')

opt = cmd:parse(arg or {})

-- PREPROCESS DATA

print('==========================================')
print('        Loading and Processing Data       ')
print('==========================================')

data=torch.load(opt['data'])
val_len=2000
n_item=data:size()[1]-val_len
data=data[{{1,n_item+val_len}}]
labels=torch.load(opt['target'])
labels=labels[{{1,n_item+val_len}}]
data=data:float()
data=(data/255.0)-.5
labels=labels+1
imr=16
randomize=torch.randperm(n_item+val_len)
new_data=torch.zeros(data:size())
new_labels=torch.zeros(labels:size())
for i=1,n_item+val_len do
    new_data[i]=data[randomize[i]]
    new_labels[i]=labels[randomize[i]]
end
data=new_data
labels=new_labels
new_data=torch.zeros(n_item+val_len,imr,imr)
for i=1,n_item+val_len do
    new_data[i]=image.scale(data[i],imr,imr)
end
data=new_data

--- final processed data
val_data=data[{{n_item+1,val_len+n_item}}]
val_labels=labels[{{n_item+1,val_len+n_item}}]
data=data[{{1,n_item}}]
labels=labels[{{1,n_item}}]



-- define Model 

print('==========================================')
print('             Creating Model               ')
print('==========================================')

loss=Criterion()
mlp=Model()
mlp:addLayer(Dense(),'dense')
mlp:addLayer(Linear((imr-0)*(imr-0),10),'linear')
mlp:addLayer(ReLU(),'relu')
mlp:addLayer(Linear(10 , 6),'linear')

-- train Model

print('==========================================')
print('             Training Model               ')
print('==========================================')


-- prediction function

function predict(data)
    local n_data=data:size()[1]
    local predict = nil
    local response = nil
    response=torch.DoubleTensor(mlp:forward(data):double())
    xx,predict=torch.max(response,2)
    return predict
end

-- model hyperparameters

n_data=data:size()[1]
batch_size=128
n_batches=math.floor(n_data/batch_size)
n_epochs=100
learning_rate = .5
momentum = .5

-- training loop
for i=1,n_epochs do
    epoch_loss=0
    for j = 1,n_batches do
        batch_loss=0
        batch_x=(data[{{(j-1)*batch_size+1,j*batch_size}}])
        batch_y=labels[{{(j-1)*batch_size+1,j*batch_size}}]
        logits=mlp:forward(batch_x)
        mlp:zeroGradParameters()
        batch_loss=loss:forward(logits,batch_y)
        epoch_loss=epoch_loss+batch_loss
        gradLoss=loss:backward(logits,batch_y)
        mlp:backward(batch_x,gradLoss)
        mlp:updateParameters(.1,.5)
    end
    pred_labels=predict(val_data)
    accuracy=(torch.sum(torch.eq(pred_labels,val_labels:long()))/val_len)*100
    print('epoch :',i,' loss :',epoch_loss/n_batches,' Val Acc :',accuracy)
end

-- save model
paths.mkdir(opt['modelName'])
mlp:save(opt['modelName']..'/')

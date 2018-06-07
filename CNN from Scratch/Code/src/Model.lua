require 'torch'
class = require 'class'
math = require 'math'
require 'src.Linear'
require 'src.Conv2D'
require 'src.Dense'
require 'src.ReLU'


Model = class('Model')


function Model:__init()
    self._layers={}
    self._output=nil
    self._n_layers=0
    self._layer_names={}
end

function Model:addLayer(layer,name)
    table.insert(self._layers,layer)
    table.insert(self._layer_names,name)
    self._n_layers=self._n_layers+1
end

function Model:save(dir)
    dir = dir or 'bestModel/'
    print(dir)
    local f=io.open(dir..'modelConfig.txt','w')
    local s=''
    s=s..self._n_layers..'\n'
    local w_file=dir..'W.bin'
    local b_file=dir..'B.bin'

    local Ws={}
    local Bs={}
    for i=1,self._n_layers do
        local l=self._layers[i]
        local n=self._layer_names[i]
        if n=='linear' then 
            s=s..n..' '..l._n_input..' '..l._n_node..'\n'
        else
            s=s..n..'\n'
        end
        if l._W == nil then 
            table.insert(Ws,0)
            table.insert(Bs,0)
        else
            table.insert(Ws,l._W)
            table.insert(Bs,l._B)
        end
    end
    torch.save(w_file,Ws)
    torch.save(b_file,Bs)
    s=s..w_file..'\n'..b_file..'\n'
    f:write(s)
    f:close()
    print('========================================')
    print('Model Save Success !')
    print('========================================')
end

function Model:load(dir,load_from_config)
    load_from_config = load_from_config or false
    dir = dir or 'bestModel/'
    f=io.open(dir..'modelConfig.txt')
    if load_from_config then
        f=io.open(dir)
    end
    n=f:read()
    for i=1,n do
        line=f:read():split(' ')
        if line[1]=='linear' then
            self:addLayer(Linear(line[2],line[3]),line[1])
        elseif line[1] == 'relu' then
            self:addLayer(ReLU(),line[1])
        elseif line[1] == 'dense' then
            self:addLayer(Dense(),line[1])
        else
            print('Unknow Layer Type !')
        end
    end
    Ws=torch.load(f:read())
    Bs=torch.load(f:read())

    for i=1,n do
        if Ws[i] ~= 0 then
            self._layers[i]._W=Ws[i]:clone()
            self._layers[i]._B=Bs[i]:clone()
        end
    end
    print('========================================')
    print('Model Load Success !')
    print('========================================')
end

function Model:forward(inputs)
    local input = inputs
    for k, v in pairs(self._layers) do
        input=v:forward(input)
    end
    return input
end

function Model:updateParameters(learning_rate,momentum)
    for i=1,self._n_layers do
        self._layers[i]:update(learning_rate,momentum)
    end
end

function Model:zeroGradParameters()
    for i=1,self._n_layers do
        self._layers[i]:gradZero()
    end
end


function Model:backward(inputs,gradIn)
    n_last_layer=self._layers[self._n_layers]._n_node
    local batch_size=inputs:size()[1]

    local currentModule = self._layers[#self._layers]
    for i=#self._layers-1,1,-1 do
        local previousModule = self._layers[i]
        gradIn=currentModule:backward(previousModule._output,gradIn)
        currentModule=previousModule
    end
    gradIn=currentModule:backward(inputs,gradIn)
    return gradIn
end 

function Model:dispGradParam()
    for i=#self._layers,1,-1 do
        if self._layers[i]._gradW ~= nil then
            print('=================================================')
            print(': gradWeight       : Layer -> ',i)
            print('=================================================')
            print(self._layers[i]._gradW)
            print('=================================================')
            print(': gradBias         : Layer -> ',i)
            print('=================================================')
            print(self._layers[i]._gradB)
        end
    end
end
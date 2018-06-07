require 'torch'
class = require 'class'
math = require 'math'
require 'src.RNN'
require 'src.preprocessing'
require 'src.Criterion'

Model = class('Model')


function Model:__init(n_layers, n_hidden, voc_size, word_vec)
    self._n_layers = n_layers
    self._H = n_hidden
    self._V = voc_size
    self._D = word_vec
    -- self._isTrain = isTrain
    --  input size = voc_size
    self.rnn = RNN(self._V,self._H,2)
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
    -- for i=1,self._n_layers do
    --     local h = self._H
    --     local v = self._V
    --     s = s..h..' '..v..'\n'
    -- end
    table.insert(Ws,self.rnn._Wh)
    table.insert(Ws,self.rnn._Wx)
    table.insert(Ws,self.rnn._Wy)
    table.insert(Bs,self.rnn._Bh)
    table.insert(Bs,self.rnn._By)
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
    print(dir)
    n=f:read()
    print("Read File")
    -- for i=1,n do
    --     line=f:read():split('')
    --     print(line)
    --     local h = tonumber(line[1])
    --     local v = tonumber(line[2])
    --     self.rnn = RNN(v,h,2)
    -- end
    Ws=torch.load(f:read())
    Bs=torch.load(f:read())

    self.rnn._Wh = Ws[1]:clone()
    self.rnn._Wx = Ws[2]:clone()
    self.rnn._Wy = Ws[3]:clone()
    self.rnn._Bh = Bs[1]:clone()
    self.rnn._By = Bs[2]:clone()

    print('========================================')
    print('Model Load Success !')
    print('========================================')
end

function Model:forward(inputs)
    local predicted_y = self.rnn:forward(inputs)
    return predicted_y
end

function Model:updateParameters(learning_rate,momentum)
    self.rnn:update(learning_rate,momentum)
end

function Model:zeroGradParameters()
    self.rnn:gradZero()
end


function Model:backward(inputs,gradOutput)

    gradOutput = self.rnn:backward(inputs,gradOutput)
    return gradOutput

end 

require 'torch'
class = require 'class'
nn = require 'nn'
math = require 'math'

preprocessing = class('preprocessing')

function oneHotEncoder(n,count,number)
    -- print(n,count,number)
	local t = torch.zeros(count)
	local temp = number[n]
	t[temp] = 1
	return t
end

function preprocessing:__init(train_X,test_X,label_X,n_batch)

	-- print("=============== start preprocessing ======================")
	self.n_batch = n_batch or 1
	self.train_y={}
	self.train_x={}
	self.number={}
	self.test_x={}
	self.vocab_len=0
	self.idx=1

	local file = io.open(label_X, "r")
	local y = {}
	if file then
	    for line in file:lines() do
	        y[#y + 1] = tonumber(line)
	    end
	end
	self.train_y=y

	file = io.open(train_X,"r")
	local index = 0
	local number = {}
	local x = {}
	if file then
	    for line in file:lines() do
	        local row = {}
	        for i in string.gmatch(line, "%S+") do
	            local  temp = tonumber(i)
	            row[#row + 1] = temp
	            if number[temp] == nil then
	                index = index + 1
	                number[temp] = index                
	            end
	        end
	        x[#x + 1] = row
	    end
	end

	
	self.train_x=x
	-- self.number=number

	file = io.open(test_X,"r")
	if file then
	    for line in file:lines() do
	    	local row = {}
	        for i in string.gmatch(line,"%S+") do
	            local temp = tonumber(i) 
	            row[#row + 1] = temp
	            if number[temp] == nil then
	                index = index + 1
	                number[temp] = index
	            end
	        end 
	        self.test_x[#self.test_x+1]=row
	    end
	end

	self.number = number

	self.vocab_len=index

	for i = 1,#self.train_x do
		local z=torch.zeros(#self.train_x[i],self.vocab_len)
		for j = 1,#self.train_x[i] do
			z[j]=oneHotEncoder(self.train_x[i][j],self.vocab_len,self.number)
		end
		self.train_x[i]=z
	end

	for i = 1,#self.test_x do
		local z=torch.zeros(#self.test_x[i],self.vocab_len)
		for j = 1,#self.test_x[i] do
			z[j]=oneHotEncoder(self.test_x[i][j],self.vocab_len,self.number)
		end
		self.test_x[i]=z
	end

	-- print("=============== end preprocessing ========================")
end


function preprocessing:get_batch()
	self.idx = self.idx + 1
	return self.train_x[self.idx-1],self.train_y[self.idx-1]
end

function preprocessing:get_test()
	self.idx = self.idx + 1
	return self.test_x[self.idx-1]
end
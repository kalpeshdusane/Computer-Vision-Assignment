require 'src.Criterion'
require 'torch'



cmd = torch.CmdLine()
cmd:option('-i', 'demo_criterion_check/input.bin')
cmd:option('-t', 'demo_criterion_check/target.bin') 
cmd:option('-og', 'demo_criterion_check/gradInput.bin') 
cmd:text('Check Criterion')

opt = cmd:parse(arg or {})

input=torch.load(opt['i'])
target=torch.load(opt['t'])

cec=Criterion()

l1=cec:forward(input,target)
gradIn=cec:backward(input,target)
print('========================================')
print('|      Loss :  ',l1)
torch.save(opt['og'],gradIn)
print('========================================')
print('|        Done Criterion Check !        |')
print('========================================')

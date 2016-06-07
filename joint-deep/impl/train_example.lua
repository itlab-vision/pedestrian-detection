require 'torch'
require 'math'
init = require 'init'
m = require 'model'

epoches_num = init.epoches_num
batch_size = init.batch_size

data = torch.rand(120, 3, 84, 28)
labels = torch.Tensor(120, 2):fill(-1)
labels[{{}, {2}}] = torch.Tensor(120, 1):fill(1)

batches_num = math.floor(data:size(1) / batch_size)

-- net = torch.load('model1.t7')
-- print(net:parameters())

net = m.set_4layer_net(batch_size)
for ep = 1, epoches_num do
    print('Epoch ' .. ep .. ' started')
    for k = 1, batches_num do
        batch_x = data[{{(k - 1) * batch_size + 1, k * batch_size}, {}, {}, {}}]
        batch_y = labels[{{(k - 1) * batch_size + 1, k * batch_size}, {}}]
        m.forward(net, batch_x)
        m.backward(net, batch_x, batch_y)
        m.updateParameters(net)
        print('batch ' .. k .. ' done')
    end
    print(net:parameters()[1])
    -- torch.save('model' .. ep .. '.t7', net)
end

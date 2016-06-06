require 'nn'
require 'torch'
require 'math'
dt = require 'dt'
utils = require 'utils'
init = require 'init'

batch_size = init.batch_size
learning_rate = init.learning_rate
start_rows = init.start_rows
end_rows = init.end_rows
start_cols = init.start_cols
end_cols = init.end_cols

connections1 = init.connections1
connections2 = init.connections2

data = torch.rand(batch_size, 3, 84, 28)
labels = torch.Tensor(batch_size, 2):fill(0)
labels[{{}, {2}}] = torch.Tensor(batch_size, 1):fill(1)
-- print(data)

net = nn.Sequential()

rand_val = 0.5
net.w1 = torch.Tensor(7, 14)
net.w1[{{1, 6}, {1, 7}}] = rand_val * torch.rand(6, 7)
net.w1[{{7}, {1, 7}}] = rand_val * torch.randn(1, 7)
net.w1[{{}, {8, 14}}] = 0.1 * torch.randn(7, 7)
net.w2 = torch.Tensor(15, 14)
net.w2[{{1, 7}, {1, 7}}] = rand_val * torch.rand(7, 7)
net.w2[{{8, 14}, {1, 7}}] = 0.1 * torch.randn(7, 7)
net.w2[{{15}, {1, 7}}] = rand_val * torch.rand(1, 7)
net.w2[{{}, {8, 14}}] = 0.1 * torch.randn(15, 7)
net.w_class = torch.Tensor(15, 2)
net.w_class[{{1, 7}, {}}] = rand_val * torch.rand(7, 2)
net.w_class[{{8, 14}, {}}] = 0.1 * torch.randn(7, 2)
net.w_class[{{15}, {}}] = rand_val * torch.rand(1, 2)
net.w_class[{{}, {2}}] = -net.w_class[{{}, {1}}]
net.c1 = torch.Tensor(14)
net.c1[{{1, 7}}] = 0.1 * torch.randn(7)
net.c1[{{8, 14}}] = torch.Tensor(7):fill(0)
net.c2 = torch.Tensor(14)
net.c2[{{1, 7}}] = 0.1 * torch.randn(7)
net.c2[{{8, 14}}] = torch.Tensor(7):fill(0)
net.b4 = rand_val * torch.rand(batch_size, #start_rows)

-- Layer 2
net:add(nn.SpatialConvolution(3, 64, 9, 9))
net:add(nn.Tanh())
net:add(nn.Abs())

-- Layer 3
net:add(nn.SpatialAveragePooling(4, 4, 4, 4))

-- Layer 4
k_sizes = {}
ppos = {}
net.defw = torch.Tensor(batch_size, #start_rows, 4)
for i = 1, #start_rows do
    k_sizes[i] = {end_rows[i] - start_rows[i] + 1, end_cols[i] - start_cols[i] + 1}
    ppos[i] = {start_rows[i] + 2, start_cols[i]}
    if i ~= 18 then
        net.defw[{{}, i, {}}] = torch.repeatTensor(torch.Tensor({0.05, 0.0, 0.05, 0.0}), batch_size, 1)
    else
        net.defw[{{}, i, {}}] = torch.repeatTensor(torch.Tensor({1000.0, 0.0, 1000.0, 0.0}), batch_size, 1)
    end
end

n = nn.ConcatTable(1)
for i = 1, #k_sizes do
    n:add(nn.SpatialConvolution(64, 1, k_sizes[i][2], k_sizes[i][1]))
end

net:add(n)
net:forward(data)

-- print(#net.output)

-- print(net.output)
defvector = torch.Tensor(net.output[1]:size(1), #net.output, 4)
part_scores = torch.Tensor(net.output[1]:size(1), #net.output)
mapSizes = torch.Tensor(#net.output, 2)

for p = 1, #net.output do
    for k = 1, net.output[1]:size(1) do
        map = net.output[p][{k, {}, {}, {}}]:squeeze(1)
        local height = map:size()[1]
        local width = map:size()[2]
        mapSizes[p][1] = height
        mapSizes[p][2] = width
        -- print({height, width})
        local dst_col = torch.Tensor(height, width)
        local dst = torch.Tensor(height, width)
        local iy_tmp = torch.Tensor(height, width)
        local iy = torch.Tensor(height, width)
        local ix = torch.Tensor(height, width)
        for j = 1, width do
            dt.dt1d_by_column(map, dst_col, iy_tmp, j, -net.defw[{k, p, 3}], -net.defw[{k, p, 4}])
        end
        for i = 1, height do
            dt.dt1d_by_row(dst_col, dst, ix, i, -net.defw[{k, p, 1}], -net.defw[{k, p, 2}])
        end

        -- print(map)
        -- print(dst)
        -- print(iy_tmp)

        ix = utils.avoid_nans(ix)

        for i = 1, height do
            for j = 1, width do
                ix[i][j] = ix[i][j] + 1
                -- print(#iy_tmp, ix[i][j])
            end
        end
        dx = ppos[p][2] - ix[ppos[p][1]][ppos[p][2]]
        dy = ppos[p][1] - iy[ppos[p][1]][ppos[p][2]]
        defvector[{k, p, {}}] = -torch.Tensor({dx*dx, dx, dy*dy, dy})
        part_scores[k][p] = dst[ppos[p][1]][ppos[p][2]] + net.b4[k][p]
        end    
    -- print(iy)
    -- print(ix)
    -- print(ppos[k])
end

s1 = part_scores[{{}, {1, 6}}]
s2 = torch.Tensor(part_scores:size(1), 14):fill(0)
s2[{{}, {1, 7}}] = part_scores[{{}, {7, 13}}]
s3 = torch.Tensor(part_scores:size(1), 14):fill(0)
s3[{{}, {1, 7}}] = part_scores[{{}, {14, 20}}]

h1 = torch.Tensor(part_scores:size(1), 7)
h1[{{}, {1, 6}}] = torch.sigmoid(s1)
h1[{{}, {7}}] = 1.0

h2 = torch.Tensor(part_scores:size(1), 15)
c1 = torch.reshape(net.c1, 1, net.c1:size(1))
h2[{{}, {1, 14}}] = torch.sigmoid(-(h1 * net.w1 + torch.cmul(s2, c1:repeatTensor(part_scores:size(1), 1))))
h2[{{}, {15}}] = 1.0

h3 = torch.Tensor(part_scores:size(1), 15)
c2 = torch.reshape(net.c2, 1, net.c2:size(1))
h3[{{}, {1, 14}}] = torch.sigmoid(-(h2 * net.w2 + torch.cmul(s3, c2:repeatTensor(part_scores:size(1), 1))))
h3[{{}, {15}}] = 1.0

targetout = torch.exp(h3 * net.w_class) -- may be sigmoid?!
net.o = torch.cdiv(targetout, torch.repeatTensor(torch.sum(targetout, 2), 1, targetout:size(2)))

-- ======= backward =======

lp_w_class = net.o - labels   -- bsx2
dLdw_class = h3:t() * lp_w_class
dLdh3 = lp_w_class * net.w_class:t()  -- bsx15
lp3 = dLdh3:cmul(h3):cmul(1 - h3) -- bsx15
lp3 = lp3[{{}, {1, 14}}]
dLdw2 = h2:t() * lp3
dLdh2 = lp3 * net.w2:t()  -- bsx14
lp2 = dLdh2:cmul(h2):cmul(1 - h2) -- bsx15
lp2 = lp2[{{}, {1, 14}}]
dLdw1 = h1:t() * lp2
dLdh1 = lp2 * net.w1:t()  -- bsx7
lp1 = dLdh1:cmul(h1):cmul(1 - h1) -- bsx7
lp1 = lp1[{{}, {1, 6}}]

dLds3 = lp3 * 1.0   -- bsx14
dLds2 = lp2 * 1.0   -- bsx14
dLds1 = lp1 * 1.0   -- bsx6

dLdc = torch.Tensor(batch_size, #start_rows, 4)
dLdc[{{}, {1, 6}, {}}] = dLds1[{{}, {1, 6}}]:repeatTensor(1, 1, 4):cmul(defvector[{{}, {1, 6}, {}}])
dLdc[{{}, {7, 13}, {}}] = dLds2[{{}, {1, 7}}]:repeatTensor(1, 1, 4):cmul(defvector[{{}, {7, 13}, {}}])
dLdc[{{}, {14, 20}, {}}] = dLds3[{{}, {1, 7}}]:repeatTensor(1, 1, 4):cmul(defvector[{{}, {14, 20}, {}}])

dLds = {}
dv = torch.Tensor(batch_size, #start_rows)
dv[{{}, {1, 6}}] = dLds1[{{}, {1, 6}}]
dv[{{}, {7, 13}}] = dLds2[{{}, {1, 7}}]
dv[{{}, {14, 20}}] = dLds3[{{}, {1, 7}}]
for p = 1, #start_rows do
    dLds[p] = torch.Tensor(batch_size, 1, mapSizes[p][1], mapSizes[p][2])
    for m = 1, batch_size do
        d = torch.Tensor(1, mapSizes[p][1], mapSizes[p][2]):fill(dv[{m, p}])
        dLds[p][{m, {}, {}}] = d
    end
end

ddef = torch.Tensor(batch_size, #start_rows, 4)
for m = 1, batch_size do
    for p = 1, #start_rows do
        d = torch.Tensor(4):fill(dv[{m, p}])
        ddef[{m, p, {}}] = d
    end
end
ddefw = ddef:cmul(defvector)

ddefw = utils.avoid_nans(ddefw)
dLdw2 = utils.avoid_nans(dLdw2)
dLdw1 = utils.avoid_nans(dLdw1)

net:zeroGradParameters()
net:backward(data, dLds)

-- ======= updating learnable parameters =======
net:updateParameters(learning_rate)

net.defw = net.defw - learning_rate * ddefw
net.defw[{{}, {}, 1}] = torch.cmax(net.defw[{{}, {}, 1}], torch.Tensor(batch_size, #start_rows):fill(0.01))
net.defw[{{}, {}, 3}] = torch.cmax(net.defw[{{}, {}, 3}], torch.Tensor(batch_size, #start_rows):fill(0.01))
net.defw[{{}, 18, 1}] = 1000.0
net.defw[{{}, 18, 3}] = 1000.0

net.w_class = net.w_class - learning_rate * dLdw_class
net.w2 = net.w2 - learning_rate * dLdw2
net.w2[{{1, 7}, {1, 7}}] = net.w2[{{1, 7}, {1, 7}}]:cmul(connections2:t())
net.w1 = net.w1 - learning_rate * dLdw1
net.w1[{{1, 6}, {1, 7}}] = net.w1[{{1, 6}, {1, 7}}]:cmul(connections1:t())

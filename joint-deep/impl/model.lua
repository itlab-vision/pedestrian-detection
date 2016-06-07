require 'nn'
require 'torch'
require 'math'
dt = require 'dt'
utils = require 'utils'
init = require 'init'

local model = {}

local learning_rate = init.learning_rate
local start_rows = init.start_rows
local end_rows = init.end_rows
local start_cols = init.start_cols
local end_cols = init.end_cols
local parts_count = #start_rows

local connections1 = init.connections1
local connections2 = init.connections2

function model.set_4layer_net(batch_size)
    model.batch_size = batch_size

    -- Layer 1
    net = nn.Sequential()

    -- initialization
    net.w1 = init.w1
    net.w2 = init.w2
    net.w_class = init.w_class
    net.c1 = init.c1
    net.c2 = init.c2
    net.b4 = init.b4

    net.defvector = torch.Tensor(model.batch_size, parts_count, 4)
    net.part_scores = torch.Tensor(model.batch_size, parts_count)
    net.mapSizes = torch.Tensor(parts_count, 2)

    -- Layer 2
    net:add(nn.SpatialConvolution(3, 64, 9, 9))
    net:add(nn.Tanh())
    net:add(nn.Abs())

    -- Layer 3
    net:add(nn.SpatialAveragePooling(4, 4, 4, 4))

    -- Layer 4
    k_sizes = {}
    ppos = {}
    net.defw = torch.Tensor(parts_count, 4)
    for i = 1, parts_count do
        k_sizes[i] = {end_rows[i] - start_rows[i] + 1, end_cols[i] - start_cols[i] + 1}
        ppos[i] = {start_rows[i] + 2, start_cols[i]}
        if i ~= 18 then
            net.defw[{i, {}}] = torch.Tensor({0.05, 0.0, 0.05, 0.0})
        else
            net.defw[{i, {}}] = torch.Tensor({1000.0, 0.0, 1000.0, 0.0})
        end
    end

    n = nn.ConcatTable(1)
    for i = 1, #k_sizes do
        n:add(nn.SpatialConvolution(64, 1, k_sizes[i][2], k_sizes[i][1]))
    end
    net:add(n)

    return net
end

function model.forward(net, data)
    net:forward(data)
    for p = 1, parts_count do
        for k = 1, model.batch_size do
            map = net.output[p][{k, {}, {}, {}}]:squeeze(1)
            local height = map:size()[1]
            local width = map:size()[2]
            net.mapSizes[p][1] = height
            net.mapSizes[p][2] = width
            local dst_col = torch.Tensor(height, width)
            local dst = torch.Tensor(height, width)
            local iy_tmp = torch.Tensor(height, width)
            local ix_tmp = torch.Tensor(height, width)
            local iy = torch.Tensor(height, width)
            local ix = torch.Tensor(height, width)
            for j = 1, width do
                dt.dt1d_by_column(map, dst_col, iy_tmp, j, -net.defw[{p, 3}], -net.defw[{p, 4}])
            end
            for i = 1, height do
                dt.dt1d_by_row(dst_col, dst, ix_tmp, i, -net.defw[{p, 1}], -net.defw[{p, 2}])
            end

            for i = 1, height do
                for j = 1, width do
                    ix[i][j] = ix_tmp[i][j] + 1
                    iy[i][j] = iy_tmp[ix_tmp[i][j] + 1][j] + 1
                end
            end
            dx = ppos[p][2] - ix[ppos[p][1]][ppos[p][2]]
            dy = ppos[p][1] - iy[ppos[p][1]][ppos[p][2]]
            net.defvector[{k, p, {}}] = -torch.Tensor({dx*dx, dx, dy*dy, dy})
            net.part_scores[k][p] = dst[ppos[p][1]][ppos[p][2]] + net.b4[p]
        end
    end

    s1 = net.part_scores[{{}, {1, 6}}]
    s2 = torch.Tensor(net.part_scores:size(1), 14):fill(0)
    s2[{{}, {1, 7}}] = net.part_scores[{{}, {7, 13}}]
    s3 = torch.Tensor(net.part_scores:size(1), 14):fill(0)
    s3[{{}, {1, 7}}] = net.part_scores[{{}, {14, 20}}]

    net.h1 = torch.Tensor(net.part_scores:size(1), 7)
    net.h1[{{}, {1, 6}}] = torch.sigmoid(s1)
    net.h1[{{}, {7}}] = 1.0

    net.h2 = torch.Tensor(net.part_scores:size(1), 15)
    c1 = torch.reshape(net.c1, 1, net.c1:size(1))
    net.h2[{{}, {1, 14}}] = torch.sigmoid(-(net.h1 * net.w1 + torch.cmul(s2, c1:repeatTensor(net.part_scores:size(1), 1))))
    net.h2[{{}, {15}}] = 1.0

    net.h3 = torch.Tensor(net.part_scores:size(1), 15)
    c2 = torch.reshape(net.c2, 1, net.c2:size(1))
    net.h3[{{}, {1, 14}}] = torch.sigmoid(-(net.h2 * net.w2 + torch.cmul(s3, c2:repeatTensor(net.part_scores:size(1), 1))))
    net.h3[{{}, {15}}] = 1.0

    targetout = torch.exp(net.h3 * net.w_class) -- may be sigmoid?!
    net.o = torch.cdiv(targetout, torch.repeatTensor(torch.sum(targetout, 2), 1, targetout:size(2)))
end

function model.backward(net, data, labels)
    lp_w_class = net.o - labels   -- bsx2
    net.dLdw_class = net.h3:t() * lp_w_class
    dLdh3 = lp_w_class * net.w_class:t()  -- bsx15
    lp3 = dLdh3:cmul(net.h3):cmul(1 - net.h3) -- bsx15
    lp3 = lp3[{{}, {1, 14}}]
    net.dLdw2 = net.h2:t() * lp3
    dLdh2 = lp3 * net.w2:t()  -- bsx14
    lp2 = dLdh2:cmul(net.h2):cmul(1 - net.h2) -- bsx15
    lp2 = lp2[{{}, {1, 14}}]
    net.dLdw1 = net.h1:t() * lp2
    dLdh1 = lp2 * net.w1:t()  -- bsx7
    lp1 = dLdh1:cmul(net.h1):cmul(1 - net.h1) -- bsx7
    lp1 = lp1[{{}, {1, 6}}]

    dLds3 = lp3 * 1.0   -- bsx14
    dLds2 = lp2 * 1.0   -- bsx14
    dLds1 = lp1 * 1.0   -- bsx6

    dLdc = torch.Tensor(model.batch_size, parts_count, 4)
    dLdc[{{}, {1, 6}, {}}] = dLds1[{{}, {1, 6}}]:repeatTensor(1, 1, 4):cmul(net.defvector[{{}, {1, 6}, {}}])
    dLdc[{{}, {7, 13}, {}}] = dLds2[{{}, {1, 7}}]:repeatTensor(1, 1, 4):cmul(net.defvector[{{}, {7, 13}, {}}])
    dLdc[{{}, {14, 20}, {}}] = dLds3[{{}, {1, 7}}]:repeatTensor(1, 1, 4):cmul(net.defvector[{{}, {14, 20}, {}}])

    dLds = {}
    dv = torch.Tensor(model.batch_size, parts_count)
    dv[{{}, {1, 6}}] = dLds1[{{}, {1, 6}}]
    dv[{{}, {7, 13}}] = dLds2[{{}, {1, 7}}]
    dv[{{}, {14, 20}}] = dLds3[{{}, {1, 7}}]
    for p = 1, parts_count do
        dLds[p] = torch.Tensor(model.batch_size, 1, net.mapSizes[p][1], net.mapSizes[p][2])
        for m = 1, model.batch_size do
            d = torch.Tensor(1, net.mapSizes[p][1], net.mapSizes[p][2]):fill(dv[{m, p}])
            dLds[p][{m, {}, {}}] = d
        end
    end

    ddef = dv:repeatTensor(4, 1, 1)
    for i = 1, 4 do
        ddef[{{i}, {}, {}}] = ddef[{{i}, {}, {}}]:cmul(net.defvector[{{}, {}, {i}}])
    end
    net.ddefw = torch.sum(ddef, 2):squeeze(2):t() / model.batch_size
    net.db4 = torch.sum(dv, 1):squeeze(1) / model.batch_size

    net:zeroGradParameters()
    net:backward(data, dLds)
end

function model.updateParameters(net)
    net:updateParameters(learning_rate)

    net.defw = net.defw - learning_rate * net.ddefw
    net.defw[{{}, 1}] = torch.cmax(net.defw[{{}, 1}], torch.Tensor(parts_count):fill(0.01))
    net.defw[{{}, 3}] = torch.cmax(net.defw[{{}, 3}], torch.Tensor(parts_count):fill(0.01))
    net.defw[{{18}, {1}}] = 1000.0
    net.defw[{{18}, {3}}] = 1000.0
    net.b4 = net.b4 - learning_rate * net.db4

    net.w_class = net.w_class - learning_rate * net.dLdw_class
    net.w2 = net.w2 - learning_rate * net.dLdw2
    net.w2[{{1, 7}, {1, 7}}] = net.w2[{{1, 7}, {1, 7}}]:cmul(connections2:t())
    net.w1 = net.w1 - learning_rate * net.dLdw1
    net.w1[{{1, 6}, {1, 7}}] = net.w1[{{1, 6}, {1, 7}}]:cmul(connections1:t())    
end

return model

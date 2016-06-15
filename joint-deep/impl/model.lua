require 'nn'
require 'torch'
require 'math'
dt = require 'dt'
utils = require 'utils'
init = require 'init'
require 'torch_extra'

local model = {}

local learning_rate = init.learning_rate
local start_rows = init.start_rows
local end_rows = init.end_rows
local start_cols = init.start_cols
local end_cols = init.end_cols
local parts_count = #start_rows

local connections1 = init.connections1
local connections2 = init.connections2

function model.initializeNet(pos_batch_size, neg_batch_size)
    model.pos_batch_size = pos_batch_size
    model.neg_batch_size = neg_batch_size
    model.batch_size = pos_batch_size + neg_batch_size

    -- Layer 1
    net = nn.Sequential()

    net.w1 = init.w1
    net.w2 = init.w2
    net.w_class = init.w_class
    net.c2 = init.c2
    net.c3 = init.c3

    net.mapSizes = torch.Tensor(parts_count, 2)

    -- Layer 2

    net:add(nn.SpatialConvolution(3, 64, 9, 9))

    local m = net.modules[1]
    for i = 1, #init.k2 do
        for j = 1, #(init.k2[1]) do
            m.weight[{j, i, {}, {}}] = torch.Tensor(init.k2[i][j])
        end
    end
    m.bias = torch.Tensor(init.b2)

    net:add(nn.Tanh())
    -- net:add(nn.Abs())

    -- Layer 3
    net:add(nn.SpatialAveragePooling(4, 4, 4, 4))

    -- Layer 4
    local k_sizes = {}
    net.ppos = {}
    net.defw = torch.Tensor(parts_count, 4)
    for i = 1, parts_count do
        k_sizes[i] = {end_rows[i] - start_rows[i] + 1, end_cols[i] - start_cols[i] + 1}
        net.ppos[i] = {start_rows[i] + 2, start_cols[i]}
        if i ~= 18 then
            net.defw[{i, {}}] = torch.Tensor({0.05, 0.0, 0.05, 0.0})
        else
            net.defw[{i, {}}] = torch.Tensor({1000.0, 0.0, 1000.0, 0.0})
        end
    end

    local kWs = torch.Tensor(parts_count)
    local kHs = torch.Tensor(parts_count)
    for p = 1, parts_count do
        kWs[p] = k_sizes[p][2]
        kHs[p] = k_sizes[p][1]
    end
    net:add(nn.CustomSpatialConvolution(64, 20, kWs, kHs))

    return net
end

function model.forward(net, data)
    net:forward(data)

    defvector = torch.Tensor(model.batch_size, parts_count, 4)
    local part_scores = torch.Tensor(model.batch_size, parts_count)
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
            dx = net.ppos[p][2] - ix[net.ppos[p][1]][net.ppos[p][2]]
            dy = net.ppos[p][1] - iy[net.ppos[p][1]][net.ppos[p][2]]
            defvector[{k, p, {}}] = -torch.Tensor({dx*dx, dx, dy*dy, dy})
            local weights, grads = net:parameters()
            local b = weights[4][p]
            part_scores[k][p] = dst[net.ppos[p][1]][net.ppos[p][2]] + b
        end
    end

    local s1 = part_scores[{{}, {1, 6}}]
    local s2 = torch.Tensor(part_scores:size(1), 14):fill(0)
    s2[{{}, {1, 7}}] = part_scores[{{}, {7, 13}}]
    local s3 = torch.Tensor(part_scores:size(1), 14):fill(0)
    s3[{{}, {1, 7}}] = part_scores[{{}, {14, 20}}]

    net.h1 = torch.Tensor(part_scores:size(1), 7)
    net.h1[{{}, {1, 6}}] = torch.sigmoid(s1)
    net.h1[{{}, {7}}] = 1.0

    net.h2 = torch.Tensor(part_scores:size(1), 15)
    local c2 = torch.reshape(net.c2, 1, net.c2:size(1))
    c2 = c2:repeatTensor(part_scores:size(1), 1)
    net.h2[{{}, {1, 14}}] = torch.sigmoid(net.h1 * net.w1 + torch.cmul(s2, c2))
    net.h2[{{}, {15}}] = 1.0

    net.h3 = torch.Tensor(part_scores:size(1), 15)
    local c3 = torch.reshape(net.c3, 1, net.c3:size(1))
    c3 = c3:repeatTensor(part_scores:size(1), 1)
    net.h3[{{}, {1, 14}}] = torch.sigmoid(net.h2 * net.w2 + torch.cmul(s3, c3))
    net.h3[{{}, {15}}] = 1.0

    local targetout = torch.exp(net.h3 * net.w_class)
    net.o = torch.cdiv(targetout, torch.repeatTensor(torch.sum(targetout, 2), 1, targetout:size(2)))
    -- print(net.o)

    if pos_batch_size and neg_batch_size then
        local pos_characteristic =
            torch.sum(net.o[{{1, model.pos_batch_size}, 1}]) / (model.pos_batch_size)
        local neg_characteristic =
            torch.sum(net.o[{{model.pos_batch_size + 1, model.batch_size}, 1}]) / (model.neg_batch_size)
        print(pos_characteristic)
        print(neg_characteristic)

        if (pos_characteristic > 0.5) and (neg_characteristic < 0.5) then
            model.can_save = true
        else
            model.can_save = false
        end
    end
end

function model.backward(net, data, labels)
    lp_w_class = net.o - labels   -- bsx2
    net.dLdw_class = net.h3:t() * lp_w_class
    dLdh3 = lp_w_class * net.w_class:t()  -- bsx15
    lp3 = dLdh3:cmul(net.h3):cmul(1 - net.h3) -- bsx15
    lp3 = lp3[{{}, {1, 14}}]
    net.dLdw2 = net.h2:t() * lp3
    net.dLdw2[{{1, 7}, {1, 7}}] = net.dLdw2[{{1, 7}, {1, 7}}]:cmul(connections2:t())
    dLdh2 = lp3 * net.w2:t()  -- bsx14
    lp2 = dLdh2:cmul(net.h2):cmul(1 - net.h2) -- bsx15
    lp2 = lp2[{{}, {1, 14}}]
    net.dLdw1 = net.h1:t() * lp2
    net.dLdw1[{{1, 6}, {1, 7}}] = net.dLdw1[{{1, 6}, {1, 7}}]:cmul(connections1:t())
    dLdh1 = lp2 * net.w1:t()  -- bsx7
    lp1 = dLdh1:cmul(net.h1):cmul(1 - net.h1) -- bsx7
    lp1 = lp1[{{}, {1, 6}}]

    dLds3 = lp3 * 1.0   -- bsx14
    dLds2 = lp2 * 1.0   -- bsx14
    dLds1 = lp1 * 1.0   -- bsx6

    dLds = {}
    dv = torch.Tensor(model.batch_size, parts_count)
    dv[{{}, {1, 6}}] = dLds1[{{}, {1, 6}}]
    dv[{{}, {7, 13}}] = dLds2[{{}, {1, 7}}]
    dv[{{}, {14, 20}}] = dLds2[{{}, {8, 14}}]
    for p = 1, parts_count do
        dLds[p] = torch.Tensor(model.batch_size, 1, net.mapSizes[p][1], net.mapSizes[p][2])
        for m = 1, model.batch_size do
            d = torch.Tensor(1, net.mapSizes[p][1], net.mapSizes[p][2]):fill(0.0)
            d[1][net.ppos[p][1]][net.ppos[p][2]] = dv[m][p]
            dLds[p][{m, {}, {}}] = d
        end
    end

    net.ddefw = torch.Tensor(parts_count, 4)
    for p = 1, parts_count do
        ddef = dv[{{}, p}]
        ddef = ddef:repeatTensor(4, 1)
        ddef = ddef:t():cmul(defvector[{{}, {p}, {}}]:squeeze(2))
        net.ddefw[{p, {}}] = torch.sum(ddef, 1):squeeze(1) / model.batch_size
    end

    net:zeroGradParameters()
    net:backward(data, dLds)
end

function model.updateParameters(net)
    if model.ready_to_save and model.can_save then
        -- Good model. Don't change parameters
        return
    end

    net:updateParameters(learning_rate)

    net.defw = net.defw - learning_rate * net.ddefw
    net.defw[{{}, 1}] = torch.cmax(net.defw[{{}, 1}], torch.Tensor(parts_count):fill(0.01))
    net.defw[{{}, 3}] = torch.cmax(net.defw[{{}, 3}], torch.Tensor(parts_count):fill(0.01))
    net.defw[{{18}, {1}}] = 1000.0
    net.defw[{{18}, {3}}] = 1000.0

    net.w_class = net.w_class - learning_rate * net.dLdw_class
    net.w2 = net.w2 - learning_rate * net.dLdw2
    net.w2[{{1, 7}, {1, 7}}] = net.w2[{{1, 7}, {1, 7}}]:cmul(connections2:t())
    net.w1 = net.w1 - learning_rate * net.dLdw1
    net.w1[{{1, 6}, {1, 7}}] = net.w1[{{1, 6}, {1, 7}}]:cmul(connections1:t())

    -- Regularize weights
    regval = 0.01
    w1_tmp = net.w1[{{1, 6}, {1, 7}}]
    w1_tmp[torch.lt(w1_tmp, regval)] = regval
    net.w1[{{1, 6}, {1, 7}}] = w1_tmp
    net.w1[{{-1}, torch.gt(net.w1[{{-1}, {}}], -0.1)}] = -0.1
    w2_tmp = net.w2[{{1, 7}, {1, 7}}]
    w2_tmp[torch.lt(w2_tmp, regval)] = regval
    net.w2[{{1, 7}, {1, 7}}] = w2_tmp
    net.w2[{{-1}, torch.gt(net.w2[{{-1}, {}}], -0.1)}] = -0.1
    w_class_tmp = net.w_class[{{1, 7}, 1}]
    w_class_tmp[torch.lt(w_class_tmp, regval)] = regval
    net.w_class[{{1, 7}, 1}] = w_class_tmp
    net.w_class[{{}, 2}] = -net.w_class[{{}, 1}]

    net.c2[torch.lt(net.c2, regval)] = regval
    net.c3[torch.lt(net.c3, regval)] = regval
end

function model.readyToSave(is_ready)
    model.ready_to_save = is_ready
end

function model.canSave()
    return model.can_save
end

return model

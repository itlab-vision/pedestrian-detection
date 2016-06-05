require 'nn'
require 'torch'
require 'math'
dt = require 'dt'

batch_size = 10
start_rows = {1, 1, 4, 4, 9, 9, 1, 1, 1, 4, 4, 4, 9, 1, 1, 1, 1, -1, 1, 1 }
end_rows = {3, 3, 9, 9, 15, 15, 3, 9, 9, 9, 15, 15, 15, 3, 9, 15, 15, 17, 15, 15 }
start_cols = {1, 3, 1, 3, 2, 4, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 1 }
end_cols = {3, 5, 3, 5, 3, 5, 5, 2, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 5, 5 }

data = torch.rand(batch_size, 3, 84, 28)
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
defw = torch.Tensor(batch_size, #start_rows, 4)
for i = 1, #start_rows do
    k_sizes[i] = {end_rows[i] - start_rows[i] + 1, end_cols[i] - start_cols[i] + 1}
    ppos[i] = {start_rows[i] + 2, start_cols[i]}
    if i ~= 18 then
        defw[{{}, i, {}}] = torch.repeatTensor(torch.Tensor({0.05, 0.0, 0.05, 0.0}), batch_size, 1)
    else
        defw[{{}, i, {}}] = torch.repeatTensor(torch.Tensor({1000.0, 0.0, 1000.0, 0.0}), batch_size, 1)
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

for p = 1, #net.output do
    for k = 1, net.output[1]:size(1) do
        map = net.output[p][{k, {}, {}, {}}]:squeeze(1)
        local height = map:size()[1]
        local width = map:size()[2]
        -- print({height, width})
        local dst_col = torch.Tensor(height, width)
        local dst = torch.Tensor(height, width)
        local iy_tmp = torch.Tensor(height, width)
        local iy = torch.Tensor(height, width)
        local ix = torch.Tensor(height, width)
        for j = 1, width do
            dt.dt1d_by_column(map, dst_col, iy_tmp, j, 0.05, 0.1)
        end
        for i = 1, height do
            dt.dt1d_by_row(dst_col, dst, ix, i, 0.05, 0.1)
        end

        -- print(map)
        -- print(dst)
        -- print(iy_tmp)

        nan_mask = ix:ne(ix)
        ix[nan_mask] = 0

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

targetout = torch.exp(h3 * net.w_class)
net.o = torch.cdiv(targetout, torch.repeatTensor(torch.sum(targetout, 2), 1, targetout:size(2)))

print(net.o)

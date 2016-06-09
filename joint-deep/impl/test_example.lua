require 'lfs'
utils = require 'utils'
init = require 'init'
m = require 'model'
require 'torch'

for i = 1, 7, 2 do
    if arg[i] == '--test-data-path' then
        test_data_path = arg[i + 1]
    end
    if arg[i] == '--test-data-name' then
        test_data_name = arg[i + 1]
    end
    if arg[i] == '--model-path' then
        model_path = arg[i + 1]
    end
    if arg[i] == '--test-batch-size' then
        test_batch_size = tonumber(arg[i + 1])
    end
end

if (test_data_path == nil or test_data_name == nil or
    model_path == nil or test_batch_size == nil) then
    print('Incorrect arguments')
    os.exit()
end

test_out_dir = 'test-out/'
test_out_path = test_out_dir .. test_data_name .. '/'
if (utils.dir_exists(test_out_dir) == false) then
    lfs.mkdir(test_out_dir)
end
if (utils.dir_exists(test_out_path) == false) then
    lfs.mkdir(test_out_path)
end

m.batch_size = test_batch_size
net = torch.load(model_path)

net.ppos = {}
for i = 1, #init.start_rows do
    net.ppos[i] = {init.start_rows[i] + 2, init.start_cols[i]}
end

fo = io.open(test_out_path .. 'test_out.txt', 'w')
file_idx = 0
fname = test_data_path .. 'test_x_' .. file_idx .. '.txt'
batch_idx = 1
chunk = nil
while(utils.file_exists(fname)) do
    dofile(fname)
    r_chunk = torch.Tensor(test_x)
    chunk_size = 0
    if (chunk ~= nil) then
        chunk_size = chunk:size(1)
    end
    ext_chunk = torch.Tensor(chunk_size + r_chunk:size(1), r_chunk:size(2), r_chunk:size(3), r_chunk:size(4))
    if (chunk ~= nil) then
        ext_chunk[{{1, chunk_size}, {}, {}, {}}] = chunk
        chunk = nil
    end
    ext_chunk[{{chunk_size + 1, chunk_size + r_chunk:size(1)}, {}, {}, {}}] = r_chunk
    ext_batches_num = math.floor(ext_chunk:size(1) / test_batch_size)
    if (ext_batches_num > 0) then
        data = torch.Tensor(ext_batches_num * test_batch_size, r_chunk:size(2), r_chunk:size(3), r_chunk:size(4))
        data[{{1, ext_batches_num * test_batch_size}, {}, {}, {}}] = ext_chunk[{{1, ext_batches_num * test_batch_size}, {}, {}, {}}]
        for k = 1, ext_batches_num do
            batch_x = data[{{(k - 1) * test_batch_size + 1, k * test_batch_size}, {}, {}, {}}]
            m.forward(net, batch_x)
            out = net.o
            for i = 1, out:size(1) do
                fo:write(out[i][1] .. ', ' .. out[i][2] .. '\n')
            end
            print('Batch ' .. batch_idx .. ' tested')
            batch_idx = batch_idx + 1
        end
    end
    if (ext_batches_num * test_batch_size < ext_chunk:size(1)) then
        fed_size = ext_chunk:size(1) - ext_batches_num * test_batch_size
        chunk = torch.Tensor(fed_size, r_chunk:size(2), r_chunk:size(3), r_chunk:size(4))
        chunk[{{1, fed_size}, {}, {}, {}}] = ext_chunk[{{ext_batches_num * test_batch_size + 1, ext_batches_num * test_batch_size + fed_size}, {}, {}, {}}]
    end
    file_idx = file_idx + 1
    fname = test_data_path .. 'test_x_' .. file_idx .. '.txt'
end

fo:close()

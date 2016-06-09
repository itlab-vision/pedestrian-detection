require 'os'
require 'lfs'
require 'torch'
require 'math'
utils = require 'utils'
init = require 'init'
m = require 'model'

epoches_num = init.epoches_num
batch_size = init.batch_size
dataset_name = 'CaltechTrain'
if (utils.dir_exists('models') == false) then
    lfs.mkdir('models')
end
models_path = 'models/' .. dataset_name
if (utils.dir_exists(models_path) == false) then
    lfs.mkdir(models_path)
end

dofile(dataset_name .. '/train_y.txt')
labels = torch.Tensor(train_y)
data_dir = dataset_name .. '/train_x'

net = m.set_4layer_net(batch_size)

for ep = 1, epoches_num do
    print('Epoch ' .. ep .. ' started')
    file_idx = 0
    batch_idx = 1
    chunk = nil
    fname = data_dir .. '/train_x_' .. file_idx .. '.txt'

    start_time = os.time()

    while(utils.file_exists(fname)) do
        dofile(fname)
        r_chunk = torch.Tensor(train_x)
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
        ext_batches_num = math.floor(ext_chunk:size(1) / batch_size)
        if (ext_batches_num > 0) then
            data = torch.Tensor(ext_batches_num * batch_size, r_chunk:size(2), r_chunk:size(3), r_chunk:size(4))
            data[{{1, ext_batches_num * batch_size}, {}, {}, {}}] = ext_chunk[{{1, ext_batches_num * batch_size}, {}, {}, {}}]
            for k = 1, ext_batches_num do
                batch_x = data[{{(k - 1) * batch_size + 1, k * batch_size}, {}, {}, {}}]
                batch_y = labels[{{(batch_idx - 1) * batch_size + 1, batch_idx * batch_size}, {}}]
                m.forward(net, batch_x)
                m.backward(net, batch_x, batch_y)
                m.updateParameters(net)
                print('Batch ' .. batch_idx .. ' done')
                batch_idx = batch_idx + 1
            end
        end
        if (ext_batches_num * batch_size < ext_chunk:size(1)) then
            fed_size = ext_chunk:size(1) - ext_batches_num * batch_size
            chunk = torch.Tensor(fed_size, r_chunk:size(2), r_chunk:size(3), r_chunk:size(4))
            chunk[{{1, fed_size}, {}, {}, {}}] = ext_chunk[{{ext_batches_num * batch_size + 1, ext_batches_num * batch_size + fed_size}, {}, {}, {}}]
        end
        file_idx = file_idx + 1
        fname = data_dir .. '/train_x_' .. file_idx .. '.txt'
        if batch_idx >= 100 then
            break
        end
    end

    elapsed_time = os.time() - start_time
    model_path = models_path .. '/model' .. ep .. '.t7'
    torch.save(model_path, net)
    print('Model trained and saved to ' .. model_path .. '. Elapsed time: ' .. elapsed_time .. ' secs')
end

require 'os'
require 'lfs'
require 'torch'
require 'math'
utils = require 'utils'
init = require 'init'
m = require 'model'

for i = 1, #arg, 2 do
    if arg[i] == '--train-data-path' then
        train_data_path = arg[i + 1]
    end
    if arg[i] == '--dataset-name' then
        dataset_name = arg[i + 1]
    end
    if arg[i] == '--data-size' then
        data_size = tonumber(arg[i + 1])
    end
end

if (train_data_path == nil or dataset_name == nil or
    data_size == nil) then
    print('Incorrect arguments')
    os.exit()
end

epoches_num = init.epoches_num
if (utils.dir_exists('models') == false) then
    lfs.mkdir('models')
end
models_path = 'models/' .. dataset_name
if (utils.dir_exists(models_path) == false) then
    lfs.mkdir(models_path)
end

-- Load labels
labels_dir = dataset_name .. '/train_y'
labels = torch.Tensor(data_size, 2)
loaded_data = 0
file_idx = 0
fname = labels_dir .. '/train_y_' .. file_idx .. '.txt'
while(utils.file_exists(fname)) do
    dofile(fname)
    r_chunk = torch.Tensor(train_y)
    labels[{{loaded_data + 1, loaded_data + r_chunk:size(1)}, {}}] = r_chunk
    loaded_data = loaded_data + r_chunk:size(1)
    file_idx = file_idx + 1
    fname = labels_dir .. '/train_y_' .. file_idx .. '.txt'
    if loaded_data >= data_size then
        break
    end
end
labels = labels[{{1, data_size}, {}}]

-- Load data
data_dir = dataset_name .. '/train_x'
data = torch.Tensor(data_size, 3, 84, 28)
loaded_data = 0
file_idx = 0
fname = data_dir .. '/train_x_' .. file_idx .. '.txt'
while(utils.file_exists(fname)) do
    dofile(fname)
    r_chunk = torch.Tensor(train_x)
    data[{{loaded_data + 1, loaded_data + r_chunk:size(1)}, {}, {}, {}}] = r_chunk
    loaded_data = loaded_data + r_chunk:size(1)
    file_idx = file_idx + 1
    fname = data_dir .. '/train_x_' .. file_idx .. '.txt'
    if loaded_data >= data_size then
        break
    end
end

-- Prepare train batch
pos_count = torch.sum(torch.eq(labels[{{}, {1}}], 1))
neg_count = torch.sum(torch.eq(labels[{{}, {1}}], 0))
neg_batch_size = init.neg_batch_size
pos_batch_size = math.floor(neg_batch_size / 5)
batch_size = pos_batch_size + neg_batch_size
batch_count = math.floor(neg_count / neg_batch_size)
pos_scale_factor = math.ceil(pos_batch_size * batch_count / pos_count)
pos_idxs = torch.LongTensor(pos_count)
neg_idxs = torch.LongTensor(neg_count)
pos_idx_counter = 1
neg_idx_counter = 1
for i = 1, labels:size(1) do
    if labels[i][1] == 1 then
        pos_idxs[pos_idx_counter] = i
        pos_idx_counter = pos_idx_counter + 1
    else
        neg_idxs[neg_idx_counter] = i
        neg_idx_counter = neg_idx_counter + 1
    end
end
pos_idxs = pos_idxs:repeatTensor(pos_scale_factor)
print('pos_count: ' .. pos_count .. ', neg_count: ' .. neg_count)
print('pos_batch_size: ' .. pos_batch_size .. ', neg_batch_size: ' .. neg_batch_size)

-- Initialize net
net = m.initializeNet(pos_batch_size, neg_batch_size)

saved_models_count = 0
local ready_to_save = false
for ep = 1, epoches_num do
    print('Epoch ' .. ep .. ' started')
    pos_perm_idxs = torch.randperm(pos_count * pos_scale_factor):type('torch.LongTensor')
    neg_perm_idxs = torch.randperm(neg_count):type('torch.LongTensor')

    start_time = os.time()

    for k = 1, batch_count do
        if (k / batch_count >= 0.95) then
            ready_to_save = true
            m.readyToSave(true)
        end
        pos_selection = pos_idxs:index(1, pos_perm_idxs[{{(k - 1) * pos_batch_size + 1, k * pos_batch_size}}])
        neg_selection = neg_idxs:index(1, neg_perm_idxs[{{(k - 1) * neg_batch_size + 1, k * neg_batch_size}}])
        batch_x = torch.Tensor(batch_size, data:size(2), data:size(3), data:size(4))
        batch_x[{{1, pos_batch_size}, {}, {}, {}}] = data:index(1, pos_selection)
        batch_x[{{pos_batch_size + 1, batch_size}, {}, {}, {}}] = data:index(1, neg_selection)
        batch_y = torch.Tensor(batch_size, 2)
        batch_y[{{1, pos_batch_size}, {}}] = labels:index(1, pos_selection)
        batch_y[{{pos_batch_size + 1, batch_size}, {}}] = labels:index(1, neg_selection)

        m.forward(net, batch_x)
        m.backward(net, batch_x, batch_y)
        m.updateParameters(net)
        print('Batch ' .. k .. ' done')

        if ready_to_save and m.canSave() then
            elapsed_time = os.time() - start_time
            saved_models_count = saved_models_count + 1
            model_path = models_path .. '/model' .. saved_models_count .. '.t7'
            torch.save(model_path, net)
            print('Model trained and saved to ' .. model_path .. '. Elapsed time: ' .. elapsed_time .. ' secs')
            ready_to_save = false
            m.readyToSave(false)
            break
        end
    end
end

require "nn"
require "image"

num_conv = 2
gauss_kernel = image.gaussian(9, 0.25, 1, true)

model = nn.Sequential()

-- 1st layer
--model:add(nn.SplitTable(1))
    par = nn.ParallelTable()

        func_y = nn.Sequential()
        func_y:add(nn.SpatialConvolutionMap(nn.tables.full(1, 32), 7, 7, 1, 1))
        func_y:add(nn.ReLU())
        func_y:add(nn.SpatialContrastiveNormalization(32, gauss_kernel))
        func_y:add(nn.SpatialAveragePooling(3, 3, 3, 3))

        func_uv = nn.Sequential()
        func_uv:add(nn.SpatialAveragePooling(3, 3, 3, 3))
        func_uv:add(nn.SpatialConvolutionMap(nn.tables.full(2, 6), 3, 3))
        func_uv:add(nn.ReLU())
        func_uv:add(nn.SpatialContrastiveNormalization(6, gauss_kernel))

    par:add(func_y)
    par:add(func_uv)

model:add(par)
model:add(nn.JoinTable(1))


-- 2nd layer
model:add(nn.SpatialConvolutionMap(nn.tables.random(38, 68, 30), 9, 9))
    func = nn.Sequential()
    func:add(nn.ReLU())
    func:add(nn.SpatialContrastiveNormalization(68, gauss_kernel))
    func:add(nn.SpatialAveragePooling(2, 2, 2, 2))
model:add(func)


-- 3rd layer
model:add(nn.View(17824))
model:add(nn.Linear(17824, 2))
model:add(nn.LogSoftMax())

-- loss
loss = nn.ClassNLLCriterion()

print(model)

print(model:forward{torch.Tensor(1, 126, 78), torch.Tensor(2, 126, 78)})--:size())

return model, num_conv

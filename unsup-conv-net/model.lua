require "nn"
require "image"

num_conv = 2

model = nn.Sequential()

-- 1st layer
model:add(nn.SplitTable(1))

par = nn.ParallelTable()

func_y = nn.Sequential()
func_y:add(nn.SpatialConvolution(1, 32, 7, 7))
func_y:add(nn.ReLU())
func_y:add(image.lcn())
func_y:add(nn.SpatialSubSampling(32, 3, 3, 3, 3))

func_uv = nn.Sequential()
func_uv:add(nn.SpatialSubSampling(1, 3, 3, 3, 3))
func_uv:add(nn.SpatialConvolution(1, 6, 5, 5))
func_uv:add(nn.ReLU())
func_uv:add(image.lcn())

par:add(func_y)
par:add(func_uv)
par:add(func_uv)

model:add(par)

model:add(nn.JointTable(1))


-- 2nd layer
table = nn.tables.random(38, 68, 2040)
scm = nn.SpatialConvolutionMap(table, 9, 9)
model:add(scm)
func:add(nn.ReLU())
func:add(image.lcn())
func:add(nn.SpatialSubSampling(68, 2, 2, 2, 2))
model:add(func)

-- 3rd layer
model:add(nn.Reshape(17824))
model:add(nn.Linear(17824, 2))
model:add(nn.LogSoftMax())

-- loss
loss = nn.ClassNLLCriterion()


return model, num_conv

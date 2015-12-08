require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 9, 9))
model:add(nn.SpatialAveragePooling(4, 4, 4, 4))

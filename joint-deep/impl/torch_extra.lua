local THNN = require 'nn.THNN'
CustomSpatialConvolution, parent = torch.class('nn.CustomSpatialConvolution', 'nn.Module')

function CustomSpatialConvolution:__init(nInputPlane, nOutputPlane, kWs, kHs, dW, dH, padW, padH)
    parent.__init(self)

    dW = dW or 1
    dH = dH or 1

    self.nInputPlane = nInputPlane
    self.nOutputPlane = nOutputPlane
    self.kWs = kWs
    self.kHs = kHs

    self.dW = dW
    self.dH = dH
    self.padW = padW or 0
    self.padH = padH or self.padW

    local weightSize = 0
    for p = 1, self.nOutputPlane do
        weightSize = weightSize + kWs[p] * kHs[p]
    end
    self.weight = torch.Tensor(self.nOutputPlane, self.nInputPlane * weightSize)
    self.bias = torch.Tensor(nOutputPlane)
    self.gradWeight = torch.Tensor(self.nOutputPlane, self.nInputPlane * weightSize)
    self.gradBias = torch.Tensor(nOutputPlane)

    self.output = nil

    self:reset()
end

function CustomSpatialConvolution:noBias()
    self.bias = nil
    self.gradBias = nil
    return self
end

function CustomSpatialConvolution:reset(stdv)
    for p = 1, self.nOutputPlane do
        if stdv then
            stdvp = stdv[p] * math.sqrt(3)
        else
            stdvp = 1/math.sqrt(self.kWs[p]*self.kHs[p]*self.nInputPlane)
        end
        if nn.oldSeed then
            self.weight[p]:apply(function()
                return torch.uniform(-stdvp, stdvp)
            end)
            if self.bias then
                self.bias:apply(function()
                return torch.uniform(-stdvp, stdvp)
                end)
            end
        else
            self.weight[p]:uniform(-stdvp, stdvp)
            if self.bias then
                self.bias:uniform(-stdvp, stdvp)
            end
        end
    end
end

local function backCompatibility(self, input)
    if not self.finput then
        self.finput = {}
        for p = 1, self.nOutputPlane do
            self.finput[p] = torch.Tensor(
                input:size(1),
                self.nInputPlane * self.kWs[p] * self.kHs[p],
                (input:size(3) - self.kHs[p]) * (input:size(4) - self.kWs[p])
            )
        end
    end

    if not self.fgradInput then
        self.fgradInput = {}
        for p = 1, self.nOutputPlane do
            self.fgradInput[p] = torch.Tensor(
                input:size(1),
                self.nInputPlane * self.kWs[p] * self.kHs[p],
                (input:size(3) - self.kHs[p]) * (input:size(4) - self.kWs[p])
            )
        end
    end
end

local function makeContiguous(self, input, gradOutput)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:resizeAs(input):copy(input)
        input = self._input
    end
    for p = 1, self.nOutputPlane do
        if gradOutput and gradOutput[p] then
            if not gradOutput[p]:isContiguous() then
                self._gradOutput[p] = self._gradOutput[p] or gradOutput[p].new()
                self._gradOutput[p]:resizeAs(gradOutput[p]):copy(gradOutput[p])
                gradOutput[p] = self._gradOutput[p]
            end
        end
    end
    return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
    for p = 1, self.nOutputPlane do
        self.weight[p] = self.weight[p]:view(1, self.nInputPlane * self.kHs[p] * self.kWs[p])
        if self.gradWeight[p] and self.gradWeight[p]:dim() > 0 then
            self.gradWeight[p] = self.gradWeight[p]:view(1, self.nInputPlane * self.kHs[p] * self.kWs[p])
        end
    end
end

local function unviewWeight(self)
    for p = 1, self.nOutputPlane do
        self.weight[p] = self.weight[p]:view(1, self.nInputPlane, self.kHs[p], self.kWs[p])
        if self.gradWeight[p] and self.gradWeight[p]:dim() > 0 then
            self.gradWeight[p] = self.gradWeight[p]:view(1, self.nInputPlane, self.kHs[p], self.kWs[p])
        end
    end
end

local function getWeightIndexes(self, pidx)
    offset = 0
    for p = 1, pidx - 1 do
        offset = offset + self.kWs[p] * self.kHs[p]
    end
    offset = offset * self.nInputPlane
    fstIdx = offset + 1
    sndIdx = offset + self.kWs[pidx] * self.kHs[pidx] * self.nInputPlane
    return fstIdx, sndIdx
end

function CustomSpatialConvolution:updateOutput(input)
    assert(input.THNN, torch.type(input)..'.THNN backend not imported')
    backCompatibility(self, input)
    -- viewWeight(self)
    -- input = makeContiguous(self, input)
    if not self.output then
        self.output = {}
        for p = 1, self.nOutputPlane do
            local h = input:size(3) - self.kHs[p] + 1
            local w = input:size(4) - self.kWs[p] + 1
            self.output[p] = torch.Tensor(input:size(1), 1, h, w)
        end
    end
    for p = 1, self.nOutputPlane do
        local fstIdx, sndIdx = getWeightIndexes(self, p)
        local w = self.weight[{p, {fstIdx, sndIdx}}]:view(1, sndIdx - fstIdx + 1)
        local b = torch.Tensor(1):fill(self.bias[p])
        input.THNN.SpatialConvolutionMM_updateOutput(
            input:cdata(),
            self.output[p]:cdata(),
            w:cdata(),
            THNN.optionalTensor(b),
            self.finput[p]:cdata(),
            self.fgradInput[p]:cdata(),
            self.kWs[p], self.kHs[p],
            self.dW, self.dH,
            self.padW, self.padH
        )    
    end
    -- unviewWeight(self)
    return self.output
end

function CustomSpatialConvolution:updateGradInput(input, gradOutput)
    assert(input.THNN, torch.type(input)..'.THNN backend not imported')
    self.gradInput = torch.Tensor(input:size(1), input:size(2), input:size(3), input:size(4)):fill(0.0)
    -- self.gradInput = torch.Tensor()
    backCompatibility(self, input)
    -- viewWeight(self)
    -- input, gradOutput = makeContiguous(self, input, gradOutput)
    local tmpGradInput = torch.Tensor()
    for p = 1, self.nOutputPlane do
        local fstIdx, sndIdx = getWeightIndexes(self, p)
        local w = self.weight[{p, {fstIdx, sndIdx}}]:view(1, sndIdx - fstIdx + 1)
        input.THNN.SpatialConvolutionMM_updateGradInput(
            input:cdata(),
            gradOutput[p]:cdata(),
            tmpGradInput:cdata(),
            w:cdata(),
            self.finput[p]:cdata(),
            self.fgradInput[p]:cdata(),
            self.kWs[p], self.kHs[p],
            self.dW, self.dH,
            self.padW, self.padH
        )
        self.gradInput = self.gradInput + tmpGradInput
    end
    -- unviewWeight(self)
    return self.gradInput
end

function CustomSpatialConvolution:accGradParameters(input, gradOutput, scale)
    assert(input.THNN, torch.type(input)..'.THNN backend not imported')
    scale = scale or 1
    backCompatibility(self, input)
    -- input, gradOutput = makeContiguous(self, input, gradOutput)
    -- viewWeight(self)
    for p = 1, self.nOutputPlane do
        local fstIdx, sndIdx = getWeightIndexes(self, p)
        local gradW = self.gradWeight[{p, {fstIdx, sndIdx}}]:view(1, sndIdx - fstIdx + 1)
        local gB = torch.Tensor(1)
        input.THNN.SpatialConvolutionMM_accGradParameters(
            input:cdata(),
            gradOutput[p]:cdata(),
            gradW:cdata(),
            THNN.optionalTensor(gB),
            self.finput[p]:cdata(),
            self.fgradInput[p]:cdata(),
            self.kWs[p], self.kHs[p],
            self.dW, self.dH,
            self.padW, self.padH,
            scale
        )
        self.gradBias[p] = gB[1]
    end
    -- unviewWeight(self)
end

function CustomSpatialConvolution:type(type,tensorCache)
    self.finput = self.finput and torch.Tensor()
    self.fgradInput = self.fgradInput and torch.Tensor()
    return parent.type(self,type,tensorCache)
end

function CustomSpatialConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
            self.nInputPlane, self.nOutputPlane, 0, 0)
    if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
      s = s .. string.format(', %d,%d', self.dW, self.dH)
    end
    if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ', ' .. self.padW .. ',' .. self.padH
    end
    if self.bias then
        return s .. ')'
    else
        return s .. ') without bias'
    end
end

function CustomSpatialConvolution:clearState()
    nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
    return parent.clearState(self)
end

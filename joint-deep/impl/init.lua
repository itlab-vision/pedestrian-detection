require 'torch'

local init = {}

init.epoches_num = 1
init.batch_size = 50
init.learning_rate = 0.025

init.start_rows = {1, 1, 4, 4, 9, 9, 1, 1, 1, 4, 4, 4, 9, 1, 1, 1, 1, -1, 1, 1 }
init.end_rows = {3, 3, 9, 9, 15, 15, 3, 9, 9, 9, 15, 15, 15, 3, 9, 15, 15, 17, 15, 15 }
init.start_cols = {1, 3, 1, 3, 2, 4, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 1 }
init.end_cols = {3, 5, 3, 5, 3, 5, 5, 2, 5, 5, 2, 5, 5, 5, 5, 2, 5, 5, 5, 5 }

init.connections1 = torch.Tensor(
{
    { 1, 1, 0, 0, 0, 0 },
    { 1, 0, 1, 0, 0, 0 },
    { 0, 1, 0, 1, 0, 0 },
    { 0, 0, 1, 1, 0, 0 },
    { 0, 0, 1, 0, 1, 0 },
    { 0, 0, 0, 1, 0, 1 },
    { 0, 0, 0, 0, 1, 1 }
})
init.connections2 = torch.Tensor(
{
    { 1, 0, 0, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 0, 0, 0 },
    { 0, 1, 0, 0, 1, 0, 0 },
    { 0, 0, 1, 0, 0, 1, 0 },
    { 1, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 0, 0, 1, 0, 0 },
    { 1, 0, 1, 0, 0, 1, 0 }
})

rand_val = 0.5
init.w1 = torch.Tensor(7, 14)
init.w1[{{1, 6}, {1, 7}}] = rand_val * torch.rand(6, 7)
init.w1[{{7}, {1, 7}}] = rand_val * torch.randn(1, 7)
init.w1[{{}, {8, 14}}] = 0.1 * torch.randn(7, 7)
init.w2 = torch.Tensor(15, 14)
init.w2[{{1, 7}, {1, 7}}] = rand_val * torch.rand(7, 7)
init.w2[{{8, 14}, {1, 7}}] = 0.1 * torch.randn(7, 7)
init.w2[{{15}, {1, 7}}] = rand_val * torch.rand(1, 7)
init.w2[{{}, {8, 14}}] = 0.1 * torch.randn(15, 7)
init.w_class = torch.Tensor(15, 2)
init.w_class[{{1, 7}, {}}] = rand_val * torch.rand(7, 2)
init.w_class[{{8, 14}, {}}] = 0.1 * torch.randn(7, 2)
init.w_class[{{15}, {}}] = rand_val * torch.rand(1, 2)
init.w_class[{{}, {2}}] = -init.w_class[{{}, {1}}]
init.c1 = torch.Tensor(14)
init.c1[{{1, 7}}] = 0.1 * torch.randn(7)
init.c1[{{8, 14}}] = torch.Tensor(7):fill(0)
init.c2 = torch.Tensor(14)
init.c2[{{1, 7}}] = 0.1 * torch.randn(7)
init.c2[{{8, 14}}] = torch.Tensor(7):fill(0)
init.b4 = rand_val * torch.rand(#init.start_rows)

return init

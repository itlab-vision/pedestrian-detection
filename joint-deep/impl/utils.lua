local utils = {}

function utils.avoid_nans(t)
    nan_mask = t:ne(t)
    t[nan_mask] = 0
    return t
end

return utils

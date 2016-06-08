require 'lfs'

local utils = {}

function utils.avoid_nans(t)
    nan_mask = t:ne(t)
    t[nan_mask] = 0
    return t
end

function utils.file_exists(fname)
    local f = io.open(fname,"r")
    if f ~= nil then
        io.close(f)
        return true
    else
        return false
    end
end

function utils.dir_exists(dirname)
    if lfs.attributes(dirname:gsub("\\$",""),"mode") == "directory" then
        return true
    else
        return false
    end
end

return utils

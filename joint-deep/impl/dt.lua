local dt = {}

function dt.dt1d_by_column(src, dst, iy, col_id, a, b)
    local n = src:size()[1]
    local v = {}
    local z = {}
    local k = 0
    v[1] = 0
    z[1] = -math.huge
    z[2] = math.huge
    for q = 1, n - 1 do
        local s = ((src[q+1][col_id] - src[v[k+1]+1][col_id]) - b*(q - v[k+1]) + a*(q*q - v[k+1]*v[k+1])) / (2*a*(q - v[k+1]))
        while s <= z[k+1] do
            k = k - 1
            s = ((src[q+1][col_id] - src[v[k+1]+1][col_id]) - b*(q - v[k+1]) + a*(q*q - v[k+1]*v[k+1])) / (2*a*(q - v[k+1]))
        end
        k = k + 1
        v[k + 1] = q
        z[k + 1] = s
        z[k + 2] = math.huge
    end

    k = 0
    for q = 0, n - 1 do
        while z[k + 2] < q do
            k = k + 1
        end
        dst[q + 1][col_id] = a*(q - v[k + 1])*(q - v[k + 1]) + b*(q - v[k + 1]) + src[v[k + 1]+1][col_id]
        iy[q + 1][col_id] = v[k + 1]
    end
end

function dt.dt1d_by_row(src, dst, iy, row_id, a, b)
    local n = src:size()[2]
    local v = {}
    local z = {}
    local k = 0
    v[1] = 0
    z[1] = -math.huge
    z[2] = math.huge
    for q = 1, n - 1 do
        local s = ((src[row_id][q+1] - src[row_id][v[k+1]+1]) - b*(q - v[k+1]) + a*(q*q - v[k+1]*v[k+1])) / (2*a*(q - v[k+1]))
        while s <= z[k+1] do
            k = k - 1
            s = ((src[row_id][q+1] - src[row_id][v[k+1]+1]) - b*(q - v[k+1]) + a*(q*q - v[k+1]*v[k+1])) / (2*a*(q - v[k+1]))
        end
        k = k + 1
        v[k + 1] = q
        z[k + 1] = s
        z[k + 2] = math.huge
    end

    k = 0
    for q = 0, n - 1 do
        while z[k + 2] < q do
            k = k + 1
        end
        dst[row_id][q + 1] = a*(q - v[k + 1])*(q - v[k + 1]) + b*(q - v[k + 1]) + src[row_id][v[k + 1]+1]
        iy[row_id][q + 1] = v[k + 1]
    end
end

return dt

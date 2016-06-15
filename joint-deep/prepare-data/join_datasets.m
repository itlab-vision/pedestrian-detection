function join_datasets(caltech_path, inria_path, kitti_path_splitted, outpath)
% function join_datasets(caltech_path, inria_path, outpath)
% function join_datasets(caltech_path, kitti_path_splitted, outpath)
% function join_datasets(inria_path, kitti_path_splitted, outpath)
out_train_x = cell(1, 3);
out_train_y = [];

saved_count = 0;
load(caltech_path, 'train_x', 'train_y')
for j = 1 : length(train_x)
    s = size(train_x{j}, 3);
    out_train_x{j}(:, :, 1 : s) = train_x{j};
end;
out_train_y(:, 1 : size(train_y, 2)) = train_y;
saved_count = saved_count + s;
clear('train_x');
clear('train_y');
display('Appended');

load(inria_path, 'train_x', 'train_y')
for j = 1 : length(train_x)
    s = size(train_x{j}, 3);
    out_train_x{j}(:, :, saved_count + 1 : saved_count + s) = train_x{j};
end;
out_train_y(:, saved_count + 1 : saved_count + size(train_y, 2)) = train_y;
saved_count = saved_count + s;
clear('train_x');
clear('train_y');
display('Appended');

data_names = dir(kitti_path_splitted);
for d = 3 : length(data_names)
    load([kitti_path_splitted '\' data_names(d).name], 'new_train_x', 'new_train_y');
    for j = 1 : length(new_train_x)
        s = size(new_train_x{j}, 3);
        out_train_x{j}(:, :, saved_count + 1 : saved_count + s) = new_train_x{j};
    end;
    out_train_y(:, saved_count + 1 : saved_count + size(new_train_y, 2)) = new_train_y;
    saved_count = saved_count + s;
    clear('new_train_x');
    clear('new_train_y');
end;

display('Appended');

display(size(out_train_x));
display(size(out_train_x{1}));
display(size(out_train_y));

save(outpath, '-v7.3', 'out_train_x', 'out_train_y');
function split_train_data(datapath, dataname)
load([datapath '\' dataname], 'train_x', 'train_y');
outdir = [datapath '\splitted'];
mkdir(outdir);
split_count = 1000;
data_count = length(train_x{1});
for i = 1 : split_count : data_count
    fname = ['SplittedData' int2str(i)];
    count = min(data_count - i, split_count);
    new_train_x{1} = train_x{1}(:, :, i : i + count - 1);
    new_train_x{2} = train_x{2}(:, :, i : i + count - 1);
    new_train_x{3} = train_x{3}(:, :, i : i + count - 1);
    new_train_y = train_y(:, i : i + count - 1);
    save([outdir '\' fname], 'new_train_x', 'new_train_y');
end;
end
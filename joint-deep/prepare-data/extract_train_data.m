function extract_train_data(dataset_paths)
for p = 1 : length(dataset_paths)
    dpath = dataset_paths{p}.path;
    dname = dataset_paths{p}.name;
    out_pos_path = [dpath '/images/train/pos'];
    out_neg_path = [dpath '/images/train/neg'];
    load([dpath '/' dname], 'train_x', 'train_y');
    mkdir(out_pos_path);
    mkdir(out_neg_path);
    
    for i = 1:size(train_x{1}, 3)
        im = zeros([size(train_x{1}, 1) size(train_x{1}, 2) 3]);
        im(:, :, 1) = train_x{1}(:, :, i);
        im(:, :, 2) = train_x{2}(:, :, i);
        im(:, :, 3) = train_x{3}(:, :, i);
        fname = ['im' int2str(i) '.png'];
        if (train_y(1, i) == 1)
            imwrite(im, [out_pos_path '\' fname]);
        else
            imwrite(im, [out_neg_path '\' fname]);
        end;
    end;
    
    clear('train_x');
    clear('train_y');
end;
end
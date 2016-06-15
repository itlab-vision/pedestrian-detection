function save_samples_to_mat(dataset_paths, out_path, out_data_filename)
samples_count = 0;
for p = 1 : length(dataset_paths)
    pos_path = dataset_paths{p}.pos_path;
    neg_path = dataset_paths{p}.neg_path;

    pos_list_fnames = dir(pos_path);
    neg_list_fnames = dir(neg_path);
    pos_count = length(pos_list_fnames) - 2;
    neg_count = length(neg_list_fnames) - 2;
    display(pos_count);
    display(neg_count);
    for i = 1:pos_count
        img = imread([pos_path '\' pos_list_fnames(i + 2).name]);
        train_x{1}(:, :, samples_count + i) = img(:, :, 1);
        train_x{2}(:, :, samples_count + i) = img(:, :, 2);
        train_x{3}(:, :, samples_count + i) = img(:, :, 3);
        train_y(:, samples_count + i) = [1; 0];
    end;

    samples_count = samples_count + pos_count;

    for i = 1:neg_count
        train_x{1}(:, :, samples_count + i) = img(:, :, 1);
        train_x{2}(:, :, samples_count + i) = img(:, :, 2);
        train_x{3}(:, :, samples_count + i) = img(:, :, 3);
        train_y(:, samples_count + i) = [0; 1];
    end;

    samples_count = samples_count + neg_count;
    
    display('Appended');
end;

save([out_path '\' out_data_filename], 'train_x', 'train_y');
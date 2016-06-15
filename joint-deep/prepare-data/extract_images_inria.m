function extract_images_inria(data_path, data_name)
load([data_path data_name], 'train_x', 'train_y');
load([data_path data_name], 'test_x', 'test_y');
pos_train_output_path = [data_path 'images\train\pos'];
neg_train_output_path = [data_path 'images\train\neg'];
pos_test_output_path = [data_path 'images\test\pos'];
neg_test_output_path = [data_path 'images\test\neg'];
mkdir(pos_train_output_path);
mkdir(neg_train_output_path);
mkdir(pos_test_output_path);
mkdir(neg_test_output_path);

for i = 1:size(train_x{1}, 3)
    im = zeros([size(train_x{1}, 1) size(train_x{1}, 2) 3]);
    im(:, :, 1) = train_x{1}(:, :, i);
    im(:, :, 2) = train_x{2}(:, :, i);
    im(:, :, 3) = train_x{3}(:, :, i);
    fname = ['im' int2str(i) '.png'];
    if (train_y(1, i) == 1)
        imwrite(im, [pos_train_output_path '\' fname]);
    else
        imwrite(im, [neg_train_output_path '\' fname]);
    end;
end;

for i = 1:size(test_x{1}, 3)
    im = zeros([size(test_x{1}, 1) size(test_x{1}, 2) 3]);
    im(:, :, 1) = test_x{1}(:, :, i);
    im(:, :, 2) = test_x{2}(:, :, i);
    im(:, :, 3) = test_x{3}(:, :, i);
    fname = ['im' int2str(i) '.png'];
    if (test_y(1, i) == 1)
        imwrite(im, [pos_test_output_path '\' fname]);
    else
        imwrite(im, [neg_test_output_path '\' fname]);
    end;
end;
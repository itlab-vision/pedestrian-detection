function write_data_to_file(input_path_file, output_path_dir)
    load(input_path_file, 'train_x', 'train_y', 'test_x', 'test_y');
    train_x_path_dir = [output_path_dir '/train_x'];
    train_y_path = [output_path_dir '/train_y.txt'];
    test_x_path_dir = [output_path_dir '/test_x'];
    test_y_path = [output_path_dir '/test_y.txt'];
    
    chunk = 250;
    
    if (~isdir(train_x_path_dir))
        mkdir(train_x_path_dir);
    end
    if (~isdir(test_x_path_dir))
        mkdir(test_x_path_dir);
    end
    
    train_x_path = [train_x_path_dir '/train_x_0.txt'];
    train_x_file = fopen(train_x_path, 'wt');
    fprintf(train_x_file, 'train_x = {\n');
    for i = 1 : size(train_x{1}, 3)
        fprintf(train_x_file, '{');
        for j = 1 : numel(train_x)
            fprintf(train_x_file, '{');
            img = train_x{j}(:, :, i);
            for k = 1 : size(img, 1)
                fprintf(train_x_file, '{');
                data_line = img(k, :);
                for d = 1 : size(data_line, 2)
                    if (d ~= size(data_line, 2))
                        fprintf(train_x_file, '%f, ', data_line(1, d));
                    else
                        fprintf(train_x_file, '%f', data_line(size(data_line, 2)));
                    end
                end
                if (k ~= size(img, 1))
                    fprintf(train_x_file, '},\n');
                else
                    fprintf(train_x_file, '}');
                end
            end
            if (j ~= numel(train_x))
                fprintf(train_x_file, '},\n');
            else
                fprintf(train_x_file, '}');
            end
        end
        if mod(i, chunk) ~= 0
            fprintf(train_x_file, '},\n');
        else
            fprintf(train_x_file, '}}');
            fclose(train_x_file);
            train_x_path = [train_x_path_dir '/train_x_' int2str(int32(floor(i / chunk))) '.txt'];
            train_x_file = fopen(train_x_path, 'wt');
            fprintf(train_x_file, 'train_x = {\n');
        end
    end
    if (mod(size(train_x{1}, 3), chunk) ~= 0)
        fprintf(train_x_file, '}}');
        fclose(train_x_file);
    end
    
    train_y_file = fopen(train_y_path, 'wt');
    fprintf(train_y_file, 'train_y = {\n');
    for i = 1 : size(train_y, 2)
        data_line = train_y(:, i);
        if (i ~= size(train_y, 2))
            fprintf(train_y_file, '{%d, %d},\n', data_line(1, 1), data_line(2, 1));
        else
            fprintf(train_y_file, '{%d, %d}\n', data_line(1, 1), data_line(2, 1));
            fprintf(train_y_file, '}');
        end
    end
    fclose(train_y_file);
    
    
    
    
    test_x_path = [test_x_path_dir '/test_x_0.txt'];
    test_x_file = fopen(test_x_path, 'wt');
    fprintf(test_x_file, 'test_x = {\n');
    for i = 1 : size(test_x{1}, 3)
        fprintf(test_x_file, '{');
        for j = 1 : numel(test_x)
            fprintf(test_x_file, '{');
            img = test_x{j}(:, :, i);
            for k = 1 : size(img, 1)
                fprintf(test_x_file, '{');
                data_line = img(k, :);
                for d = 1 : size(data_line, 2)
                    if (d ~= size(data_line, 2))
                        fprintf(test_x_file, '%f, ', data_line(1, d));
                    else
                        fprintf(test_x_file, '%f', data_line(size(data_line, 2)));
                    end
                end
                if (k ~= size(img, 1))
                    fprintf(test_x_file, '},\n');
                else
                    fprintf(test_x_file, '}');
                end
            end
            if (j ~= numel(test_x))
                fprintf(test_x_file, '},\n');
            else
                fprintf(test_x_file, '}');
            end
        end
        if mod(i, chunk) ~= 0
            fprintf(test_x_file, '},\n');
        else
            fprintf(test_x_file, '}}');
            fclose(test_x_file);
            test_x_path = [test_x_path_dir '/test_x_' int2str(int32(floor(i / chunk))) '.txt'];
            test_x_file = fopen(test_x_path, 'wt');
            fprintf(test_x_file, 'test_x = {\n');
        end
    end
    if (mod(size(test_x{1}, 3), chunk) ~= 0)
        fprintf(test_x_file, '}}');
        fclose(test_x_file);
    end
    
    test_y_file = fopen(test_y_path, 'wt');
    fprintf(test_y_file, 'test_y = {\n');
    for i = 1 : size(test_y, 2)
        data_line = test_y(:, i);
        if (i ~= size(test_y, 2))
            fprintf(test_y_file, '{%d, %d},\n', data_line(1, 1), data_line(2, 1));
        else
            fprintf(test_y_file, '{%d, %d}\n', data_line(1, 1), data_line(2, 1));
            fprintf(test_y_file, '}');
        end
    end
    fclose(test_y_file);
end

function test(test_out_path, train_data_name, test_data_name)
    close all;
    if ~exist('Pathadd', 'var')
        addpath ../../util
        addpath ../../tmptoolbox/matlab
        addpath ../../tmptoolbox/classify
        addpath ../../tmptoolbox
        addpath ../../tmptoolbox/images
        addpath ../../dbEval
        Pathadd = 1;
    end;
    
    out{1} = [];
    fi = fopen(test_out_path, 'rt');
    out{1} = fscanf(fi, '%f, %f', [2 inf]);
    fclose(fi);
    
    model_name = ['Torch-JointDeep-' train_data_name];
    
    if strcmp(test_data_name, 'CaltechTest')
        ReaderDataFName = ['../../data/CaltechTrain/CNNDLTData3Color63_4.mat'];
        load(ReaderDataFName, 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame');
        res_dir{1} = ['../../dbEval/data-USA/res/' model_name '/'];
        testCNN_CaltechTest(out, Test_Boxes, Test_Frame, res_dir);
    elseif strcmp(test_data_name, 'ETH')
        ReaderDataFName = ['../../data/INRIATrain/CNNDLTData3Color63_4HOGcssNoPosNeg.mat'];
        load(ReaderDataFName, 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame');
        res_dir{1} = ['../../dbEval/data-ETH/res/' model_name '/'];
        testCNN_ETH(out, Test_Boxes, Test_Frame, res_dir);
    end
end
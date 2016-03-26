function save_samples_to_mat(pos_path, neg_path, out_path, out_data_filename, out_labels_filename)
pos_list_fnames = dir(pos_path);
neg_list_fnames = dir(neg_path);
pos_count = length(pos_list_fnames) - 2;
neg_count = length(neg_list_fnames) - 2;
for i = 1:pos_count
    img = imread([pos_path '\' pos_list_fnames(i + 2).name]);
    AllimBoxesBeforeNmsRsz{i}{1}.im = img;
    AllimBoxesBeforeNmsRsz{i}{1}.score = -1.0;
    Labels{i} = 1;
end;

for i = 1:neg_count
    img = imread([neg_path '\' neg_list_fnames(i + 2).name]);
    AllimBoxesBeforeNmsRsz{pos_count + i}{1}.im = img;
    AllimBoxesBeforeNmsRsz{pos_count + i}{1}.score = -1.0;
    Labels{pos_count + i} = -1;
end;

save([out_path '\' out_data_filename], 'AllimBoxesBeforeNmsRsz');
save([out_path '\' out_labels_filename], 'Labels');
function get_images(data_path, data_name, labels_name)
load([data_path data_name], 'AllimBoxesBeforeNmsRsz');
load([data_path labels_name], 'Labels')
pos_output_path = [data_path 'images/pos'];
neg_output_path = [data_path 'images/neg'];
mkdir(pos_output_path);
mkdir(neg_output_path);
for i = 1:length(AllimBoxesBeforeNmsRsz)
    for j = 1:length(AllimBoxesBeforeNmsRsz{i})
        fname = [int2str(i) '_' int2str(j) '.png'];
        if (Labels{i}(j) > 0)
            imwrite(AllimBoxesBeforeNmsRsz{i}{j}.im, [pos_output_path '/' fname]);
        elseif (Labels{i}(j) < 0)
            imwrite(AllimBoxesBeforeNmsRsz{i}{j}.im, [neg_output_path '/' fname]);
        end;
    end;
end;

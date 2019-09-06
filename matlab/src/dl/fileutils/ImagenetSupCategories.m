function [] = ImagenetSupCategories(path)

% '/mnt/hdd/Dropbox/Tmp/ILSVRC2012_validation_meta.mat'
load(path)


for i = 1001:length(synsets)
    current_childs = sub_children(synsets, synsets(i).children);
    
    [~, idx] = unique(current_childs);
    current_childs = current_childs(idx,:);
    
    if length(current_childs) > 1000
        disp('Malamente!')
    end
    
    fid = fopen(['/home/arash/Desktop/super_cats/cat_', num2str(i), '_', synsets(i).WNID, '.txt'], 'w');
    for row = 1:length(current_childs)
        fprintf(fid, '%s', current_childs{row});
        if row < length(current_childs)
            fprintf(fid, '\n');
        end
    end
    fclose(fid);
end

end


function children_wind = sub_children(all_list, children_ind)

children_wind = {};

for child = children_ind
    if child < 1001
        children_wind = [children_wind; all_list(child).WNID];
    else
        children_wind = [children_wind; sub_children(all_list, all_list(child).children)];
    end
end

end


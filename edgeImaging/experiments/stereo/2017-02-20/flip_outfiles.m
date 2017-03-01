
% resfolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb20/results';
resfolder  = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20/results';

all_files = dir(resfolder);


for i = 1:length(all_files)
    fname = fullfile(resfolder, all_files(i).name);
    [path, name, ext] = fileparts(fname);
    if ~strcmp(ext, '.mat')
        continue
    end
    load(fname);
    name
    
    if sign(left_angles(2) - left_angles(1)) ~=...
            sign(left_floor_angles(2) - left_floor_angles(1))
        % messed up, unique sorted angles in increasing order
        % flip all of the angles and scenes
        fprintf('flipping floors and scenes\n');
        left_floor = fliplr(left_floor);
        left_scene = fliplr(left_scene);
        right_floor = fliplr(right_floor);
        right_scene = fliplr(right_scene);
    end
            
    fname = strcat('left_floor', name(4:end), '.png')
    imwrite((left_floor+0.05)/0.1, fullfile(resfolder, fname));

    fname = strcat('left_scene', name(4:end), '.png')
    imwrite((left_scene+0.05)/0.1, fullfile(resfolder, fname));

    fname = strcat('right_floor', name(4:end), '.png')
    imwrite((right_floor+0.05)/0.1, fullfile(resfolder, fname));

    fname = strcat('right_scene', name(4:end), '.png')
    imwrite((right_scene+0.05)/0.1, fullfile(resfolder, fname));
end
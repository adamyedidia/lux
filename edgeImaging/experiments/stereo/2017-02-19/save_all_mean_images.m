addpath(genpath('../../../corner_cam'));

datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb20/experiments';

all_files = dir(datafolder);

for i = 3:length(all_files)
    vid = all_files(i).name;
    if ~strcmp(vid(1:4), 'cali')
        fname = fullfile(datafolder, vid);
        fprintf('computing mean for %s\n', fname);
        saveMeanImage(0, fname);
    end
end

addpath(genpath('../../../corner_cam'));

clear; close all;

% datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20';
datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb20';
expfolder = fullfile(datafolder, 'experiments');
gridfile = fullfile(expfolder, 'calibrationgrid.MP4');
moviefile = fullfile(expfolder, 'red12_walking.MP4');

resfolder = fullfile(datafolder, 'results');
outfile = fullfile(resfolder, 'out_red12_walking.mat');

ncorners = 4; % total ncorners in the scene
% corner_idx = [1, 2, 3, 4];
corner_idx = [2];

theta_lims{1} = [pi/2, 0]; % top left
theta_lims{2} = [pi, pi/2]; % bottom left
theta_lims{3} = [0, pi/2]; % bottom right
theta_lims{4} = [pi/2, pi]; % top right

direction{1} = -1; 
direction{2} = -1; 
direction{3} = 1; 
direction{4} = 1; 

params = initParams(moviefile, gridfile, ncorners, corner_idx);
% params.sub_mean = 1;
params.endframe = params.endframe/8;
params.inf_method = 'spatial_smoothing';
params.amat_method = 'allpix';
% corners = params.corner;

for i = 1:length(corner_idx)
    c = corner_idx(i);
%     params.corner = corners(c,:);
    params.theta_lim = theta_lims{c};
    outframe = doCornerRecon(params, moviefile);
    outframe = outframe(:,2:end,:); % throwing away the constant light
    figure; imagesc(outframe);

    if direction{c} == -1
        outframe = fliplr(outframe); 
    end

    outframes{i} = outframe;
end

% save(outfile);

addpath(genpath('../../../corner_cam'));

clear; close all;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb20';
% datafolder = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb20';
expfolder = fullfile(datafolder, 'experiments');
gridfile = fullfile(expfolder, 'calibrationgrid.MP4');
moviefile = fullfile(expfolder, 'red12_walking.MP4');

resfolder = fullfile(datafolder, 'results');
outfile = fullfile(resfolder, 'out_red12_walking.mat');

ncorners = 4; % total ncorners in the scene
corner_idx = [1, 2, 3, 4];
corner_idx = [2];

theta_lims{1} = [pi/2, 0]; % top left
theta_lims{2} = [pi, pi/2]; % bottom left
theta_lims{3} = [0, pi/2]; % bottom right
theta_lims{4} = [pi/2, pi]; % top right

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.sub_mean = 1;
params.endframe = params.endframe/3;
params.inf_method = 'spatial_smoothing';
params.amat_method = 'interp';

plotting = 1;
corners = params.corner;

for i = 1:length(corner_idx)
    c = corner_idx(i);
    params.corner = corners(i,:);
    params.theta_lim = theta_lims{c};
    outframe = doCornerRecon(params, moviefile);
    if params.sub_mean
        multiplier = 30;
    else
        multiplier = 1;
    end
    
    if plotting
        figure; 
        imagesc(multiplier * outframe);
        title(sprintf('corner %d, unflipped', c));
    end

    outframes{c} = outframe;
end
 
left_floor = [outframes{2}, outframes{1}];
right_floor = [fliplr(outframes{4}), fliplr(outframes{3})]; 
 
left_scene = [fliplr(outframes{1}), fliplr(outframes{2})];
right_scene = [outframes{3}, outframes{4}];

% save(outfile);

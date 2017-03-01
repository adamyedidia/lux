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
corner_idx = [1, 2];

theta_lims{1} = [pi/2, 0]; % top left
theta_lims{2} = [pi, pi/2]; % bottom left
theta_lims{3} = [0, pi/2]; % bottom right
theta_lims{4} = [pi/2, pi]; % top right

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.sub_mean = 1;
params.endframe = params.endframe/6;
params.inf_method = 'spatial_smoothing';
params.amat_method = 'interp';

plotting = 1;
corners = params.corner;

outframes = cell(size(corner_idx));
all_angles = cell(size(corner_idx));

for i = 1:length(corner_idx)
    c = corner_idx(i);
    params.corner = corners(i,:);
    params.theta_lim = theta_lims{c};
    [outframe, angles] = doCornerRecon(params, moviefile);
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
    all_angles{c} = reshape(angles, [1, length(angles)]); % make row vector
end

left_angles = [all_angles{2}, all_angles{1}];
left_floor = [outframes{2}, outframes{1}];
left_floor = avgRepeatColumns(left_floor, left_angles);

right_angles = [fliplr(all_angles{4}), fliplr(all_angles{3})]; 
right_floor = [fliplr(outframes{4}), fliplr(outframes{3})]; 
right_floor = avgRepeatColumns(right_floor, right_angles);

left_scene = fliplr(left_floor);
right_scene = fliplr(right_floor);

% save(outfile);

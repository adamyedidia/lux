addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));
addpath(genpath('../corner_cam'));

clear; close all;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb14';
expfolder = fullfile(datafolder, 'experiments');
resfolder = fullfile(datafolder, 'results');
gridfile = fullfile(expfolder, 'loc2_caligrid.MOV');
backfile = fullfile(expfolder, 'loc2_dark.MOV');
moviefile = fullfile(expfolder, 'loc2_dark.MOV');
outfile = fullfile(resfolder, 'out_loc2_dark.MOV');

ncorners = 4; % total corners in the scene
corner_idx = [1]; % corners we care about now

theta_lims{1} = [pi/2, 0]; % top left
theta_lims{2} = [pi, pi/2]; % bottom left
theta_lims{3} = [0, pi/2]; % bottom right
theta_lims{4} = [pi/2, pi]; % top right

direction{1} = -1; 
direction{2} = -1; 
direction{3} = 1; 
direction{4} = 1; 

params = initParams(moviefile, gridfile, ncorners, corner_idx);
params.endframe = params.endframe/3;
params.inf_method = 'spatial_smoothing';
params.amat_method = 'allpix';

corners = params.corner;

load(params.corner_datafile, 'avg_img');
rchan = avg_img(:,:,1);

for i = 1:length(corner_idx)
    c = corner_idx(i);
    params.corner = corners(c,:);
    params.theta_lim = theta_lims{c};
    [~, x0, y0] = allPixelAmat(params.corner,...
        params.outr, params.nsamples, params.theta_lim);
    crop_idx = sub2ind([params.framesize(1), params.framesize(2), 1], y0, x0);
    cropped = rchan(crop_idx);
    [corner_idx, wall_line_theta] = findWall(cropped, params.theta_lim);
    % update our theta and corner
    params.theta_lim(1) = wall_line_theta;
    params.corner = [x0(1,corner_idx(1)), y0(corner_idx(2),1)];

    if 0 % show the new cropped region
       figure; subplot(211); imagesc(cropped);
       [~, x0, y0] = allPixelAmat(params.corner,...
            params.outr, params.nsamples, params.theta_lim);
       crop_idx = sub2ind(params.framesize, y0, x0);
       cropped = rchan(crop_idx);
       subplot(212); imagesc(cropped);
    end
    
    outframe = doCornerRecon(params, moviefile);
    outframe = outframe(:,2:end,:); % throwing away the constant light
    figure; imagesc(outframe);

    if direction{c} == -1
        outframe = fliplr(outframe); 
    end

    outframes{c} = outframe;
end

if length(corner_idx) > 1
    x1_1d = [outframes{1} outframes{2}];
    figure; imagesc(x1_1d);
end

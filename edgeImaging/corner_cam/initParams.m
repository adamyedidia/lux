function params = initParams(moviefile, gridfile, ncorners, corner_idx)
if nargin < 4
    corner_idx = 1;
end

% init paths
[mydir, ~, ~] = fileparts(mfilename('fullpath'));
[edgesdir, ~, ~] = fileparts(mydir);
addpath(genpath(fullfile(edgesdir, 'rectify')));
addpath(genpath(fullfile(edgesdir, 'utils', 'pyr')));

% init params
params.inf_method = 'kalman_smoothing';
params.amat_method = 'interp';
params.nsamples = 200;
params.rs = 4:2:30;
params.outr = 50;
params.theta_lim = [pi/2, 0];

params.lambda = 15; % pixel noise
params.sigma = 1; % prior variance
params.alpha = 5e-3; % process noise
params.eps = 1e-5;

params.sub_background = 0;
params.sub_mean = 0;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 1;
params.start = 120*5;
params.step = 5;

params.minclip = 0;
params.maxclip = 2;

params.cal_datafile = saveCalData(0, gridfile);
params.mean_datafile = saveMeanImage(0, moviefile);
params.corner_datafile = saveCorners(0, moviefile, ncorners, params);

load(params.mean_datafile, 'endframe');
params.endframe = endframe; % put this in params to be modifiable later

load(params.corner_datafile, 'corners', 'avg_img');
params.corner = corners(corner_idx,:);
params.framesize = size(avg_img);
end
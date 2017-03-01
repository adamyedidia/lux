function params = initParams(moviefile, gridfile, ncorners, corner_idx)
addpath(genpath('../utils/pyr'));

if nargin < 4
    corner_idx = 1;
end

params.inf_method = 'spatial_smoothing';
params.amat_method = 'interp';
params.nsamples = 50;
params.rs = 10:2:30;
params.outr = 50;

params.lambda = 15; % pixel noise
params.sigma = 0.4; % prior variance
params.alpha = 5e-3; % process noise
params.eps = 1e-5;

params.sub_background = 0;
params.sub_mean = 1;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 4;
params.start = 60*5;
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



function params = initParams(moviefile, gridfile, ncorners, corner_idx,...
                             cornerfile, default_mean)
if nargin < 6
    default_mean = 1;
end
if nargin < 5
    cornerfile = moviefile;
end
if nargin < 4
    corner_idx = 1;
end

% init paths
rootdir = fileparts(fileparts(mfilename('fullpath')));

addpath(genpath(fullfile(rootdir, 'rectify')));
addpath(genpath(fullfile(rootdir, 'utils', 'pyr')));

% set isvid, frame_rate, maxframe, start, step, navg in params
params = getInputProperties(moviefile);
params.endframe = params.maxframe - params.start;

% set the default inference methods
params.inf_method = 'spatial_smoothing';
params.amat_method = 'allpix';
params.nsamples = 200;
params.rs = 4:2:30;
params.outr = max(params.rs);

% set some inference parameters
params.use_noise_lambda = 0;
params.lambda = 15*4; % pixel noise
params.sigma = 0.4; % prior variance (spatial prior)
params.alpha = 5e-3; % process noise (kalman methods)
params.eps = 0;

params.sub_mean = 1;
params.use_median = 0;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 1;
params.minclip = 0;
params.maxclip = 2;

% compute calibration and mean images
params.cal_datafile = saveCalData(0, params.isvid, gridfile);
if default_mean
    params.mean_datafile = saveMeanImage(0, params.isvid,...
        moviefile, params.start, params.endframe);
end
if params.use_median
    params.median_datafile = saveMedianImage(0, params.isvid,...
        moviefile, params.start, params.endframe);
end
params.corner_datafile = saveCorners(0, params.isvid,...
    cornerfile, ncorners, params);


% set corners and framesize for the downsampling we're using
load(params.corner_datafile, 'corners', 'avg_img');
params.corner = corners(corner_idx,:);
params.framesize = size(avg_img);
end


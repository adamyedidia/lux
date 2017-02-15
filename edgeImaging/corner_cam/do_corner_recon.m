addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));

close all;

%% Parameters and file locations

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_space_all.MOV', resfolder);

params.inf_method = 'kalman_filter';
params.amat_method = 'interp';
params.nsamples = 50;
params.rs = 10:2:30;
params.outr = 50;
params.theta_lim = [pi/2, 0];

params.lambda = 15; % pixel noise
params.sigma = 0.4; % prior variance
params.alpha = 5e-3; % process noise

params.sub_background = 0;
params.sub_mean = 0;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 4;
params.start = 60*5;
params.step = 5;

params.minclip = 0;
params.maxclip = 2;

%% Reconstruction

caldata = saveCalData(0, moviefile, gridfile, backfile, params.start);
load(caldata, 'frame1', 'endframe');

params.endframe = endframe/3;

frame1 = blurDnClr(frame1, params.downlevs, params.filt);
figure; imagesc(frame1(:,:,1));
corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

params.corner = corner;
params.framesize = size(frame1);

% get A matrix
amat_out = getAmat(params);
amat = amat_out{1};
if ~strcmp(params.amat_method, 'interp')   
    params.crop_idx = amat_out{2};
end

% spatial prior
bmat = eye(params.nsamples) - diag(ones([params.nsamples-1,1]), 1);

switch params.inf_method
    case 'spatial_smoothing'
        outframes = spatialSmoothingRecon(moviefile, caldata, params, amat, bmat);
    case 'kalman_filter'
        fmat = eye(params.nsamples);
        outframes = kalmanFilterRecon(moviefile, caldata, params, amat, bmat, fmat);
    case 'kalman_smoothing'
        fmat = eye(params.nsamples);
        outframes = kalmanSmoothingRecon(moviefile, caldata, params, amat, bmat, fmat);
    otherwise % default to naive corner cam
        outframes = cornerRecon(moviefile, caldata, params, amat);
end

figure; imagesc(outframes);

% writeReconVideo(outfile, outframes);
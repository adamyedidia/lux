addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));
addpath(genpath('../corner_cam'));


close all; clear;

ncorners = 2;
theta_lims{1} = [pi/2, 0]; % top left
theta_lims{2} = [pi, pi/2]; % bottom left
theta_lims{3} = [0, pi/2]; % bottom right
theta_lims{4} = [pi/2, pi]; % top right
direction{1} = -1; 
direction{2} = -1; 
direction{3} = 1; 
direction{4} = 1; 

params.inf_method = 'naive';
params.amat_method = 'interp';
params.nsamples = 50;
params.rs = 10:2:30;
params.outr = 40;
params.theta_lim = [pi/2, 0];

params.lambda = 15; % pixel noise
params.sigma = 0.4; % prior variance
params.alpha = 5e-3; % process noise

params.sub_background = 0;
params.sub_mean = 0;
params.downlevs = 2;
params.filt = binomialFilter(5);

params.smooth_up = 1;
params.start = 60*5;
params.step = 5;

params.minclip = 0;
params.maxclip = 2;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Feb14';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/loc2_caligrid.MOV', expfolder);
backfile = sprintf('%s/loc2_dark.MOV', expfolder);
moviefile = sprintf('%s/loc2_dark.MOV', expfolder);
outfile = sprintf('%s/out_loc2_dark.MOV', resfolder);


caldata = saveCalData(0, moviefile, gridfile, backfile, params.start);
load(caldata, 'frame1', 'endframe');

frame1 = blurDnClr(frame1, params.downlevs, params.filt);
rchan = frame1(:,:,1);
params.framesize = size(frame1);
params.endframe = endframe/2;

figure; imagesc(rchan); hold on;
corners = ginput(ncorners);
plot(corners(:,1), corners(:,2), 'ro');

for c = 1:ncorners
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

    if 1 % show the new cropped region
       figure; subplot(211); imagesc(cropped);
       [~, x0, y0] = allPixelAmat(params.corner,...
            params.outr, params.nsamples, params.theta_lim);
       crop_idx = sub2ind(params.framesize, y0, x0);
       cropped = rchan(crop_idx);
       subplot(212); imagesc(cropped);
    end

    amat_out = getAmat(params);
    amat = amat_out{1};

    % spatial prior
    bmat = eye(size(amat,2)) - diag(ones([size(amat,2)-1,1]), 1);
    bmat = bmat(1:end-1,:);
    bmat(1,:) = 0; % don't use the constant light to smooth
    if ~strcmp(params.amat_method, 'interp')   
       params.crop_idx = amat_out{2};
    end

    switch params.inf_method
        case 'spatial_smoothing'
            outframe = spatialSmoothingRecon(moviefile, caldata, params, amat, bmat);
        case 'kalman_filter'        
            fmat = eye(params.nsamples);
            outframe = kalmanFilterRecon(moviefile, caldata, params, amat, bmat, fmat);
        case 'kalman_smoothing'
            fmat = eye(params.nsamples);
            outframe = kalmanSmoothingRecon(moviefile, caldata, params, amat, bmat, fmat);
        otherwise % default to naive corner cam
            outframe = cornerRecon(moviefile, caldata, params, amat);
    end

    figure; imagesc(outframe);

    outframe = outframe(:,2:end,:); % throwing away the constant light

    if direction{c} == -1
        outframe = fliplr(outframe); 
    end

    outframes{c} = outframe;
end

if ncorners > 1
    x1_1d = [outframes{1} outframes{2}];
    figure; imagesc(x1_1d);
end

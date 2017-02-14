addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
% gridfile = sprintf('%s/calibrationgrid.MOV', expfolder);
% backfile = sprintf('%s/dark_calibration.MOV', expfolder);
% moviefile = sprintf('%s/dark_MovieLines_greenred1.MOV', expfolder);
% outfile = sprintf('%s/out_dark_greenred_kalman_all', resfolder);

gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_kalman_all.MOV', resfolder);

theta_lim = [pi/2, 0];
% theta_lim = [0, pi/2];
minclip = 0;
maxclip = 2;
nsamples = 20;
smooth_up = 4;
step = 5;
sub_background = 0;
start = 60*5;
do_rectify = 1;
downlevs = 2;
outr = 80;

filt = binomialFilter(5);

if ~(exist('corner', 'var') && exist('frame1', 'var'))
    v = VideoReader(backfile);
    background = double(read(v, 1));
    if ~sub_background
        background = zeros(size(background));
    end
    
    v = VideoReader(moviefile);
    endframe = v.NumberOfFrames;
    frame1 = double(read(v,start));

    if do_rectify == 1
        vcali = VideoReader(gridfile);
        caliImg = read(vcali,100);
        [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg, 1);
        %figure;imagesc(rectified_img./max(rectified_img(:)))
        frame1 = rectify_image(frame1, iold, jold, ii, jj);
        background = rectify_image(background, iold, jold, ii, jj);
    end

    frame1 = blurDnClr(frame1, downlevs, filt);
    % frame1 = imresize(frame1, 0.5^downlevs);
    figure; imagesc(frame1(:,:,1));

    corner = ginput(1);
    hold on; plot(corner(1), corner(2), 'ro');
    maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;
end

[amat, x0, y0] = allPixelAmat(corner, outr, nsamples, theta_lim);
crop_idx = sub2ind(size(frame1), y0, x0);

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);
lambda = 15; % pixel noise
sigma = 0.4; % prior
alpha = 5e-3; % process noise

% transition prior
fmat = eye(nsamples); % stationary for now

vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);

clear out1 rgbq diffs
frame = start:step:endframe/2;
nout = length(frame);
nchans = size(frame1, 3);

% initialize filter variables
cur_mean = zeros([nsamples, nchans]);
pred_mean = zeros([nsamples, nchans]);
cur_cov = zeros([nsamples, nsamples, nchans]);
prior_cov = inv(bmat' * bmat / sigma^2);
pred_cov = repmat(prior_cov, [1, 1, 3]);
rmat = lambda * eye(size(amat,1)); % independent pixel noise
qmat = alpha * eye(nsamples); % independent process noise

tic;
nout = 10;
all_gains = zeros([nsamples, size(amat,1), nout]);
for i = 1:nout
    n = frame(i);
    fprintf('Iteration %i\n', n);
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen - background, downlevs, filt);
    
    % kalman forward filtering
    for c = 1:nchans
        % observation at this frame
        chan = framen(:,:,c);
        chan(isnan(chan)) = 0;
        y = reshape(chan(crop_idx), [outr^2, 1]);
        % update step
        gain = pred_cov(:,:,c) * amat' * inv(amat*pred_cov(:,:,c)*amat' + rmat);
        cur_mean(:,c) = pred_mean(:,c) + gain * (y - amat * pred_mean(:,c));
        cur_cov(:,:,c) = pred_cov(:,:,c) - gain * amat * pred_cov(:,:,c);
        all_gains(:,:,i) = gain;
        
        % next predict step
        pred_mean(:,c) = fmat * cur_mean(:,c);
        pred_cov(:,:,c) = fmat * cur_cov(:,:,c) * fmat' + qmat;
    end
    toc;

    % write out the video
    out1(1,:,:) = smoothSamples(cur_mean, smooth_up);
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    writeVideo(vout, (repmat(out1, [round(size(out1,2)/2), 1])-minclip)./(maxclip-minclip));
end
close(vout);

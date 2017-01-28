addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));



datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_kalsmooth.MOV', resfolder);

theta_lim = [pi/2, 0];
minclip = 0;
maxclip = 0.5;
nsamples = 200;
smooth_up = 4;
step = 5;
sub_background = 0;
start = 60*5;
do_rectify = 1;
downlevs = 3;

filt = binomialFilter(5);

rs = 10:4:32;

if ~(exist('corner', 'var') && exist('frame1', 'var'))
    v = VideoReader(backfile);
    background = double(read(v, 1));
    if ~sub_background
        background = zeros(size(background));
    end

    v = VideoReader(moviefile);
    nframes = v.NumberOfFrames;
    frame1 = double(read(v,start));

    if do_rectify == 1
        vcali = VideoReader(gridfile);
        caliImg = read(vcali,100);
        [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
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

% y = Ax
amat = zeros([nsamples*length(rs), nsamples]);
for i = 1:length(rs)
    si = nsamples*(i-1) + 1;
    ei = nsamples*i;
    amat(si:ei,:) = tril(ones(nsamples));
end

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);
lambda = 10; sigma = 1; alpha = 5e-3;

% transition prior
fmat = eye(nsamples); % stationary for now


clear out1 rgbq diffs
[nrows, ncols, nchans] = size(frame1);
endframe = nframes/4;
frames = start:step:endframe;
nout = length(frames);

% initialize filter variables
rmat = lambda * eye(nsamples*length(rs)); % independent pixel noise
qmat = alpha * eye(nsamples); % independent process noise
% need to save the means and covs for backwards smoothing
cur_mean = zeros([nsamples, nchans, nout]);
pred_mean = zeros([nsamples, nchans, nout+1]);
cur_cov = zeros([nsamples, nsamples, nchans, nout]);
pred_cov = zeros([nsamples, nsamples, nchans, nout+1]);
prior_cov = inv(bmat' * bmat / sigma^2);
pred_cov(:,:,:,1) = repmat(prior_cov, [1, 1, 3]); % pred 1|0

for i = 1:nout
    n = frames(i);
    fprintf('Iteration %i\n', n);
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj) - background;
    end
    framen = blurDnClr(framen, downlevs, filt);
    [rgbq, diffs] = gradientAlongCircle(framen, corner, rs, nsamples, theta_lim);
    % rgbq is nsamples x nchans x length(rs)
    rgbq = permute(rgbq, [1, 3, 2]);
    y = reshape(rgbq, [nsamples*length(rs), nchans]);
    
    % kalman forward filtering
    for c = 1:nchans
        % update step
        gain = pred_cov(:,:,c,i) * amat' * inv(amat*pred_cov(:,:,c,i)*amat' + rmat);
        cur_mean(:,c,i) = pred_mean(:,c,i) + gain * (y(:,c) - amat * pred_mean(:,c,i));
        cur_cov(:,:,c,i) = pred_cov(:,:,c,i) - gain * amat * pred_cov(:,:,c,i);
        
        % next predict step, i+1|i
        pred_mean(:,c,i+1) = fmat * cur_mean(:,c,i);
        pred_cov(:,:,c,i+1) = fmat * cur_cov(:,:,c,i) * fmat' + qmat;
    end
end

out_mean = zeros(size(cur_mean)); % don't need the out cov
for c = 1:nchans
    out_mean(:,c,nout+1) = pred_mean(:,c,nout+1);
    for i = nout:-1:1
        res = pred_mean(:,c,i+1) - out_mean(:,c,i+1);
        fi_res = cur_cov(:,:,c,i)*fmat'*(pred_cov(:,:,c,i+1)\res);
        out_mean(:,c,i) = cur_mean(:,c,i) + fi_res;
    end
end

vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);

for i = 1:nout
    %write out the video
    out1(1,:,:) = smoothSamples(out_mean(:,:,i), smooth_up);
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    writeVideo(vout, (repmat(out1, [size(out1,2),1])-minclip)./(maxclip-minclip));
end
close(vout);
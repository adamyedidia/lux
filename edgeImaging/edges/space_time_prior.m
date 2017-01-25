addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos';
expfolder = sprintf('%s/experiment_2', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/calibrationgrid.MOV', expfolder);
backfile = sprintf('%s/light_calibration.MOV', expfolder);
moviefile = sprintf('%s/light_MovieLines_greenred1.MOV', expfolder);
outfile = sprintf('%s/out_light_greenred_kalman_300.MOV', resfolder);

% theta_lim = [pi/2, 0];
theta_lim = [0, pi/2];
nsamples = 300;
start = 1;
do_rectify = 0;
downlevs = 3;

if ~(exist('corner', 'var') && exist('frame1', 'var'))
    v = VideoReader(backfile);
    background = double(read(v, 1));

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

    frame1 = blurDnClr(frame1, downlevs, binomialFilter(5));
    % frame1 = imresize(frame1, 0.5^downlevs);
    imagesc(frame1(:,:,1));

    corner = ginput(1);
    hold on; plot(corner(1), corner(2), 'ro');
    maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;
end

% y = Ax
rs = 10:2:30;
amat = zeros([nsamples*length(rs), nsamples]);
for i = 1:length(rs)
    si = nsamples*(i-1) + 1;
    ei = nsamples*i;
    amat(si:ei,:) = tril(ones(nsamples));
end

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);
lambda = 1; sigma = 1; alpha = 1;

% transition prior
fmat = eye(nsamples); % stationary for now

minclip = 0;
maxclip = 1;
step = 10;
vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);

clear out1 rgbq diffs
[nrows, ncols, nchans] = size(frame1);

% initialize filter variables
pred_mean = zeros([nsamples, nchans]);
cur_mean = zeros([nsamples, nchans]);
cur_cov = zeros([nsamples, nsamples, nchans]);
prior_cov = inv(bmat' * bmat / sigma^2);
pred_cov = repmat(prior_cov, [1, 1, 3]);
rmat = lambda * eye(nsamples*length(rs)); % independent pixel noise
qmat = alpha * eye(nsamples); % independent process noise
for n=start:step:500
    fprintf('Iteration %i\n', n);
    % read the nth frame
    framen = double(read(v,n)) - background;
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
    [rgbq, diffs] = gradientAlongCircle(framen, corner, rs, nsamples, theta_lim);
    % rgbq is nsamples x nchans x length(rs)
    rgbq = permute(rgbq, [1, 3, 2]);
    y = reshape(rgbq, [nsamples*length(rs), nchans]);
    
    % kalman forward filtering
    for c = 1:nchans
        % update step
        gain = pred_cov(:,:,c) * amat' * inv(amat*pred_cov(:,:,c)*amat' + rmat);
        cur_mean(:,c) = pred_mean(:,c) + gain * (y(:,c) - amat * pred_mean(:,c));
        cur_cov(:,:,c) = pred_cov(:,:,c) - gain * amat * pred_cov(:,:,c);
        
        % next predict step
        pred_mean(:,c) = fmat * cur_mean(:,c);
        pred_cov(:,:,c) = fmat * cur_cov(:,:,c) * fmat' + qmat;
    end
   
    %write out the video
    out1(1,:,:) = cur_mean;
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    writeVideo(vout, (repmat(out1, [nsamples/2, 1])-minclip)./(maxclip-minclip));
end
close(vout);
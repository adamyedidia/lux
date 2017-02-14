addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan29';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_location3.MOV', expfolder);
backfile = sprintf('%s/calibration_location3.MOV', expfolder);
moviefile = sprintf('%s/blue_location3.MOV', expfolder);
outfile = sprintf('%s/out_blue_loc3_kal2', resfolder);

% gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
% backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
% moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
% outfile = sprintf('%s/out_red_dark_greenscreen_kal_nomean.MOV', resfolder);

theta_lim = [0, pi/2];
minclip = 0;
maxclip = 0.2;
nsamples = 200;
smooth_up = 4;
step = 5;
sub_background = 1;
start = 60*5;
do_rectify = 1;
downlevs = 3;
lambda = 15; % pixel noise
sigma = 0.4; % prior
alpha = 5e-3; % process noise

filt = binomialFilter(5);

rs = 10:4:30;

if ~(exist('corner', 'var') && exist('frame1', 'var'))
    v = VideoReader(moviefile);
    endframe = v.NumberOfFrames;
    frame1 = double(read(v,start));
    
%     vback = VideoReader(backfile);
%     frame1 = double(read(vback, 1));
%     background = double(read(vback, 1));
%     if ~sub_background
%         background = zeros(size(background));
%     end
    
    background = zeros(size(frame1));
    if sub_background
        count = 0;
        for n = start:endframe
            background = background + double(read(v,n));
            count = count + 1;
        end
        background = background / count;
    end
    mean_pixel = mean(mean(background, 1), 2);

    if do_rectify == 1
        vcali = VideoReader(gridfile);
        caliImg = read(vcali,100);
        [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
        %figure;imagesc(rectified_img./max(rectified_img(:)))
        frame1 = rectify_image(frame1, iold, jold, ii, jj);
        background = rectify_image(background, iold, jold, ii, jj);
    end

    frame1 = blurDnClr(frame1, downlevs, filt);
    mean_pixel = repmat(mean_pixel, [size(frame1, 1), size(frame1, 2)]);
    
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

% transition prior
fmat = eye(nsamples); % stationary for now

clear out1 rgbq diffs
frame = start:step:endframe;
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


% vout = VideoWriter(outfile);
% vout.FrameRate = 10;
% open(vout);
out = zeros([nout, (nsamples-1)*smooth_up+1, nchans]);
tic;
% nout = 1;
for i = 1:nout
    n = frame(i);
    fprintf('Iteration %i\n', n);
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen - background, downlevs, filt) + mean_pixel;
%     writeVideo(vout, (framen-min(framen(:)))/(max(framen(:))-min(framen(:))));
    
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
    toc;

    % write out the video
%     out1(1,:,:) = smoothSamples(cur_mean, smooth_up);
    out1 = 40*smoothSamples(cur_mean, smooth_up);
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    out(i,:,:) = (out1 - minclip) ./ (maxclip - minclip);
%     writeVideo(vout, (repmat(out1, [round(size(out1,2)/2), 1])-minclip)./(maxclip-minclip));
end
% close(vout);

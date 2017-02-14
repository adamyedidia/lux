addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_space_all.MOV', resfolder);

theta_lim = [pi/2, 0];
% theta_lim = [0, pi/2];
minclip = 0;
maxclip = 2;
nsamples = 50;
smooth_up = 4;
step = 5;
sub_background = 0;
start = 60*5;
do_rectify = 1;
downlevs = 2;
outr = 50;

filt = binomialFilter(5);

if ~exist('frame1', 'var')
%     v = VideoReader(backfile);
%     background = double(read(v, 1));
%     if ~sub_background
%         background = zeros(size(background));
%     end

    v = VideoReader(moviefile);
    nframes = v.NumberOfFrames;
    frame1 = double(read(v,start));
    
    background = zeros(size(frame1));
    count = 0;
    for n = start:endframe
        background = background + double(read(v,n))/(count+1);
        count = count + 1;
    end
    background = background / count;
    if ~sub_background
        background = zeros(size(background));
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
    mean_pixel = repmat(mean_pixel, [size(frame1,1), size(frame1,2)]);
    figure; imagesc(frame1(:,:,1));

    corner = ginput(1);
    hold on; plot(corner(1), corner(2), 'ro');
end

% % y = Ax
% rs = 10:2:30;
% amat = zeros([nsamples*length(rs), nsamples]);
% for i = 1:length(rs)
%     si = nsamples*(i-1) + 1;
%     ei = nsamples*i;
%     amat(si:ei,:) = tril(ones(nsamples));
% end

[amat, x0, y0] = allPixelAmat(corner, outr, nsamples, theta_lim);
crop_idx = sub2ind(size(frame1), y0, x0);

% spatial prior
bmat = eye(nsamples) - diag(ones([nsamples-1,1]), 1);
lambda = 15; sigma = 0.4;

vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);

clear out1 rgbq diffs
[nrows, ncols, nchans] = size(frame1);
tic;
for n=start:step:nframes/2
    fprintf('Iteration %i\n', n);
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen - background, downlevs, filt) - mean_pixel;
    
%     [rgbq, diffs] = gradientAlongCircle(framen, corner, rs, nsamples, theta_lim);
%     % rgbq is nsamples x nchans x length(rs)
%     rgbq = permute(rgbq, [1, 3, 2]);
%     y = reshape(rgbq, [nsamples*length(rs), nchans]);
    
    % using only spatial prior
    out = zeros([nsamples, nchans]);
    for c = 1:nchans
        chan = framen(:,:,c);
        chan(isnan(chan)) = 0;
        y = reshape(chan(crop_idx), [outr^2, 1]);
%         out(1,:,c) = (amat'*amat/lambda + bmat'*bmat/sigma^2)\(amat'*y(:,c)/lambda);
        out(:,c) = (amat'*amat/lambda + bmat'*bmat/sigma^2)\(amat'*y/lambda);
    end
    toc
    %write out the video
    out1(1,:,:) = smoothSamples(out, smooth_up);
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    writeVideo(vout, (repmat(out1, [round(size(out1,2)/2), 1]) -minclip)./(maxclip-minclip));
end
close(vout);

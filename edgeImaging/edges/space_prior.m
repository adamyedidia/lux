addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_smooth.MOV', resfolder);

theta_lim = [pi/2, pi];
minclip = 0;
maxclip = 0.5;
nsamples = 200;
step = 5;
start = 200;
do_rectify = 1;
downlevs = 3;

if ~exist('frame1', 'var')
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
lambda = 1; sigma = 1;

vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);

clear out1 rgbq diffs
[nrows, ncols, nchans] = size(frame1);
for n=start:step:nframes
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
    
    % using only spatial prior
    out1 = zeros([1, nsamples, nchans]);
    for c = 1:nchans
        out1(1,:,c) = (amat'*amat/lambda + bmat'*bmat/sigma^2)\(amat'*y(:,c)/lambda);
    end

    %write out the video
    out1(out1<minclip) = minclip;
    out1(out1>maxclip) = maxclip;
    writeVideo(vout, (repmat(out1, [nsamples/2, 1]) -minclip)./(maxclip-minclip));
end
close(vout);
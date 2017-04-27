% gosynth.m  starting script for corner cam analysis.
% dec. 14, 2017  billf created.
% dech. 20, 2016 to do:  visualize the noise level of observation noise that
% I'm adding in.  Do I need to use pca coefficients in the reconstruction??


% dec. 22, 2016   visualize xAll and xhatAll as images.  print this out to
% re-organize (like an image) xAll and xhatAll   (rename it xHatAll).  Make
% sure that the first frame of the video is either suppressed or that it
% looks fine.  Make sure that the image versions of the estimated and true
% videos look somewhat the same.  
% figure;showIm(shiftdim(xAll,2))
% figure;showIm(shiftdim(xAll,2))  why is that black and white and now
% color?


% set observation noise level
obsNoise = 0.0005;
lambda = 10.0;
% set number of video frames to read in
nFrames = 10;

% load movie.  first, get one frame.
v = VideoReader('IMG_3294.MOV');
% Get the number of frames in the video sequence.
maxNumFrames = get(v, 'numberOfFrames');
frameOne = double(read(v,1));
frameOne = blurDnClr(blurDnClr(blurDnClr(frameOne)));
sizeFrame = size(frameOne);
figure;showIm(frameOne);

% initialize parameters
% set a variable with all the start-up parameters
paramsVoffset = initParams(2);   % this has a vertical offset
%% paramsNoVoffset = initParams(1);   % this has no vertical offset of the scene.


% calibrate inversion amat for this movie size.
[ amat ] = getAmat(paramsVoffset, frameOne(1,:,:));
amat2d = reshape(amat,[paramsVoffset.imageH * paramsVoffset.imageV, ...
    size(frameOne,2)]);

% from mooncam7.pdf, xhat = A^T (A A^T + lambda * I)^{-1}*(y-ybar)
% where lambda is the ratio of observation pixel noise variance to the
% scene image pixel variation variance.  The ratio of the square root
% variances mightb e 10, so lambda could be 0.01
aat = amat2d * amat2d';
kgain = amat2d' * inv(aat + lambda * eye(size(aat,1)));

% now compute the mean observation
ybar = zeros(paramsVoffset.imageH, paramsVoffset.imageV,3);
xbar = zeros(size(frameOne(1,:,:)));
for n = 1:nFrames
    n
    thisFrame = double(read(v,n));
    thisFrame = blurDnClr(blurDnClr(blurDnClr(thisFrame)));
    % now, render this frame to the ground plane.  for speed, render a
    % squished, 1-d  version.
    [ yrgb, pos ] = renderScene( paramsVoffset, mean(thisFrame,1) );
    ybar = ybar + yrgb;
    xbar = xbar + mean(thisFrame,1);
end
ybar = ybar ./ nFrames;
xbar = xbar ./ nFrames;
mx = max(ybar(:));
mn = min(ybar(:));

% for the n frames,
% get the frame, render it, add noise, estimate the resulting scene,
% upreplicate it to a 2d image, create output video.
xhatAll = zeros([size(kgain, 1),3,nFrames]);
xAll = zeros(size(xhatAll));
vidObj = VideoWriter('rendered.avi');
open(vidObj);
figure;
for n = 1:nFrames
    n
    thisFrame = double(read(v,n));
    thisFrame = blurDnClr(blurDnClr(blurDnClr(thisFrame)));
    % now, render this frame to the ground plane.  for speed, render a
    % squished version.
    [ yrgb, pos ] = renderScene( paramsVoffset, mean(thisFrame,1) );
    %% add noise
    yrgbRaw = yrgb + obsNoise * randn(size(yrgb)) * mx;
    yrgb = (yrgbRaw - mn) ./ (mx-mn);
    yrgb(:) = max(min(yrgb(:),1), 0);
    image(yrgb);
    currFrame = getframe;
    writeVideo(vidObj, currFrame);
    
    %% invert the yrgb image back to a 1-d image, and store that
    % remove the mean
    zeromeancolumn = reshape(yrgbRaw - ybar, [size(yrgb,1)*size(yrgb,2), size(yrgb,3)]);
    
    for icol = 1:3
        xhatAll(:,icol,n) = kgain * zeromeancolumn(:, icol);
        tmp = mean(thisFrame,1);
        xAll(:,icol,n) = reshape(tmp(:,:,icol), [size(tmp,2),1]);
    end
    
end
close(vidObj);

% plot the xbar's and the xhat's.
figure; clr = 2;
for n = 1:min(nFrames,4);
    subplot(min(nFrames,4),1,n);
    plot(xAll(:,clr,n) - squeeze(xbar(1,:,clr))', 'r'); hold on;
    plot(xhatAll(:,clr,n), 'g'); hold off;
    title(['frame ' num2str(n) ' r real, g est']);
end
figure;showIm(reshape(kgain(101,:), ...
[paramsVoffset.imageH, paramsVoffset.imageV]));

vidObj = VideoWriter('estimated.avi');
open(vidObj);
figure;
for n = 1:nFrames
    tmp = xhatAll(:,:,n);
    tmp = (tmp-min(tmp(:)))./ (max(tmp(:)) - min(tmp(:)));
    % now upsample and output that frame to the rendered video.
    image(repmat(reshape(tmp, [1, size(kgain,1), 3]), [500, 1 1]));
    currFrame = getframe;
    writeVideo(vidObj, currFrame);
    
end
close(vidObj);


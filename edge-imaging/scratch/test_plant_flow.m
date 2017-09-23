addpath(genpath('opticalflow'));

close all; 
clearvars -except medimg

vidfile = '~/Downloads/MVI_2633.MOV';
vsrc = VideoReader(vidfile);

start_t = 5;
end_t = vsrc.Duration - 1;
fps = vsrc.FrameRate;

downlevs = 2;

sfilt = binomialFilter(5);
ts = start_t: 1/fps: end_t;

if ~exist('medimg', 'var')
    medimg = getMedianFrame(vidfile, downlevs, fps, start_t, end_t);
end

alpha = 0.1;
ratio = 0.5;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

width = 20;
fsize = width/2;
tfilt = zeros([1,1,1,fsize]);
f = binomialFilter(width);
tfilt(1,1,1,:) = f(1:fsize) * 2;
frame_win = zeros([size(medimg), fsize]);

prevframe = zeros(size(medimg));
[yy, xx] = ndgrid(1:size(medimg,1), 1:size(medimg,2));
for i = 1:length(ts)
    vsrc.CurrentTime = ts(i);
    curframe = blurDnClr(double(readFrame(vsrc)), downlevs, sfilt);
    frame_win(:,:,:,1:end-1) = frame_win(:,:,:,2:end);
    frame_win(:,:,:,end) = curframe;
    if i < fsize
        continue;
    end
    frame = sum(bsxfun(@times, tfilt, frame_win), 4);
    if i > fsize + 1
        res1 = frame - medimg;
        mclip = 20;
        resclip1 = (res1 + mclip)/2/mclip;
        resclip1(resclip1 < 0) = 0;
        resclip1(resclip1 > 1) = 1;

        res2 = prevframe - medimg;
        mclip = 20;
        resclip2 = (res2 + mclip)/2/mclip;
        resclip2(resclip2 < 0) = 0;
        resclip2(resclip2 > 1) = 1;

        [vx, vy, warpI2] = Coarse2FineTwoFrames(resclip1,resclip2, para);

        imagesc(resclip1); hold on;
        quiver(xx, yy, vx, vy); hold off; set(gca, 'YDir', 'reverse');
        axis([1, size(resclip1,2), 1, size(resclip1,1)]); pause(0.1); 
    end
    prevframe = frame;
end
        
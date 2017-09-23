addpath(genpath('matlabPyrTools'));

close all; clear all;


vidfile = '~/Downloads/MVI_2633.MOV';
vidfile = '~/Downloads/zoomedMVI_4958.MOV';
vidfile = '~/Downloads/reenactmentMVI_4959.MOV';
vidfile = '~/Downloads/pt_light_source MVI_4960.MOV';
vsrc = VideoReader(vidfile);

start_t = 5;
end_t = vsrc.Duration - 1;
fps = vsrc.FrameRate;

downlevs = 2;

if ~exist('medimg', 'var')
    medimg = getMedianFrame(vidfile, downlevs, fps, start_t, end_t);
end

sfilt = binomialFilter(5);
ts = start_t: 1/fps: end_t;

width = 10;
filt = zeros([1,1,1,width]);
filt(1,1,1,:) = binomialFilter(width);
frame_win = zeros([size(medimg), width]);

vid = 1;
if vid
    vout = VideoWriter(sprintf('residuals_timeblur%d.avi', width));
    vout.FrameRate = 10;
    open(vout);
end

prevframe = zeros(size(medimg));
for i = 1:length(ts)
    vsrc.CurrentTime = ts(i);
    curframe = blurDnClr(double(readFrame(vsrc)), downlevs, sfilt);
    frame_win(:,:,:,1:end-1) = frame_win(:,:,:,2:end);
    frame_win(:,:,:,end) = curframe;
    if i < width
        continue;
    end
    frame = sum(bsxfun(@times, filt, frame_win), 4);
    if i > width*2
        res0 = curframe - medimg;
        res1 = frame - medimg;
        tdiff = (imgaussfilt(frame - prevframe, 2) + 1)/2;
        im0 = vertcat(curframe/255, (res0 + 10)/20);
        im1 = vertcat((res1 + 10)/20, tdiff);
        im = horzcat(im0, im1);
%         imshow(im); pause(0.1);
        if vid
            writeVideo(vout, uint8(255*im));
        else
            imshow(im); pause(0.1);
        end
    end
    prevframe = frame;
end

if vid
    close(vout);
end
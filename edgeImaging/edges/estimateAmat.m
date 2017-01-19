addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));

clear; close all;

datafolder = '../data/testvideos/experiment_2';
% datafolder = '/Users/klbouman/Downloads';

gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
calfile = sprintf('%s/dark_calibration.MOV', datafolder);
moviefile = sprintf('%s/dark_MovieLines_greenred1.MOV', datafolder);
outfile = sprintf('%s/out_greenred_revert.MOV', datafolder);
% v = VideoReader(calfile);
% background = imresize(double(read(v, 1)), 0.25);

v = VideoReader(moviefile);
nframes = v.NumberOfFrames;

frame1 = imresize(double(read(v, 1)), 0.25);
do_rectify = 0;
if do_rectify == 1
    vcali = VideoReader(gridfile);
    caliImg = readFrame(vcali);
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
end
% frame1 = blurDnClr(frame1, 3, binomialFilter(5));
imagesc(frame1(:,:,1));
background = zeros(size(frame1));
warning('background set to zeros');

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

nsamples = 200;
maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;

vout = VideoWriter(outfile);
vout.FrameRate = 10;
minclip = 0;
maxclip = 1;
open(vout)

for n=1:10:500
    n
    % read the nth frame
    framen = imresize(double(read(v, n)), 0.25) - background;
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
%     framen = blurDnClr(framen, 3, binomialFilter(5));
    
    rs = 10:2:30;
    for i = 1:length(rs)
        [rgbq(:,:,:,i), diffs(:,:,:,i)] = gradientAlongCircle(rs(i), nsamples, framen, corner);
    end
    outframe(1,:,:) = mean(diffs, 4);

    %compute the average frame from all the circle differences
%     [pixel_avg, diffs] = anglesAveragePixel(framen, corner, nsamples, maxr);
%     [pixel_avg, diffs] = estimatedGradient(framen, corner, nsamples, maxr);
%     diffs = sectorAverageGradient(framen, corner, nsamples, maxr);
%     outframe(1,:,:) = diffs;

    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;

    writeVideo(vout, (repmat(outframe, [100 1]) -minclip)./(maxclip-minclip));
end

close(vout);

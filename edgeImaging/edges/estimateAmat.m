addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));

datafolder = '../data/testvideos/experiment_2';
% datafolder = '/Users/klbouman/Downloads';

gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
do_rectify = 0;
calfile = sprintf('%s/dark_calibration.MOV', datafolder);
v = VideoReader(calfile);
background = imresize(double(read(v, 1)), 0.25);

moviefile = sprintf('%s/dark_MovieLines_greenred1.MOV', datafolder);

v = VideoReader(moviefile);
nframes = v.NumberOfFrames;
frame1 = imresize(double(read(v, 1)), 0.25);

if do_rectify == 1
    vcali = VideoReader(gridfile);
    caliImg = readFrame(vcali);
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
end
% frame1 = blurDnClr(frame1, 3, binomialFilter(5));
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');


nsamples = 200;
maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;

% make video of mask rotating
vout = VideoWriter('mask_vid');
vout.FrameRate = 10;
open(vout);
[yy, xx] = ndgrid(1:size(frame1, 1), 1:size(frame1, 2));
x0 = xx - corner(1);
y0 = yy - corner(2);
% zero out pixels that are more than max_r from corner
theta = atan2(y0, x0);
angles = linspace(0, pi/2, nsamples+2);
angles = angles(2:end-1);
eps = (angles(2) - angles(1))/2;
for i = 1:nsamples
    adiff = theta - angles(i);
    mask = abs(adiff) <= eps;
    writeVideo(vout, uint8(2^8*mask));
end
close(vout);

vout = VideoWriter(sprintf('%s/out_greenred2.MOV', datafolder));
vout.FrameRate = 10;
minclip = 0;
maxclip = 1;
open(vout)

for n=1:10:nframes
    n
    % read the nth frame
    framen = imresize(double(read(v, n)), 0.25) - background;
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
%     framen = blurDnClr(framen, 3, binomialFilter(5));
    
%     count = 1;
%     for r = 10:5:30
%         [rgbq(:,:,:,count), diffs(:,:,:,count)] = gradientAlongCircle(r, nsamples, framen, corner);
%         count = count + 1;
%     end
%     outframe(1,:,:) = mean(diffs, 4);

    %compute the average frame from all the circle differences
    [pixel_avg, diffs] = anglesAveragePixel(framen, corner, nsamples, maxr);
    outframe(1,:,:) = diffs;
    
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    
    writeVideo(vout, (repmat(outframe, [100 1]) -minclip)./(maxclip-minclip));
end

close(vout);


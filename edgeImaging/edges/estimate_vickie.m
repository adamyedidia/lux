addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
expfolder = sprintf('%s/experiments', datafolder);
resfolder = sprintf('%s/results', datafolder);
gridfile = sprintf('%s/grid_greenscreen.MOV', expfolder);
backfile = sprintf('%s/calibration_dark_greenscreen.MOV', expfolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
outfile = sprintf('%s/out_red_dark_greenscreen_0.MOV', resfolder);

theta_lim = [pi, pi/2];
minclip = 0;
maxclip = 0.5;
nsamples = 200;
step = 5;
sub_background = 0;
start = 60*5;
do_rectify = 1;
downlevs = 3;

rs = 10:2:30;


if ~exist('frame1', 'var')
    v = VideoReader(backfile);
    background = double(read(v, 1));

    v = VideoReader(moviefile);
    nframes = v.NumberOfFrames;
    frame1 = double(read(v,start)) - background;

    if do_rectify == 1
        vcali = VideoReader(gridfile);
        caliImg = read(vcali,100);
        [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
        %figure;imagesc(rectified_img./max(rectified_img(:)))
        frame1 = rectify_image(frame1, iold, jold, ii, jj);
    end

    frame1 = blurDnClr(frame1, downlevs, binomialFilter(5));
    % frame1 = imresize(frame1, 0.5^downlevs);
    imagesc(frame1(:,:,1));

    corner = ginput(1);
    hold on; plot(corner(1), corner(2), 'ro');
    maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;
end

vout = VideoWriter(outfile);
vout.FrameRate = v.FrameRate/step;
open(vout);
clear outframe rgbq diffs
[nrows, ncols, ~] = size(frame1);

for n=start:step:nframes/2
    fprintf('Iteration %i\n', n);

    % read the nth frame
    framen = double(read(v,n)) - background;
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
%     imagesc(framen(:,:,1));
    
    [rgbq, diffs(1,:,:,:)] = gradientAlongCircle(framen, corner, rs, nsamples, theta_lim);
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = mean(diffs, 4);
    
%     diffs = sectorAverageGradient(framen, corner, nsamples, maxr);
%     [pixel_avg, diffs] = estimatedGradient(framen, corner, nsamples, 10, 30);
%     outframe(1,:,:) = diffs;
    
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    writeVideo(vout, (repmat(outframe, [nsamples/2, 1]) -minclip)./(maxclip-minclip));
end

close(vout);

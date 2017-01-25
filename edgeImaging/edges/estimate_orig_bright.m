addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


datafolder = '../data/testvideos/experiment_2';
% datafolder = '/Users/klbouman/Downloads';
gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
backfile = sprintf('%s/dark_calibration.MOV', datafolder);
moviefile = sprintf('%s/dark_MovieLines_greenred1.MOV', datafolder);
outfile = sprintf('%s/out_dark_greenred_400_0.MOV', datafolder);
nsamples = 400;
start = 1;
do_rectify = 0;
downlevs = 3;

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

minclip = 0;
maxclip = 1;
step = 10;
frames = zeros([1 + floor((nframes - start)/step), nsamples-1, 3]);
vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);
clear outframe rgbq diffs
[nrows, ncols, ~] = size(frame1);

rs = 10:2:30;
angles = linspace(0, pi/2, 100);
for n=start:step:800
    n
    
    % read the nth frame
    framen = double(read(v,n)) - background;
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
%     imagesc(framen(:,:,1));
    
    [rgbq, diffs(1,:,:,:)] = gradientAlongCircle(framen, corner, rs, nsamples);
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = mean(diffs, 4);
%     frames(n,:,:) = mean(diffs, 4);
%     sum(frames(:))
    
%     diffs = sectorAverageGradient(framen, corner, nsamples, maxr);
%     [pixel_avg, diffs] = estimatedGradient(framen, corner, nsamples, 10, 30);
%     outframe(1,:,:) = diffs;
    
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    writeVideo(vout, (repmat(outframe, [nsamples/2, 1]) -minclip)./(maxclip-minclip));
end

close(vout);

amat = zeros([nsamples-1, nrows*ncols]);
for i = 1:length(rs)
    amat = amat + inverseAmat(nrows, ncols, rs(i), nsamples);
end


addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));

clear; close all;

datafolder = '../data/testvideos/experiment_2';
% datafolder = '/Users/klbouman/Downloads';

gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
moviefile = sprintf('%s/dark_MovieLines_greenred1.MOV', datafolder);
outfile = sprintf('%s/out_greenred_revert2.MOV', datafolder);

v = VideoReader(moviefile);
frame1 = double(read(v,1));
downlevs = 2;

do_rectify = 0;
if do_rectify == 1
    vcali = VideoReader(gridfile);
    caliImg = read(vcali,100);
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
end

frame1 = blurDnClr(frame1, downlevs, binomialFilter(5));
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

vout = VideoWriter(outfile);
vout.FrameRate = 10;
minclip = 0;
maxclip = 1;
open(vout)

for n=1:10:500
    n
    
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
    
    rs = 10:2:30;
    for i = 1:length(rs)
        [rgbq(:,:,:,i), diffs(:,:,:,i)] = gradientAlongCircle(rs(i), 200, framen, corner);
    end
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = mean(diffs,4);
    
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    
    writeVideo(vout, (repmat(outframe, [100 1]) -minclip)./(maxclip-minclip));
end

close(vout);


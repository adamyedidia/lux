addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));

close all; clear;

%datafolder = '../data/testvideos/experiment_2';
datafolder = '/Users/klbouman/Downloads';


gridfile = sprintf('%s/calibration2.MOV', datafolder);
moviefile = sprintf('%s/redblue_chairsitting.MOV', datafolder);
outfile = sprintf('%s/out_redblue_chairsitting.MOV', datafolder);
background = sprintf('%s/calibration3.MOV', datafolder);

startframe = 8*60; 

vback = VideoReader(background);
backgroundframe = double(read(vback,100)); 

v = VideoReader(moviefile);
frame1 = double(read(v,startframe));
endframe = v.NumberOfFrames;
downlevs = 3;

do_rectify = 1;
if do_rectify == 1
    vcali = VideoReader(gridfile);
    caliImg = double(read(vcali,100));
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
    backgroundframe = rectify_image(backgroundframe, iold, jold, ii, jj);
end

frame1 = blurDnClr(frame1, downlevs, binomialFilter(5));
% frame1 = imresize(frame1, 0.5^downlevs);
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');
nsamples = 200;
maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;

outframes = nan(1,nsamples-1,3,endframe);  
for n=startframe:10:endframe
    n
    
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = framen  - backgroundframe; 
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
%     framen = imresize(framen, 0.5^downlevs);
    
    rs = 10:2:80;
    for i = 1:length(rs)
        [rgbq(:,:,:,i), diffs(:,:,:,i)] = gradientAlongCircle(rs(i), nsamples, framen, corner);
    end
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = nanmean(diffs,4);
    outframes(:,:,:,n) = outframe; 
end




% write out the video
vout = VideoWriter(outfile);
vout.FrameRate = 10;
minclip = 0;
maxclip = 1;
open(vout)

outframes2 = outframes - min(outframes(:)); 
outframes2 = outframes2./max(outframes2(:)); 

for n=startframe:10:endframe
    
    outframe = outframes2(:,:,:,n); 
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    
    writeVideo(vout, (repmat(outframe, [nsamples/2, 1]) -minclip)./(maxclip-minclip));

end
close(vout);


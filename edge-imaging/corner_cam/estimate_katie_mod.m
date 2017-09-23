addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


% datafolder = '/Users/klbouman/Downloads';
datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22/experiments';

gridfile = sprintf('%s/grid_greenscreen.MOV', datafolder);
moviefile = sprintf('%s/red_dark_greenscreen.MOV', datafolder);
outfile = sprintf('%s/out_red_dark_greenscreen_1.MOV', datafolder);
background = sprintf('%s/calibration_dark_greenscreen.MOV', datafolder);

% gridfile = sprintf('%s/grid_light.MOV', datafolder);
% moviefile = sprintf('%s/red_noartificiallight.MOV', datafolder);
% outfile = sprintf('%s/out_red_noartificiallight.MOV', datafolder);
% background = sprintf('%s/calibration_dark.MOV', datafolder);


startframe = 300; %8*60; 
delta = 5; 
subtract_background = 0;
do_rectify = 1;
nsamples = 400;

theta_lim = [pi, pi/2];

vback = VideoReader(background);
backgroundframe = double(read(vback,100)); 

v = VideoReader(moviefile);
frame1 = double(read(v,startframe));
endframe = v.NumberOfFrames;
downlevs = 3;

if do_rectify == 1
    vcali = VideoReader(gridfile);
    caliImg = double(read(vcali,100));
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
    backgroundframe = rectify_image(backgroundframe, iold, jold, ii, jj);
end

if ~subtract_background
    backgroundframe = zeros(size(backgroundframe)); 
end

frame1 = blurDnClr(frame1, downlevs, binomialFilter(5));
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');
maxr = min(size(frame1, 2) - corner(1), size(frame1, 1) - corner(2)) - 1;


if subtract_background 
    
    backgroundframe_blur = blurDnClr(backgroundframe, downlevs, binomialFilter(5));
    
    rs = 10:2:80;
    [rgbq_background(1,:,:,:), diffs_background(1,:,:,:)] = ...
        gradientAlongCircle(backgroundframe_blur, corner, rs, nsamples, theta_lim);
    
    background_img(1,:,:) = nanmean(diffs_background,4);
else
    background_img = zeros(1, nsamples-1, 3); 
end


outframes = nan(1,nsamples-1,3,endframe);  
for n=startframe:delta:endframe
    n
    
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    %framen = framen  - backgroundframe; 
    framen = blurDnClr(framen, downlevs, binomialFilter(5));
%     framen = imresize(framen, 0.5^downlevs);
    
    rs = 10:2:80;
    [~, diffs] = gradientAlongCircle(framen, corner, rs,...
            nsamples, theta_lim);
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = nanmean(diffs,3);
    outframes(:,:,:,n) = outframe - background_img;
end

% background_img = nanmean(outframes, 4); 
% 
% for n=startframe:delta:endframe
%     outframes(:,:,:,n) = outframes(:,:,:,n) - background_img; 
% end


if subtract_background
    minclip = min(outframes(:));
    maxclip = max(outframes(:));
else
    minclip = 0; 
    maxclip = .5; 
end

% write out the video
vout = VideoWriter(outfile);
vout.FrameRate = v.FrameRate/delta;
open(vout)

for n=startframe:delta:endframe
    
    outframe = outframes(:,:,:,n); 

    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    
    writeVideo(vout, (repmat(outframe, [nsamples/2, 1]) -minclip)./(maxclip-minclip));

end
close(vout);


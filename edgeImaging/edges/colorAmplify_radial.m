addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


%datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
datafolder = '/Users/klbouman/Research/shadowImaging/edgeImaging/data';
expfolder = sprintf('%s/testVideos_Jan29', datafolder);
resfolder = sprintf('%s/results', datafolder);

name = 'blue_location2';
moviefile = sprintf('%s/%s.MOV', expfolder, name);
outfile = sprintf('%s/magnify_%s_radial_alpha40.MOV', resfolder, name);
outfile_mean = sprintf('%s/mean_%s_radial.png', resfolder, name);
outfile_noise = sprintf('%s/residual_%s_radial.MOV', resfolder, name);
outfile_magnifyDC = sprintf('%s/magnfiyDC_%s_radial_alpha40.MOV', resfolder, name);


% moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
% outfile = sprintf('%s/magnify_red_dark_greenscreen_circular.MOV', resfolder);

% ======== PARAMETERS =============% 
startframe = 60*3;
step = 1;
alpha = 40.0; 
blurSz = [20 2];
radialImgSz = [1000 1000];
temporalfactor = .5; 

% load the video 
v = VideoReader(moviefile);
nframes = v.NumberOfFrames;
endframe = nframes-60*3; 

% ======== GET WARPING MATRICES =============% 

% select the corner
frame1 = read(v,1); 
figure;imagesc(frame1); title('click on the corner'); 
corner = ginput(1); 
[h, w, c] = size(frame1); 


% get the indices centered around the corner
[yy, xx] = ndgrid(1:h, 1:w);
x0 = xx - floor(corner(1));
y0 = yy - floor(corner(2));

% find the angle each point makes with the wall and corner
theta = atan2(double(y0), double(x0));
R = sqrt(x0.^2 + y0.^2); 

rs = linspace(0, max(R(:)), radialImgSz(1)); 
thetas = linspace(-pi, pi, radialImgSz(2)); 
[rr, tt] = ndgrid(rs, thetas); 

% get the x and y point corresponding with each r, theta pair
ynew = zeros(radialImgSz); 
xnew = zeros(radialImgSz); 
countr = 1; 
for r = rs
    countt = 1; 
    for t = thetas
        y = r*sin(t);
        x = r*cos(t); 
        ynew(countr, countt) = y; 
        xnew(countr, countt) = x;  
        countt = countt + 1; 
    end
    countr = countr + 1; 
end



% ======== AMPLIFYING =============% 

% calculate the mean image
disp('calculating mean image...'); 
meanImg = zeros(size(read(v,1))); 
count = 1; 
for n=startframe:endframe
    n
    meanImg = meanImg + (double(read(v,n)) - meanImg)/(count+1); 
    count = count + 1; 
end
meanmeanImg(1,1,:) = [ mean(mean(meanImg(:,:,1))), mean(mean(meanImg(:,:,2))), mean(mean(meanImg(:,:,3)))]; 
imwrite(meanImg./255, outfile_mean); 



disp('saving video...'); 

vout = VideoWriter(outfile);
vout.FrameRate = v.FrameRate / step;
open(vout);
vout_noise = VideoWriter(outfile_noise);
vout_noise.FrameRate = v.FrameRate / step;
open(vout_noise);
vout_magnifyDC = VideoWriter(outfile_magnifyDC);
vout_magnifyDC.FrameRate = v.FrameRate / step;
open(vout_magnifyDC);

for n=startframe:step:endframe
    n
    % get the frame
    origFrame = double(read(v,n));
    
    %subtract the mean
    meanSub = origFrame - meanImg;
    
    % warp into theta r space
    for k =1:c
        meanSub_theta(:,:,k) = interp2(x0, y0, meanSub(:,:,k) , xnew, ynew, 'linear', 0);
    end
    
    %blur
    meanSub_theta = imgaussfilt(meanSub_theta, blurSz);
    
    % warp back
    for k =1:c
        meanSub_blur(:,:,k) = interp2(tt, rr, meanSub_theta(:,:,k) , theta, R);
    end
    
    % amplify and smooth and add back in
    if n==startframe
        newFrame = meanImg + alpha*meanSub_blur; 
    else
        newFrame = temporalfactor*oldFrame + (1-temporalfactor)*(meanImg + alpha*meanSub_blur); 
    end
    oldFrame = newFrame; 
    
    % save
    newFrame(newFrame<0) = 0.0;
    newFrame(newFrame>255) = 255.0;
    writeVideo(vout, uint8(newFrame));
    
    % save the noise video with constnat DC
    meanSubConst = bsxfun(@plus, meanSub, meanmeanImg);  
    meanSubConst(meanSubConst<0) = 0.0;
    meanSubConst(meanSubConst>255) = 255.0; 
    writeVideo(vout_noise, uint8(meanSubConst));
    
    % save amplify color video with a constnat DC 
    newFrameConst = bsxfun(@plus, newFrame-meanImg, meanmeanImg);  
    newFrameConst(newFrameConst<0) = 0.0;
    newFrameConst(newFrameConst>255) = 255.0; 
    writeVideo(vout_magnifyDC, uint8(newFrameConst));
    
    
end

close(vout);
close(vout_noise);
close(vout_magnifyDC);


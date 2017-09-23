addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


%datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
datafolder = '/Users/klbouman/Research/shadowImaging/edgeImaging/data';
expfolder = sprintf('%s/testVideos_Jan29', datafolder);
resfolder = sprintf('%s/results', datafolder);

name = 'redflashlight2_location3';
moviefile = sprintf('%s/%s.MOV', expfolder, name);
outfile = sprintf('%s/magnify_%s_circular.MOV', resfolder, name);
outfile_mean = sprintf('%s/mean_%s_circular.MOV', resfolder, name);
outfile_noise = sprintf('%s/residual_%s_circular.MOV', resfolder, name);
outfile_magnifyDC = sprintf('%s/magnfiyDC_%s_circular.MOV', resfolder, name);

% moviefile = sprintf('%s/red_dark_greenscreen.MOV', expfolder);
% outfile = sprintf('%s/magnify_red_dark_greenscreen_circular.MOV', resfolder);

% ======== PARAMETERS =============% 
startframe = 60*3;
step = 1;
alpha = 20.0; 
blurSz = 4.0;
temporalfactor = .5; 

v = VideoReader(moviefile);
nframes = v.NumberOfFrames;
endframe = nframes-60*3; 


sigma = 5;
temporalfiltsz = 30;    % length of gaussFilter vector
x = linspace(-temporalfiltsz / 2, temporalfiltsz / 2, temporalfiltsz);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter_norm(1,1,1,:) = gaussFilter / sum (gaussFilter); % normalize

% ======== AMPLIFYING =============% 

% calculate the mean image
meanImg = zeros(size(read(v,1))); 
count = 1; 

vout = VideoWriter(outfile);
vout.FrameRate = v.FrameRate / step;
open(vout);
vout_mean = VideoWriter(outfile_mean);
vout_mean.FrameRate = v.FrameRate / step;
open(vout_mean);
vout_noise = VideoWriter(outfile_noise);
vout_noise.FrameRate = v.FrameRate / step;
open(vout_noise);
vout_magnifyDC = VideoWriter(outfile_magnifyDC);
vout_magnifyDC.FrameRate = v.FrameRate / step;
open(vout_magnifyDC);


segVidNan = nan([size(meanImg) temporalfiltsz]); 
for n=startframe:endframe
    n
    
    framen = double(read(v,n));
    segVidNan(:,:,:,end) = framen; 
    
    %meanImg = nanmean( bsxfun(@times, segVidNan, gaussFilter_norm), 4 );
    meanImg = nanmean(segVidNan, 4);
    meanmeanImg(1,1,:) = [ mean(mean(meanImg(:,:,1))), mean(mean(meanImg(:,:,2))), mean(mean(meanImg(:,:,3)))]; 
    segVidNan = circshift(segVidNan,-1,4); 
    
    
    %subtract mean
    meanSub = framen - meanImg; 
    
    % amplify and smooth and add back in
    if n==startframe
        newFrame = meanImg + alpha*imgaussfilt(meanSub,blurSz); 
    else
        newFrame = temporalfactor*oldFrame + (1-temporalfactor)*(meanImg + alpha*imgaussfilt(meanSub,blurSz)); 
    end
    oldFrame = newFrame; 
    
    %save the mean video
    meanImg(meanImg<0) = 0.0;
    meanImg(meanImg>255) = 255.0; 
    writeVideo(vout_mean, uint8(meanImg));
    
    % save the noise video with constnat DC
    meanSubConst = bsxfun(@plus, meanSub, meanmeanImg);  
    meanSubConst(meanSubConst<0) = 0.0;
    meanSubConst(meanSubConst>255) = 255.0; 
    writeVideo(vout_noise, uint8(meanSubConst));
    
    
    % save amplify color video with a mean image
    newFrame(newFrame<0) = 0.0;
    newFrame(newFrame>255) = 255.0; 
    writeVideo(vout, uint8(newFrame));
    
    % save amplify color video with a constnat DC 
    newFrameConst = bsxfun(@plus, newFrame-meanImg, meanmeanImg);  
    newFrameConst(newFrameConst<0) = 0.0;
    newFrameConst(newFrameConst>255) = 255.0; 
    writeVideo(vout_magnifyDC, uint8(newFrameConst));
    
end
close(vout);
close(vout_mean);
close(vout_noise);
close(vout_magnifyDC);


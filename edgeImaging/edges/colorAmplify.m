addpath(genpath('../utils/pyr'));
addpath(genpath('rectify'));


%datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/testvideos_Jan22';
datafolder = '/Users/klbouman/Research/shadowImaging/edgeImaging/data';
expfolder = sprintf('%s/testVideos_Jan29', datafolder);
resfolder = sprintf('%s/results', datafolder);

name = 'blue_location2';
moviefile = sprintf('%s/%s.MOV', expfolder, name);
outfile = sprintf('%s/magnify_%s_circular.MOV', resfolder, name);
outfile_mean = sprintf('%s/mean_%s_circular.png', resfolder, name);
outfile_noise = sprintf('%s/residual_%s_circular.MOV', resfolder, name);
outfile_magnifyDC = sprintf('%s/magnfiyDC_%s_circular.MOV', resfolder, name);


%moviefile = sprintf('%s/redblue_dark.MOV', expfolder);
%outfile = sprintf('%s/magnify_redblue_dark_circular_2.MOV', resfolder);

% ======== PARAMETERS =============% 
startframe = 60*3;
step = 1;
alpha = 40.0; 
blurSz = 10.0;
temporalfactor = .5; 

v = VideoReader(moviefile);
nframes = v.NumberOfFrames;
endframe = nframes-60*3; 


% ======== AMPLIFYING =============% 

% calculate the mean image
meanImg = zeros(size(read(v,1))); 
count = 1; 
for n=startframe:endframe
    n
    meanImg = meanImg + (double(read(v,n)) - meanImg)/(count+1); 
    count = count + 1; 
end
meanmeanImg(1,1,:) = [ mean(mean(meanImg(:,:,1))), mean(mean(meanImg(:,:,2))), mean(mean(meanImg(:,:,3)))]; 
imwrite(meanImg./255, outfile_mean); 


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
    
    % get frame
    origFrame = double(read(v,n));
    
    %subtract mean
    meanSub = origFrame - meanImg; 
    
    % amplify and smooth and add back in
    if n==startframe
        newFrame = meanImg + alpha*imgaussfilt(meanSub,blurSz); 
    else
        newFrame = temporalfactor*oldFrame + (1-temporalfactor)*(meanImg + alpha*imgaussfilt(meanSub,blurSz)); 
    end
    oldFrame = newFrame; 
    
    % save video
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

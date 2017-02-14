addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));

close all; clear;
close all; clear;
close all; clear;


%datafolder = '../data/testvideos/experiment_2';
datafolder = '/Users/klbouman/Research/shadowImaging/edgeImaging/data/testVideos_Jan22/';


%gridfile = sprintf('%s/grid_greenscreen.MOV', datafolder);
%moviefile = sprintf('%s/red_dark_greenscreen.MOV', datafolder);
%outfile = sprintf('%s/out_red_dark_greenscreen.MOV', datafolder);
%background = sprintf('%s/calibration_dark_greenscreen.MOV', datafolder);

gridfile = sprintf('%s/grid_light.MOV', datafolder);
moviefile = sprintf('%s/red_noartificiallight.MOV', datafolder);
outfile = sprintf('%s/out_red_noartificiallight.MOV', datafolder);
background = sprintf('%s/calibration_noartificallight.MOV', datafolder);


startframe = 60; %8*60; 
delta = 1; 
subtract_background = 0;
do_rectify = 1;
nsamples = 400;

vback = VideoReader(background);
endframe = vback.NumberOfFrames-60;

count = 1;
for n=startframe:endframe
    n
    backgroundframes(:,:,:,count) = double(read(vback,n));
    count = count + 1; 
end


signal = mean(backgroundframes, 4); 
noise = bsxfun(@minus, backgroundframes, signal); 

snr_after =  signal .^ 2 ./ mean( noise .^ 2, 4 ); 
snr_after_db = 10 * log10( snr_after );


imwrite(signal./255, '/Users/klbouman/Downloads/signal_noartificallight.png');

figure;imagesc(signal(:,:,1)); colorbar; title('Signal Obtained By Averaging Frames');set(gcf,'color','w'); 
blah2 = getframe(gcf);
imwrite(blah2.cdata, '/Users/klbouman/Downloads/signal_noartificallight_colorbar.png');

figure;imagesc(snr_after_db(:,:,1)); colorbar; title('Estimated SNR in dB'); set(gcf,'color','w');
blah2 = getframe(gcf);
imwrite(blah2.cdata, '/Users/klbouman/Downloads/snr_noartificallight_colorbar.png');

variance = var(backgroundframes, [], 4);
figure;imagesc(variance(:,:,1)); colorbar; title('Estimated Variance in Noise'); set(gcf,'color','w');
blah2 = getframe(gcf);
imwrite(blah2.cdata, '/Users/klbouman/Downloads/var_noartificallight_colorbar.png');

addpath(genpath('../utils'));

close all; clear;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereodoor_Mar06';
datafolder = fullfile(datafolder, 'results');
recon_name = 'floor_1';
left_file = fullfile(datafolder, sprintf('left_%s_tracks.png', recon_name));
right_file = fullfile(datafolder, sprintf('right_%s_tracks.png', recon_name));

write_outvid = 0;
show_depth_over_time = 0;
show_pos_over_time = 0;
outvid_file = fullfile(datafolder, strcat('depth_vid_', recon_name, '.avi'));
depth_img_file = fullfile(datafolder, strcat('depths_over_time_', recon_name));
xypos_img_file = fullfile(datafolder, strcat('xypos_', recon_name));

% tracks are the floor scene, going from pi to 0
left_tracks = double(imread(left_file))./255;
right_tracks = double(imread(right_file))./255;

baseline = 1; % set the size of the baseline. This will just scale the depth
doorwidth = 0.3; % the width of the door, will matter at edges
scaleSigma = 0.25; % set the window size for matching in time
ntestsamp = 100; 

% subtract from 1 so that the background is black instead of white. This
% helps with the matching because convolution assumes its black outside of
% the bounds
right_tracks = 1 - right_tracks;
left_tracks = 1 - left_tracks;

% median filter the trajectories to clean up tracks
nchan = size(left_tracks,3);
for c=1:nchan
    right_tracks(:,:,c) = medfilt2(right_tracks(:,:,c),[5 5]);
    left_tracks(:,:,c) = medfilt2(left_tracks(:,:,c),[5 5]);
end

nsamples = size(left_tracks,2);

left_gray = rgb2gray(left_tracks); 
right_gray = rgb2gray(right_tracks); 

left_angles = linspace(0,pi,nsamples)'; 
right_angles = linspace(pi,0,nsamples)'; 

x = zeros([size(right_tracks,1),1]);
y = zeros(size(x));
x_error = zeros(size(x));
y_error = zeros(size(y));
x_samps = zeros([size(right_tracks,1), ntestsamp]);
y_samps = zeros(size(x_samps));

% loop over each timestep
for f = 1:size(right_tracks,1)   
    % FIND THE MATCH BETWEEN THE TRACKS
    vec = left_gray(f,:)'; 
    left_gaussfits = fit(left_angles,vec,'gauss1');
    if left_gaussfits.a1 < 0.1 % if no track on left side
        left_mu = nan;
    else
        left_mu = left_gaussfits.b1;
    end
    left_sigma = scaleSigma * left_gaussfits.c1;
    left_samps = normrnd(left_mu,left_sigma, [ntestsamp, 1]); 
    %figure; plot(left_guassfits,left_angles,vec) 
    
    vec = right_gray(f,:)'; 
    right_gaussfits = fit(right_angles,vec,'gauss1');
    if right_gaussfits.a1 < 0.1 % if no track on right side
        right_mu = nan;
    else
        right_mu = right_gaussfits.b1;
    end
    right_sigma = scaleSigma * right_gaussfits.c1;
    right_samps = normrnd(right_mu,right_sigma, [ntestsamp, 1]); 
    %figure; plot(right_guassfits,right_angles,vec) 
   
    disparity = cot(left_mu) + cot(right_mu);
    disparity_samps = cot(left_samps) + cot(right_samps);
    
    % calculate the depths, y, and corresponding x-position
    b = baseline;
    if right_mu > pi/2
        b = b - doorwidth * cot(left_mu);
    else if left_mu > pi/2
            b = b - doorwidth * cot(right_mu);
        end
    end
    y(f) = b ./ disparity;
    x(f) = y(f)./tan(left_mu)';
    
    y_samps(f,:) = baseline./disparity_samps; 
    x_samps(f,:) = y_samps(f,:)./tan(left_samps)'; 

    % derivative : https://www.wolframalpha.com/input/?i=derivative+of+b%2F(cot(x)+%2B+cot(a))
    y_error(f) = sqrt( ( ( (baseline*csc(left_mu).^2)./((cot(left_mu) + cot(right_mu)).^2)).^2 * left_sigma.^2 ) + ...
        ( ( (baseline*csc(right_mu).^2)./((cot(left_mu) + cot(right_mu)).^2)).^2 * right_sigma.^2 ) ); 
    
    % derivative: https://www.wolframalpha.com/input/?i=derivative+of+b%2F(1+%2B+tan(a)%2Ftan(x))
    x_error(f) = sqrt( ( (-(baseline*cot(right_mu)*(sec(left_mu).^2))./((cot(right_mu)*tan(left_mu) + 1).^2)).^2  * left_sigma.^2 ) + ...
        ( ((baseline*tan(left_mu)*(csc(right_mu).^2))./((tan(left_mu)/tan(right_mu) + 1).^2) ).^2  * right_sigma.^2 ) );  

    clf; 

    stretchval = size(left_tracks,2)./pi; 
    subplot(221); hold on; imagesc(left_tracks(f,:,:)); plot( stretchval*left_samps, ones(size(left_samps)), '*' ); 

    subplot(223); hold on; imagesc(right_tracks(f,:,:)); plot( stretchval*(pi-right_samps), ones(size(right_samps)) , '*' ); 

    % display the locations of the points based on the shifting
    subplot(122); hold on; 
    plot(x(f),y(f),'r*'); 
    plot(x_samps(f,:),y_samps(f,:),'b*');
    rs = linspace(0, 10, 100);
    plot(rs * cos(left_mu), rs * sin(left_mu));
    plot(1 - rs*cos(right_mu), rs * sin(right_mu));
%     ellipse(x_error(f),y_error(f),0,x(f),y(f),'r');
    axis equal;
    xlim([-10, 10]);
    ylim([0, 10]);
    title('depth estimate of each point');
    
end

if write_outvid
    % VISUALIZE THE MATCHING OF THE TRACKS

    vout = VideoWriter(outvid_file);
    vout.FrameRate = 10;
    open(vout);

    for f = 1:size(right_tracks,1)          
        % display the overlap image (this only is using a rounded shifting)
        % the left track and the approx circularly shifted right track
        clf; 

        stretchval = size(left_tracks,2)./pi; 
        subplot(221); hold on; imagesc(left_tracks(f,:,:)); plot( stretchval*left_samps, ones(size(left_samps)), '*' ); 

        subplot(223); hold on; imagesc(right_tracks(f,:,:)); plot( stretchval*(pi-right_samps), ones(size(right_samps)) , '*' ); 

        % display the locations of the points based on the shifting
        subplot(122); hold on; 
        plot(x(f),y(f),'r*'); 
        plot(x_samps(f,:),y_samps(f,:),'b*');
        ellipse(x_error(f),y_error(f),0,x(f),y(f),'r');
        axis equal;
        xlim([-10, 10]);
        ylim([0, 10]);
        title('depth estimate of each point');

        framepic = getframe(gcf);
        writeVideo(vout, framepic.cdata);
    end
    close(vout);
end 

% DISPLAY THE DEPTHS OVER TIME
if show_depth_over_time
    figure; hold on;
    title('Depths Over Time'); 
    ylabel('Depth with a Baseline of 1');
    xlabel('Time'); 

    for f = 1:size(right_tracks,1)
        plot( f ,y(f) ,'*');
    end
    % ylim([0, 10]);
    saveas(gcf, depth_img_file);
end


% DISPLAY THE POSITIONS OVER TIME
if show_pos_over_time
    figure; hold on;
    title('Position Over Time'); 
    ylabel('Y with a Baseline of 1');
    xlabel('X with a Baseline of 1'); 

    for f = 1:size(right_tracks,1)
        ellipse(x_error(f),y_error(f),0,x(f),y(f), 'r'); axis equal;
        plot( x(f), y(f) ,'*');
    end
    % xlim([0, 10]);
    % ylim([0, 10]);
    axis equal;
    saveas(gcf, xypos_img_file);
end

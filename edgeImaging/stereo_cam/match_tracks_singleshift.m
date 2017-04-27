dir = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb22/results'
name = 'blue_circlewalking_paperfloor';

left_tracks = double(imread(strcat(dir, '/left_', name, '.png')))./255;
right_tracks = double(imread(strcat(dir, '/right_', name, '.png')))./255;

baseline = 1; % set the size of the baseline. This will just scale the depth
winsize = 1; % set the window size for matching in time


% identify the pixels that we care about in the left tracks
valids = (left_tracks(:,:,2)>0.4); 

% subtract from 1 so that the background is black instead of white. This
% helps with the matching because convolution assumes its black outside of
% the bounds
right_tracks = 1 - right_tracks;
left_tracks = 1 - left_tracks;

% medican filter the trajectories to clean up tracks
nchan = size(left_tracks,3);
for c=1:nchan
    right_tracks(:,:,c) = medfilt2(right_tracks(:,:,c),[5 5]);
    left_tracks(:,:,c) = medfilt2(left_tracks(:,:,c),[5 5]);
end


nsamples = size(left_tracks,2);

% loop over each 1D frame in time
for f = 1:size(right_tracks,1)-winsize
    
    % FIND THE MATCH BETWEEN THE TRACKS
    
    % compute the cross correlation for each channel
    for c=1:size(left_tracks,3)
        left_chan = (left_tracks(f:f+winsize-1,:,c));
        right_chan = (right_tracks(f:f+winsize-1,:,c));
        xcorr(:,:,c) = conv2(left_chan, fliplr(right_chan), 'same');
    end
    % sum the cross correlations of each channel and get the middle row corresponding
    xcorr = sum(xcorr,3);
    vec = xcorr(round(winsize/2),:);
    
    % fit the guassian
    guassfits = fit((1:length(vec))',vec(:),'gauss1');
    % plot to see the gaussian fit
    %figure; plot(f,1:length(vec),vec) 
    mu = guassfits.b1;
    sigma = guassfits.c1; % we may want to use this in the future to propogate error
    
    % compute the pixel offset on the angles to use for the right tracks
    idxoffset = mu - nsamples/2;
    pix = (1:size(left_tracks,2)) - idxoffset;
    
    % crop off angles outside of the valid range
    pix(pix>length(pix)) = length(pix);
    pix(pix<1) = 1;
    pix = pix(:); 
    
    
    % COMPUTE THE LOCATIONS
    
    % compute the matching angles
    a2 = interp1( (1:length(vec)) , linspace(pi,0,nsamples) , pix);
    a1 = linspace(0,pi,nsamples)'; 
    disparity = cot(a1(:)) + cot(a2(:));
    
    % calculate the depths, y, and corresponding x-position
    y(f,:) = baseline ./ disparity;
    x(f,:) = y(f,:)./tan(a1)';

    colors = left_tracks(f, :, :);
    
    
    % VISUALIZE THE MATCHING OF THE TRACKS
    
    % display the overlap image - note: this only is using a rounded
    % shifting
    subplot(321);
    fusedimg = imfuse(left_tracks(f,:,:), circshift(right_tracks(f,:,:), [0 round(idxoffset) 0]),'falsecolor','Scaling','joint');
    imagesc(fusedimg);
    
    % display the left track and the circularly shifted approx right track
    subplot(323);imagesc(left_tracks(f,:,:));
    subplot(325);imagesc( circshift(right_tracks(f,:,:), [0 round(idxoffset) 0]) );
    
    % display the locations of the points based on the shifting
    subplot(122);
    scatter(x(f,:),y(f,:),100*ones(size(x(f,:))), squeeze(left_tracks(f, :, :)), 'filled');
    xlim([-10, 10]);
    ylim([0, 10]);
    title('depth estimation of each point');
    
    
end

% DISPLAY THE DEPTHS OVER TIME
figure; hold on;
title('Depths Over Time'); 
ylabel('Depth with a Baseline of 1');
xlabel('Time'); 

for f = 1:size(right_tracks,1)-winsize
    try
        valid = valids(f,:);
        scatter( f*ones(size(y(f,valid))) ,y(f,valid) ,100*ones(size(y(f,valid))), squeeze(left_tracks(f,valid,:)), 'filled');
    end
end




% DISPLAY THE POSITIONS OVER TIME
figure; hold on;
title('Position Over Time'); 
ylabel('Y with a Baseline of 1');
xlabel('X with a Baseline of 1'); 

for f = 1:size(right_tracks,1)-winsize
    try
        %valid = sum((left_tracks(f, :, :) - matching_color).^2); %
        valid = valids(f,:);
        meanX = mean(x(f,valid));
        meanY = mean(y(f,valid));
        plot( meanX, meanY ,'*');
        xlim([-5 5]);
        ylim([0 10]);
        %pause(0.1);
    end
end

dir = '/data/vision/billf/shadowImaging/edgeImaging/data/stereoDoor_Feb22/results'
name = 'blue_circlewalking_paperfloor';

left_tracks = double(imread(strcat(dir, '/left_', name, '.png')))./255;
right_tracks = double(imread(strcat(dir, '/right_', name, '.png')))./255;

baseline = 1; % set the size of the baseline. This will just scale the depth
scaleSigma = 0.25; % set the window size for matching in time


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

left_gray = rgb2gray(left_tracks); 
right_gray = rgb2gray(right_tracks); 

left_angles = linspace(0,pi,nsamples)'; 
right_angles = linspace(pi,0,nsamples)'; 

% loop over each 1D frame in time
for f = 1:size(right_tracks,1)-winsize
    
    % FIND THE MATCH BETWEEN THE TRACKS
    vec = left_gray(f,:)'; 
    left_guassfits = fit(left_angles,vec,'gauss1');
    left_mu = left_guassfits.b1;
    left_sigma = scaleSigma * left_guassfits.c1;
    %figure; plot(left_guassfits,left_angles,vec) 
    
    vec = right_gray(f,:)'; 
    right_guassfits = fit(right_angles,vec,'gauss1');
    right_mu = right_guassfits.b1;
    right_sigma = scaleSigma * right_guassfits.c1;
    %figure; plot(right_guassfits,right_angles,vec) 
   
    disparity = cot(left_mu) + cot(right_mu);
    
    % calculate the depths, y, and corresponding x-position
    y(f) = baseline ./ disparity;
    x(f) = y(f)./tan(left_mu)';

    % calculating the derivative : https://www.wolframalpha.com/input/?i=derivative+of+b%2F(cot(x)+%2B+cot(a))
    y_error(f) = sqrt( ( ( (baseline*csc(left_mu).^2)./((cot(left_mu) + cot(right_mu)).^2)).^2 * left_sigma.^2 ) +  ...
        ( ( (baseline*csc(right_mu).^2)./((cot(left_mu) + cot(right_mu)).^2)).^2 * right_sigma.^2 ) ); 
    
    % calculating the derivative: https://www.wolframalpha.com/input/?i=derivative+of+b%2F(1+%2B+tan(x)%2Ftan(a))
    % calculating the derivative: https://www.wolframalpha.com/input/?i=derivative+of+b%2F(1+%2B+tan(a)%2Ftan(x))
    x_error(f) = sqrt( ( (-(baseline*cot(right_mu)*(sec(left_mu).^2))./((cot(right_mu)*tan(left_mu) + 1).^2)).^2  * left_sigma.^2 )   + ...
        ( ((baseline*tan(left_mu)*(csc(right_mu).^2))./((tan(left_mu)/tan(right_mu) + 1).^2) ).^2  * right_sigma.^2 ) );  
    
    % VISUALIZE THE MATCHING OF THE TRACKS
   
    % display the overlap image - note: this only is using a rounded
    % shifting
    % display the left track and the circularly shifted approx right track
    clf; 
    subplot(221);imagesc(left_tracks(f,:,:));
    subplot(223);imagesc( right_tracks(f,:,:) );
    
    % display the locations of the points based on the shifting
    subplot(122);
    plot(x(f),y(f),'*');
    hold on; 
    ellipse(x_error(f),y_error(f),0,x(f),y(f),'r'); axis equal;
    xlim([-10, 10]);
    ylim([0, 10]);
    title('depth estimation of each point');
    %pause(0.1); 
    
end

% DISPLAY THE DEPTHS OVER TIME
figure; hold on;
title('Depths Over Time'); 
ylabel('Depth with a Baseline of 1');
xlabel('Time'); 

for f = 1:size(right_tracks,1)-winsize
    try
        plot( f ,y(f) ,'*');
    end
end




% DISPLAY THE POSITIONS OVER TIME
figure; hold on;
title('Position Over Time'); 
ylabel('Y with a Baseline of 1');
xlabel('X with a Baseline of 1'); 

for f = 1:size(right_tracks,1)-winsize
    try
        plot( x(f), y(f) ,'*');
        ellipse(x_error(f),y_error(f),0,x(f),y(f),'r'); axis equal;
    end
end

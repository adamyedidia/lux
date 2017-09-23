function matchTracks(params)
% matches tracks given in params.left_file, params.right_file, and
% writes the output values to params.outfile

% tracks are the floor scene, going from pi to 0
% subtract from 1 so tracks are 1, not 0
left_tracks = 1 - double(imread(params.left_file))./255;
right_tracks = 1 - double(imread(params.right_file))./255;

% clean up tracks with median filter
nchan = size(left_tracks,3);
for c=1:nchan
    right_tracks(:,:,c) = medfilt2(right_tracks(:,:,c),[5 5]);
    left_tracks(:,:,c) = medfilt2(left_tracks(:,:,c),[5 5]);
end

left_gray = rgb2gray(left_tracks); 
right_gray = rgb2gray(right_tracks);
nsamples = size(left_tracks,2);

left_angles = linspace(0, pi, nsamples)'; 
right_angles = linspace(pi, 0, nsamples)'; 
        
x = zeros([size(right_gray,1),1]);
y = zeros(size(x));
left_mus = zeros(size(x));
left_sigmas = zeros(size(x));
right_mus = zeros(size(x));
right_sigmas = zeros(size(x));
x_samps = zeros([size(right_gray,1), params.ntestsamp]);
y_samps = zeros(size(x_samps));

% loop over each timestep
for f = 1:params.timestep:size(right_gray,1)
    % FIND THE MATCH BETWEEN THE TRACKS
    vec = left_gray(f,:)'; 
    left_gaussfits = fit(left_angles, vec,'gauss1');
    if left_gaussfits.a1 < 0.1 % if no track on left side
        left_mu = nan;
    else
        left_mu = left_gaussfits.b1;
    end
    left_sigma = params.scaleSigma * left_gaussfits.c1;
    left_samps = normrnd(left_mu, left_sigma, [params.ntestsamp, 1]); 
    
    vec = right_gray(f,:)';
    right_gaussfits = fit(right_angles, vec,'gauss1');
    if right_gaussfits.a1 < 0.1 % if no track on right side
        right_mu = nan;
    else
        right_mu = right_gaussfits.b1;
    end
    right_sigma = params.scaleSigma * right_gaussfits.c1;
    right_samps = normrnd(right_mu, right_sigma, [params.ntestsamp, 1]); 
    
    disparity = cot(left_mu) + cot(right_mu);
    disparity_samps = cot(left_samps) + cot(right_samps);

    % calculate the depths, y, and corresponding x-position
    b = params.baseline;
    doorshift = 0;
    if right_mu > pi/2
        fprintf('right_mu big, right_mu %.3f, left_mu %.3f\n', right_mu, left_mu);
        b = b - params.doorwidth * cot(left_mu);
        doorshift = params.doorwidth;
    else if left_mu > pi/2
            fprintf('left_mu big, right_mu %.3f, left_mu %.3f\n', right_mu, left_mu);
            b = b - params.doorwidth * cot(right_mu);
            doorshift = params.doorwidth;
        end
    end
    y(f) = b ./ disparity + doorshift;
    x(f) = y(f)./tan(left_mu)';
    
    y_samps(f,:) = b ./disparity_samps + doorshift; 
    x_samps(f,:) = y_samps(f,:)./tan(left_samps)'; 
    
    left_mus(f) = left_mu;
    right_mus(f) = right_mu;
    left_sigmas(f) = left_sigma;
    right_sigmas(f) = right_sigma;
    
    % SHOW EACH MATCH
    if params.plot
        clf;
        % display the locations of the points based on the shifting
        subplot(111); hold on; 
        plot(x(f),y(f),'r*'); 
        
        if ~isnan(x(f)) && ~isnan(y(f))
            samp_cov = cov(x_samps(f,:), y_samps(f,:));
            error_ellipse(samp_cov, [x(f), y(f)]);
        end
        
        rs = linspace(0, 2*params.ymax, 100);
        plot(rs * cos(left_mu), rs * sin(left_mu));
        plot(baseline - rs*cos(right_mu), rs * sin(right_mu));
        if isfield(params, 'gt_depth')
            numx = (params.xmax - params.xmin + 1);
            plot((params.xmin:params.xmax), ones([numx,1]) * params.gt_depth);
        end
        xlim([params.xmin, params.xmax]);
        ylim([params.ymin, params.ymax]);
        title('depth estimate of each point');
    end
end

save(params.outfile,...
    'x', 'y', 'x_samps', 'y_samps',...
    'left_tracks', 'right_tracks',...
    'left_mus', 'right_mus',...
    'left_sigmas', 'right_sigmas');
end
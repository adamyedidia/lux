addpath(genpath('../utils'));

close all; clear;

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereoDoor_Mar08';
datafolder = fullfile(datafolder, 'results');
recon_name = 'floor_location4_dim_iso640';

params.left_file = fullfile(datafolder, sprintf('left_%s_tracks.png', recon_name));
params.right_file = fullfile(datafolder, sprintf('right_%s_tracks.png', recon_name));
params.outfile = fullfile(datafolder, sprintf('out_%s.mat', recon_name));

params.baseline = 1; % set the size of the baseline. This will just scale the depth
params.doorwidth = (6+5/8)/(30+5/8); % the width of the door, will matter at edges
params.gt_depth = 2; % the ground truth depth (or what we expect)

params.scaleSigma = 0.25; % set the window size for matching in time
params.ntestsamp = 100; 
params.timestep = 5;

params.plot = 0;
params.xmin = -3;
params.xmax = 4;
params.ymin = 0;
params.ymax = 7;

match_tracks(params);

load(params.outfile);

write_outvid = 0;
show_depth_over_time = 0;
show_pos_over_time = 0;
outvid_file = fullfile(datafolder, strcat('depth_vid_', recon_name, '.avi'));
depth_img_file = fullfile(datafolder, strcat('depths_over_time_', recon_name));
xypos_img_file = fullfile(datafolder, strcat('xypos_', recon_name));

if write_outvid
    % VISUALIZE THE MATCHING OF THE TRACKS

    vout = VideoWriter(outvid_file);
    vout.FrameRate = 10;
    open(vout);

    for f = 1:timestep:size(right_tracks,1)          
        % display the overlap image (this only is using a rounded shifting)
        % the left track and the approx circularly shifted right track
        clf;
        stretchval = size(left_tracks,2)./pi; 
        subplot(221); imagesc(left_tracks(f,:,:));
%         hold on; plot(stretchval*left_samps, ones(size(left_samps)), '*'); 

        subplot(222); imagesc(right_tracks(f,:,:));
%         hold on; plot(stretchval*(pi - right_samps), ones(size(right_samps)), '*' ); 

        % display the locations of the points based on the shifting
        subplot(212); hold on; 
        plot(x(f),y(f),'r*'); 
%         plot(x_samps(f,:), y_samps(f,:), 'b*');
        error_ellipse(samp_cov, [x(f), y(f)]);
        rs = linspace(0, 2*maxdispy, 100);

        plot(rs * cos(left_mus(f)), rs * sin(left_mus(f)));
        plot(baseline - rs*cos(right_mus(f)), rs * sin(right_mus(f)));
        
        if isfield(params, 'gt_depth')
            numx = (params.xmax - params.xmin + 1);
            plot((params.xmin:params.xmax), ones([numx,1]) * params.gt_depth);
        end
        xlim([params.xmin, params.xmax]);
        ylim([params.ymin, params.ymax]);
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

    for f = 1:timestep:size(right_tracks,1)
        plot(f, y(f), '*');
    end
    % ylim([0, 10]);
    saveas(gcf, depth_img_file);
end

plot_err = 1;
% DISPLAY THE POSITIONS OVER TIME
if show_pos_over_time
    figure; hold on;
    title('Position Over Time'); 
    ylabel('Y with a Baseline of 1');
    xlabel('X with a Baseline of 1'); 
    
    count = 1;
    for f = 1:timestep:size(right_tracks,1)
        if isnan(x(f)) || isnan(y(f))
            continue;
        end
        if plot_err && mod(count, 10) == 1
            samp_cov = cov(x_samps(f,:), y_samps(f,:));
            eigvals = eig(samp_cov);
            if max(eigvals(:)) > 6
                continue;
            end
            error_ellipse(samp_cov, [x(f), y(f)]);
        end
        plot( x(f), y(f) ,'*'); hold on;
        count = count + 1;
    end
%     axis([-6, 1, 0, 6]);
%     axis([-2, 3, 0, maxdispy]);
%     saveas(gcf, xypos_img_file);
end

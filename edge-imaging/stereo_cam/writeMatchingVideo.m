function writeMatchingVideo(params)
load(params.outfile);

vout = VideoWriter(params.outvid_file);
vout.FrameRate = 10;
open(vout);

for f = 1:params.timestep:size(x,1)          
    % display the overlap image (this only is using a rounded shifting)
    % the left track and the approx circularly shifted right track
    clf;
    stretchval = size(left_tracks, 2)./pi; 
    subplot(221); imagesc(left_tracks(f,:,:));
    hold on; plot(stretchval*left_samps, ones(size(left_samps)), '*'); 

    subplot(222); imagesc(right_tracks(f,:,:));
    hold on; plot(stretchval*(pi - right_samps), ones(size(right_samps)), '*' ); 

    % display the locations of the points based on the shifting
    subplot(212); hold on; 
    plot(x(f),y(f),'r*');
    if ~isnan(x(f)) && ~isnan(y(f))
        plot(x_samps(f,:), y_samps(f,:), 'b.');
        samp_cov = cov(x_samps(f,:), y_samps(f,:));
        error_ellipse(samp_cov, [x(f), y(f)]);
    end

    rs = linspace(0, 2*params.ymax, 100);
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
addpath(genpath('../utils'))

datafolder = '/Users/vickieye/Dropbox (MIT)/shadowImaging/edgeImaging/data/stereoDoor_Mar08';
datafolder = fullfile(datafolder, 'results');

pointstyle = {'b*', 'g*', 'c*', 'm*'};
linestyle = {'b', 'g', 'c', 'm'};
xmin = -2*20;
xmax = 3*20;
ymin = 0*20;
ymax = 7*20;

figure; hold on;
for i = 1:4
    recon_name = sprintf('floor_location%d_dim_iso640', i);
    fprintf('adding %s to plot\n', recon_name');
    outfile = fullfile(datafolder, sprintf('out%s.mat', recon_name));
    load(outfile, 'x', 'y', 'x_samps', 'y_samps', 'timestep');
    x = 20*x;
    y = 20*y;
    x_samps = 20*x_samps;
    y_samps = 20*y_samps;
    
    plot(x, y, pointstyle{i});

    count = 0;
    for f = 1:timestep:size(x,1)
        count = count + 1;
        if isnan(x(f)) || isnan(y(f)) || mod(count, 4) ~= 1
            continue;
        end
        samp_cov = cov(x_samps(f,:), y_samps(f,:));
%         eigvals = eig(samp_cov);
%         if max(eigvals(:)) > 6*20
%             continue;
%         end
        error_ellipse(samp_cov, [x(f), y(f)], 'style', 'r');
    end
    axis([xmin, xmax, ymin, ymax]);
    plot((xmin:xmax), ones([xmax-xmin+1,1]) * 20*i, linestyle{i});
end

title('Position Over Time'); 
ylabel('Y with a Baseline of 20');
xlabel('X with a Baseline of 20'); 
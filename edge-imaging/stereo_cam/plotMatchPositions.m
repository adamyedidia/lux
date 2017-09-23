function plotMatchPositions(params)
load(params.outfile);

figure; hold on;

xp = x(1:params.timestep:end);
yp = y(1:params.timestep:end);
xp_samps = x_samps(1:params.timestep:end,:);
yp_samps = y_samps(1:params.timestep:end,:);

colors = jet(length(xp));

for i= 1:length(xp)-1
    if isnan(xp(i)) || isnan(yp(i))
        continue;
    end
    if mod(i, 2) == 1
        samp_cov = cov(xp_samps(i,:), yp_samps(i,:));
        [evecs, evals] = eig(samp_cov);
        if max(evals(:)) > 10
            continue;
        end
        ang = atan2(evecs(2,1), evecs(1,1));
        ellipse(sqrt(evals(1,1)), sqrt(evals(2,2)), ang, xp(i), yp(i), colors(i,:));
    end
end

% plot the track
for i = 1:length(xp)-1
    plot([xp(i) xp(i+1)], [yp(i) yp(i+1)], '-', 'Color', colors(i,:), 'LineWidth', 7);
end

% scatter(xp, yp, 100*ones(size(xp)), colors, 'filled'); 
axis equal; 

% plot the door
plot([-10 0], [0 0], 'k', 'LineWidth', 7);
plot([1 11], [0 0], 'k', 'LineWidth', 7);
text(0.4,0.2,'\downarrow', 'FontSize', 20)
text(-0.3,0.6,'Baseline', 'FontSize', 20); 

axis([-5 2 0 7]);
axis off; 
set(gcf, 'color', 'w');
end
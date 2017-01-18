function [pixel_avg, diffs] = anglesAveragePixel(frame, corner, nsamples, maxr)
% calculate the average pixel in every evenly spaced sector
% we then take the differences between these samples

% calculate the angle each pixel is at with the corner/wall
[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
x0 = xx - corner(1);
y0 = yy - corner(2);
% zero out pixels that are more than max_r from corner
rmask = x0.^2 + y0.^2 <= maxr^2;
theta = atan2(y0, x0);
angles = linspace(0, pi/2, nsamples+2);
angles = angles(2:end-1);
eps = (angles(2) - angles(1))/2;
pixel_avg = zeros([nsamples, size(frame, 3)]);
for c = 1:size(frame, 3)
    color = frame(:,:,c) .* rmask;
%     imagesc(color);
    for i = 1:nsamples
        adiff = theta - angles(i);
        mask = abs(adiff) <= eps;
%         imagesc(mask);
        pixel_avg(i,c) = mean(color(mask));
    end
end
diffs = diff(pixel_avg);
end
function [pixel_avg, diffs] = anglesAveragePixel(frame, corner, nsamples,...
                                                    maxr, mweight)
% calculate the average pixel in every evenly spaced sector
% we then take the differences between these samples

if nargin < 5
    mweight = 'normal';
end

% calculate the angle each pixel is at with the corner/wall
[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
x0 = xx - corner(1);
y0 = yy - corner(2);
% zero out pixels that are more than max_r from corner
rmask = x0.^2 + y0.^2 <= maxr^2;
theta = atan2(y0, x0);
angles = linspace(0, 3*pi/8, nsamples+2);
angles = angles(2:end-1);
eps = (angles(2) - angles(1))/2;
pixel_avg = zeros([nsamples, size(frame, 3)]);
% figure;
for c = 1:size(frame, 3)
    color = frame(:,:,c) .* rmask;
%     color(~rmask) = nan;
%     imagesc(color);
    for i = 1:nsamples
        adiff = theta - angles(i);
        if strcmp(mweight, 'normal') == 1
            mask = exp(-adiff.^2/(2*eps^2));
            mask(adiff > 2*eps) = 0;
        else if strcmp(mweight, 'hinge') == 1        
                mask = 1 - abs(adiff)/(2*eps);
                mask(mask < 0) = 0;
            else if strcmp(mweight, 'step') == 1
                    mask = abs(adiff) < eps;
                end
            end
        end
        mask = mask / sum(mask(:));
%         imagesc(mask); pause(0.1);
        pixel_avg(i,c) = sum(sum(color .* mask));
    end
end
diffs = diff(pixel_avg);
end
function [pixel_avg, diffs] = estimatedGradient(frame, corner, nsamples, maxr)
[nrows, ncols, nchans] = size(frame);
rate = 0.1;
[yy, xx] = ndgrid(1:rate:nrows, 0:rate:ncols-1); % indexing by top-left corner
x0 = xx - corner(1);
y0 = yy - corner(2);
% zero out pixels that are more than max_r from corner
[yy, xx] = ndgrid(1:nrows, 1:ncols);
rr = sqrt((xx - corner(1)).^2 + (yy - corner(2)).^2);
rmask = rr <= maxr;
theta = atan2(y0, x0);
angles = linspace(0, 3*pi/8, nsamples+2);
angles = angles(2:end-1);
pixel_avg = zeros([nsamples, nchans]);
% v = VideoWriter('mask_vid_frac');
% v.FrameRate = 10;
% open(v);
for c = 1:nchans
    color = frame(:,:,c) .* rmask;
    for i = 2:nsamples
        mask2 = double(theta <= angles(i));
        mask1 = double(theta <= angles(i-1));
        mask = imresize(mask2 - mask1, rate, 'box') ./ rr;
        mask = mask / sum(mask(:));
%         imagesc(mask);
%         frame = getframe(gcf);
%         writeVideo(v, frame.cdata);
        pixel_avg(i,c) = sum(sum(color .* mask));
    end
end
% close(v);
diffs = diff(pixel_avg);
end

function px_frac = fractionOfPixel(nrows, ncols, samples, theta0)
r = linspace(0, nrows, 200);
rate = 1/samples;
[yy, xx] = ndgrid(1:rate:nrows, 1:rate:ncols);
theta = atan2(yy, xx);
mask = double(theta <= theta0);
figure; imagesc(mask); hold on; plot(r*cos(theta0), r*sin(theta0)); 
px_frac = imresize(mask, rate, 'box');
figure; imagesc(px_frac); hold on; plot(r*cos(theta0), r*sin(theta0));
end

function px_frac = fractionOfPixel0(x1, y1, x2, y2, theta)
% calculates the fraction of the pixel in the sector from 0 to theta

% compute the area of top-left corner that's cut off and
% the bottom-right corner that's included
r = linspace(0, max(x2(:)), 200);

px_frac = ones(size(x1));
% distance between top-left and where ray intersects x1
dy1 = x1 * tan(theta) - y1;
% distance between bottom-right and where ray intersects x2
dy2 = x2 * tan(theta) - y2;

px_frac(-dy1 > 1) = 0;
figure; imagesc(px_frac); hold on; plot(r*cos(theta), r*sin(theta));
% only concerned with pixels the ray intersects
figure; imagesc(dy1); hold on; plot(r*cos(theta), r*sin(theta));
dy1(abs(dy1) > 1) = 0;
figure; imagesc(dy1); hold on; plot(r*cos(theta), r*sin(theta));
% area of the top-left corner that's cut off
a1 = 0.5 * cot(theta) * dy1.^2;
figure; imagesc(a1); hold on; plot(r*cos(theta), r*sin(theta));
px_frac = px_frac - a1; % subtracting the top-left corner


% only concerned with pixels the ray intersects
dy2(abs(dy2) > 1) = 0;
figure; imagesc(dy2); hold on; plot(r*cos(theta), r*sin(theta));
a2 = 0.5 * cot(theta) * dy2.^2;
figure; imagesc(a2); hold on; plot(r*cos(theta), r*sin(theta));
px_frac = px_frac + a2; % adding the bottom-right corner
figure; imagesc(px_frac); hold on; plot(r*cos(theta), r*sin(theta));
end
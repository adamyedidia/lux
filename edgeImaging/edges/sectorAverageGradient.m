function avg_diffs = sectorAverageGradient(frame, corner, nsamples, maxr)
[nrows, ncols, nchans] = size(frame);
[yy, xx] = ndgrid(1:nrows, 1:ncols);
x0 = xx - corner(1);
y0 = yy - corner(2);
rr = sqrt(x0.^2 + y0.^2); % distance from corner
rmask = rr <= maxr;

% compute unit vector along circular gradient at every pixel
ux = -y0 ./ sqrt(x0.^2 + y0.^2);
uy = x0 ./ sqrt(x0.^2 + y0.^2);

theta = atan2(y0, x0); % to average over each sector
angles = linspace(0, 3*pi/8, nsamples+2);
angles = angles(2:end-1);
avg_diffs = zeros([nsamples, nchans]);
for c = 1:nchans
    color = frame(:,:,c);
    % for every pixel, interpolate the next pixel along the gradient
    nextpx = interp2(x0, y0, color, x0+ux, y0+uy);
    pxdiffs = (nextpx - color) .* rmask;
    pxdiffs(isnan(pxdiffs)) = 0;
%     figure; imagesc(pxdiffs);
    for i = 2:nsamples
        mask2 = double(theta <= angles(i));
        mask1 = double(theta <= angles(i-1));
        mask = (mask2 - mask1) ./ rr;
        mask = mask / sum(mask(:));
%         imagesc(mask); pause(0.1);
        avg_diffs(i, c) = sum(sum(pxdiffs .* mask));
    end
end
end
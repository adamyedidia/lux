function [rgbq, diffs] = gradientAlongCircle(frame, corner, rs, nsamples, theta_lim)
if size(rs, 1) ~= 1 % not a row vector
    rs = rs';
end
angles = linspace(theta_lim(1), theta_lim(2), nsamples)';
xq = corner(1) + cos(angles) * rs;
yq = corner(2) + sin(angles) * rs;
% hold on; plot(xq, yq);
[nrows, ncols, nchans] = size(frame);
[yy, xx] = ndgrid(1:nrows, 1:ncols);
rgbq = zeros([nsamples, nchans, length(rs)]);
for i = 1:nchans
    rgbq(:,i,:) = interp2(xx, yy, frame(:,:,i), xq, yq);
end

diffs = diff(rgbq)/ (200/nsamples);
end
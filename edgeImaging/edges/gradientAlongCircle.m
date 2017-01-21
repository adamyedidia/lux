function [rgbq, diffs] = gradientAlongCircle(frame, corner, r, nsamples,...
                                                theta_lim)
if nargin < 5
    theta_lim = [0, pi/2];
end
angles = linspace(theta_lim(1), theta_lim(2), nsamples);
xq = corner(1) + r * cos(angles);
yq = corner(2) + r * sin(angles);

[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
rgbq = zeros([nsamples, size(frame, 3)]);
for i = 1:size(rgbq,2)
    rgbq(:,i) = interp2(xx, yy, frame(:,:,i), xq, yq);
end

diffs = diff(rgbq);%/ (200/nsamples);
end
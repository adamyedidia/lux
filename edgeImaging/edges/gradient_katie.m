function [rgbq, diffs] = gradientAlongCircle(r, nsamples, frame, corner)
angles = linspace(0, pi/2, nsamples);
xq = corner(1) + r * cos(angles);
yq = corner(2) + r * sin(angles);

[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
rgbq = zeros([nsamples, size(frame, 3)]);
for i = 1:size(rgbq,2)
    rgbq(:,i) = interp2(xx, yy, frame(:,:,i), xq, yq);
end

rgbq = rgbq(end:-1:1,:,:,:); 
diffs = diff(rgbq);
end
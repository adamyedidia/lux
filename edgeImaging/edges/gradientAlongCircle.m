function [rgbq, diffs] = gradientAlongCircle(r, nsamples, frame, corner)
angles = linspace(0, pi/2, nsamples);
xq = corner(1) + r * cos(angles);
yq = corner(2) + r * sin(angles);

[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
rgbq = zeros([nsamples, size(frame, 3)]);
for i = 1:size(rgbq,2)
    rgbq(:,i) = interp2(xx, yy, frame(:,:,i), xq, yq);
end

out = repmat(reshape(rgbq, [1, nsamples, 3]), [nsamples, 1, 1]);
% figure; imagesc(uint8(out)); title(sprintf('r=%d stripe'));
diffs = diff(rgbq);
for i = 1:3
    minv = min(diffs(:,i));
    maxv = max(diffs(:,i));
    diffs(:, i) = 2^8*(diffs(:, i) - minv)/(maxv - minv);
end
outdiffs = repmat(reshape(diffs, [1, nsamples-1, 3]), nsamples);
figure; imagesc(uint8(outdiffs)); title(sprintf('r=%d gradient'));
end
function [strip, diffs] = gradientAlongCircle(r, nsamples, frame, corner)
angles = linspace(0, pi/2, nsamples);
xq = corner(1) + r * cos(angles);
yq = corner(2) + r * sin(angles);

[yy, xx] = ndgrid(1:size(frame, 1), 1:size(frame, 2));
strip = zeros([nsamples, size(frame, 3)]);
for i = 1:size(strip,2)
    strip(:,i) = interp2(xx, yy, frame(:,:,i), xq, yq);
end

% out = repmat(reshape(strip, [1, nsamples, 3]), [nsamples, 1, 1]);
% figure; imagesc(uint8(out)); title(sprintf('r=%d stripe'));
diffs = diff(strip);
% for i = 1:3
%     minv = min(diffs(:,i));
%     maxv = max(diffs(:,i));
%     diffs(:, i) = 2^8*(diffs(:, i) - minv)/(maxv - minv);
% end
outdiffs = repmat(reshape(diffs, [1, nsamples-1, 3]), nsamples);
figure; imagesc(uint8(outdiffs)); title(sprintf('r=%d gradient', r));
end
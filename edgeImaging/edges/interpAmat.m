function amat = interpAmat(nrows, ncols, r, nsamples, theta_lim)
% the transfer matrix we infer through interpolation
% we assume the corner is at (1/2, 1/2)
% we use points in the middle of the pixel, (x+1/2, y+1/2)
if nargin < 5
    theta_lim = [0, pi/2];
end
angles = linspace(theta_lim(1), theta_lim(2), nsamples);
xq = r * cos(angles) + 1/2;
yq = r * sin(angles) + 1/2;

% find the four pixel indices corresponding to each xq, yq
ix1 = floor(xq - 0.5) + 1;
iy1 = floor(yq - 0.5) + 1;
ix2 = ix1 + 1;
iy2 = iy1 + 1;

% our frame is flattened into nrows * ncols
interpmat = zeros([nrows, ncols, nsamples]);
% eventually we want the finite difference, which is 
% interp val1 - interp val2 at consecutive angles

% first interp val at angle 1
% put weights for each point used in interpolation at each angle
% i11 = nrows * (ix1-1) + iy1 + nrows * ncols * (0:nsamples-1);
i11 = sub2ind(size(interpmat), iy1, ix1, (1:nsamples));
i21 = sub2ind(size(interpmat), iy1, ix2,(1:nsamples));
i12 = sub2ind(size(interpmat), iy2, ix1, (1:nsamples));
i22 = sub2ind(size(interpmat), iy2, ix2, (1:nsamples));
% pixel midpoints are (ix - 0.5, iy - 0.5)
interpmat(i11) = (ix2-0.5 - xq) .* (iy2-0.5 - yq);
interpmat(i21) = (xq - (ix1-0.5)) .* (iy2-0.5 - yq);
interpmat(i12) = (ix2-0.5 - xq) .* (yq - (iy1-0.5));
interpmat(i22) = (xq - (ix1-0.5)) .* (yq - (iy1-0.5));

for i = 1:nsamples
    imagesc(interpmat(:,:,i)); pause(0.2);
end

% amat is difference between the interped vals at successive angles
amat = diff(interpmat, 1, 3);
size(amat)
amat = reshape(amat, [nrows*ncols, nsamples-1])';
end

function testAmat(amat)
unflattened = reshape(amat, [nsamples-1, nrows, ncols]);
end
function [ pos ] = sceneIndexToPos(params, ix, jz, sceneImageSize )
% convert from scene image index to 3d position
% This function also works for arrays input into i and j.
% Lets center each position within the square pixel.
% Dec. 10, 2016  billf created.

r = params.sceneToCorner;
% Gives the theta for the position of the scene element.
theta = params.sceneRenderThetaMax - ...
    (ix - 0.5) .* params.sceneRenderDeltaTheta ./ sceneImageSize(2);
% get x and y for this theta
x = cos(theta) * r;
y = sin(theta) * r;

% find the pixel size.  assume square pixels.  get the radial width per
% pixel
sceneDistPerPixel = r * params.sceneRenderDeltaTheta ...
    ./ sceneImageSize(2);

% now get the z-value
z = (jz -0.5) * sceneDistPerPixel + params.sceneVerticalOffset;

% now put it all together into pos.
pos = zeros([size(x,1), size(x,2), 3]);
pos(:,:,1) = x;
pos(:,:,2) = y;
pos(:,:,3) = z;


end


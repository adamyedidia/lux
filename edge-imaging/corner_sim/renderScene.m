function [ im, pos ] = renderScene( params, scene )
% Render a scene image onto the corner's image plane.
% scene is a 3-color image, of dimensions
%   Dec. 10, 2016  billf created.

% ix, jz are the meshgrid positions of the image
% clr is the color image array.


[xx, zz] = meshgrid(1:size(scene,2), 1:size(scene,1));
pos = sceneIndexToPos(params, xx, zz, size(scene) );

% need to use sceneIndexToPos
% Since this renders the entire 2d floor image all at once, I need to
% render each scene image point one point at a time, looping through, and
% adding up each image into the cumulative image.
pp = reshape(pos, size(pos,1)*size(pos,2),3)';
ss = reshape(scene, size(scene,1)*size(scene,2), 3)';

im = zeros(params.imageH, params.imageV,3);
for i = 1:size(pp,2)
    im = im + renderPt(params, ss(:,i), pp(:,i));
end


end


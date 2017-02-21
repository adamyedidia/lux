function [ amatOut ] = getAmat(params, scene)
% getAmat constructs the transfer matrix
% telling what the image is for each point of light turned on in the estimated image.
% Dec. 8, 2017  billf created.
% Dec. 21, 2016  billf:  to avoid aliasing effects, I need to blur out the
% light from any individual point in the scene--to make the light come from
% a blob, not from an infinitesimal point.  params.amatBlurKernel  specifies
% the blur, and amatUpsampleFactor, which indicates the horizontal upsampling factor.
% The procedure:  upsample the sampling grid.  get the amat for that scene sampling.  then convolve
% and subsample the columns of the amat by the blur kernel to get the new
% amat.  For now:  only allow horizontal blur kernels.

% scene is an image, just used to find its x and y dimensions for computing
% where the scene image points go in the world behind the corner.
% the output resulting amat is just for a single black and white plane.
% all the result of the software assumes r,g,b.

% geometry:  params.sceneToCorner sets the radius distance that the
% cylinder on which the scene image is wrapped around is from the corner
% 0,0,0 position.  the pixels of the scene image are assumed to be square.
% In addition, there is an additive offset in the height of the scene
% image, by params.sceneVerticalOffset.  This allows a 1-d scene to be not
% right on the ground.

% for the  transfer matrix calculation, just use an image of 1's
scene = ones(size(scene));
scene = repmat(scene, [1, params.amatUpsampleFactor]);

[xx, zz] = meshgrid(1:size(scene,2), 1:size(scene,1));
pos = sceneIndexToPos(params, xx, zz, size(scene) );

% need to use sceneIndexToPos
% Since this renders the entire 2d floor image all at once, I need to
% render each scene image point one point at a time, looping through, and
% adding up each image into the cumulative image.
pp = reshape(pos, size(pos,1)*size(pos,2),3)';
ss = reshape(scene, size(scene,1)*size(scene,2), 3)';

amat = zeros(params.imageH, params.imageV, size(pp,2));
for i = 1:size(pp,2)
   tmp =  renderPt(params, ss(:,i), pp(:,i));
   amat(:,:,i) = tmp(:,:,1);
end

% now blur the appropriate columns of the amat by params.amatBlurKernel
amatOut = ...
    reshape(amat, [params.imageH, params.imageV, size(xx,1), size(xx,2)]);

fourdblur = shiftdim(params.amatBlurKernel, -2);
amatOut = convn(amatOut, fourdblur, 'same');
amatOut = amatOut(:,:,:,1:params.amatUpsampleFactor:end);

end

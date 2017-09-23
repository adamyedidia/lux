npix = 256;
l = 10; % length of scene and obs planes
d = 20; % dist from scene to occluder
D = 40; % dist from scene to obs
r = 1; % radius of occluding circle
c = [l/2, l/2, d]; % center of occluding circle

scene_bounds = [0 0 0; l 0 0; l l 0; 0 l 0];
obs_bounds = scene_bounds;
obs_bounds(:, 3) = D;

occluder((xx - c(1)).^2 + (yy - c(2)).^2 <= r^2) = 0;
imshow(occluder);

scene = imresize(imread('cameraman.tif'), [npix, npix]);

amat = zeros(numel(scene));

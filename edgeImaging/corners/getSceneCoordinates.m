function [door1, door2, door_height, ceil_corner, wall_height] =...
    getSceneCoordinates(ceiling_cal, wall_cal)

addpath(genpath('rectify'));
addpath(genpath('../utils/pyr'));

filter = binomialFilter(5);
% rectify to ceiling plane (xy)
% vceil = VideoReader(ceiling_cal);
% ceil_img = read(vceil,100);
ceil_img = blurDnClr(double(imread(ceiling_cal)), 4, filter);

warning('RECTIFY SO THAT CEILING CORNER IS TOP LEFT OF IMAGE');

[iold, jold, ii, jj, ~] = rectify_image_solve(ceil_img);
ceil_img = rectify_image(ceil_img, iold, jold, ii, jj);
% rectify_params = [iold, jold, ii, jj];

% get door corners and ceiling corner
figure; imagesc(ceil_img(:,:,1)); title('choose the ceiling corner');
ceil_corner = ginput(1);
hold on; plot(ceil_corner(1), ceil_corner(2), 'ro');

title('choose the door corner closest to ceiling corner');
door1 = ginput(1);
hold on; plot(door1(1), door1(2), 'bo');
assert(all(door1 > 0), ...
    'did not rectify correctly, door corner should have pos coords');

title('choose the door corner further from the ceiling corner');
door2 = ginput(1);
hold on; plot(door2(1), door2(2), 'yo');
assert(all(door2 > 0), ...
    'did not rectify correctly, door corner should have pos coords');
assert(door1(1) <= door2(1), ...
    'did not rectify image correctly, door1 x-coord should be smaller');

% get the height of the room and door
% rectify to the wall plane (xz)
% vwall = VideoReader(wall_cal);
% wall_img = read(vwall, 100);
wall_img = blurDnClr(double(imread(wall_cal)), 4, filter);
[iold, jold, ii, jj, ~] = rectify_image_solve(wall_img);
wall_img = rectify_image(wall_img, iold, jold, ii, jj);

% get the ceiling corners and door corners again
figure; imagesc(wall_img(:,:,1));
title('choose ceiling corner and floor corner');
ceil_corners = ginput(2);
hold on; plot(ceil_corners(1), ceil_corners(2), 'ro');
wall_height = sqrt((ceil_corners(1,1) - ceil_corners(1,2))^2 + ...
    (ceil_corners(2,1) - ceil_corners(2,2))^2);

title('choose door corners from top left, clockwise');
door_corners = ginput(4);
hold on; plot(door_corners(1), door_corners(2), 'bo');
width1 = sqrt((door_corners(1,1) - door_corners(1,2))^2 + ...
    (door_corners(2,1) - door_corners(2,2))^2);
width2 = sqrt((door1(1) - door2(1))^2 + (door1(2) - door2(2))^2);
door_height = sqrt((door_corners(1,2) - door_corners(1,3))^2 + ...
    (door_corners(2,2) - door_corners(2,3))^2);
door_height = door_height * width2 / width1;
wall_height = wall_height * width2 / width1;
end
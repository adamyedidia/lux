function [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg) 

figure;imagesc(caliImg(:,:,1));
title('Select 4 points on a Square in Counter-clockwise order from Top Left'); 
pts = ginput(4); %counter clockwise from top left

% solve for the homography that takes you from the rectified square to
% points in the image...and the inverse
v = homography_solve([0 0; 0 1; 1 1; 1 0]', pts');
v_inv = homography_solve(pts', [0 0; 0 1; 1 1; 1 0]');

% calculate the points on the rectified square image that correspond to the
% corners of the original image
[h, w, ~] = size(caliImg);
tL = v_inv * [1; 1; 1]; tL = tL./tL(3);
bL = v_inv * [1; h; 1]; bL = bL./bL(3);
tR = v_inv * [w; 1; 1]; tR = tR./tR(3);
bR = v_inv * [w; h; 1]; bR = bR./bR(3);
minX = min([tL(1), bL(1), tR(1), bR(1)]); 
maxX = max([tL(1), bL(1), tR(1), bR(1)]); 
minY = min([tL(2), bL(2), tR(2), bR(2)]); 
maxY = max([tL(2), bL(2), tR(2), bR(2)]); 

% make a grid of points on the rectified square  
[jjj, iii] = ndgrid(linspace(minY,maxY,h), linspace(minX,maxX,w));

% compute the corresponding point in the original image
cord = v * [iii(:)'; jjj(:)'; ones(1,h*w)];
iold = reshape( cord(1,:)./cord(3,:), size(iii)); 
jold = reshape( cord(2,:)./cord(3,:), size(jjj)); 

% interpolate the image to be rectified. 
[jj, ii] = ndgrid(1:h, 1:w); 
[rectified_img] = rectify_image(caliImg, iold, jold, ii, jj);
end



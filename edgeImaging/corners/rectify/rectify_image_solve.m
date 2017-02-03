function [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg, verbose) 

if nargin<2
    verbose = 0;
end

[h, w, ~] = size(caliImg);
sz = min(h,w)/3; 

figure;imagesc(caliImg(:,:,1));
title('Select 4 points on a Square in Counter-clockwise order from Top Left'); 
pts = ginput(4); %counter clockwise from top left
% first row are x locations and second row are y locations

hold on; plot(pts(:,1), pts(:,2), 'r');



% solve for the homography that takes you from the rectified square to
% points in the image...and the inverse
pts_ref = [0 0; 0 1; 1 1; 1 0]; 
v = homography_solve(pts_ref', pts');
v_inv = homography_solve(pts', pts_ref');

% calculate the points on the rectified square image that correspond to the
% corners of the original image
tL = v_inv * [1; 1; 1]; tL = tL./tL(3);
bL = v_inv * [1; h; 1]; bL = bL./bL(3);
tR = v_inv * [w; 1; 1]; tR = tR./tR(3);
bR = v_inv * [w; h; 1]; bR = bR./bR(3);
minX = min([tL(1), bL(1), tR(1), bR(1)]); 
maxX = max([tL(1), bL(1), tR(1), bR(1)]); 
minY = min([tL(2), bL(2), tR(2), bR(2)]); 
maxY = max([tL(2), bL(2), tR(2), bR(2)]); 

hvec = linspace(minY,maxY,round((maxY - minY)*sz)); 
wvec = linspace(minX,maxX,round((maxX - minX)*sz)); 
warning('this is rounded to be a square - alternatively we can set hvec and wvec to be the same'); 

[jjj, iii] = ndgrid(hvec, wvec);

cord = v * [iii(:)'; jjj(:)'; ones(1,numel(iii))];
iold = reshape( cord(1,:)./cord(3,:), size(iii)); 
jold = reshape( cord(2,:)./cord(3,:), size(jjj)); 

[jj, ii] = ndgrid(1:h, 1:w); 
[rectified_img] = rectify_image(caliImg, iold, jold, ii, jj);

if verbose
    figure;imagesc(rectified_img./255, 'XData',wvec,'YData',hvec); axis equal; 
    hold on; plot(pts_ref(:,1), pts_ref(:,2), 'r');
end

end

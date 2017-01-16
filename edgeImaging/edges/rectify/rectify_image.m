function [rectified_img] = rectify_image(img, iold, jold, ii, jj) 

rectified_img = zeros(size(img)); 

for c=1:size(img,3)
    rectified_img(:,:,c) = interp2(ii,jj,double(img(:,:,c)),iold,jold);
end
end
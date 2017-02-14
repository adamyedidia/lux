function [ imOut] = renderPt(params,clr,pos)
%  renderPt renders a point of light of color/intensity clr (a 3-vector) 
%  The light source is assumed to be at single 3d position pos (a 3-vector) 
%   Dec. 8, 2017  billf created.
%   Note:  this currently assumes image is in z=0 plane, and that the x,y 
%   coords are axis-aligned.

% the x, y, z coordinate of each point in the plane of the image to be
% observed.  normal is the 3-vector surface normal
% step * (imageH -1) = xb - xa, similarly for the vertical
xa = params.imageCorner1(1);
xb = params.imageCorner2(1);
xgrid = xa: (xb - xa)/(params.imageH -1) : xb;
ya = params.imageCorner1(2);
yb = params.imageCorner4(2);
ygrid = ya: (yb - ya)/(params.imageV -1) : yb;

[imx,imy] = meshgrid(xgrid, ygrid);
imz = zeros(size(imx));
normal = params.imageNormal;

% find vector from pos to the image plane. Render each point. Then check 
% to see whether it is visible around the corner, and zero it out, if not.
% Make the light vector have a unit normal
toLight = zeros(size(imx,1), size(imx,2), 3);
toLight(:,:,1) = imx;
toLight(:,:,2) = imy;
toLight(:,:,3) = imz;
toLight = repmat(reshape(pos,[1,1,3]), [size(imx,1), size(imx,2), 1]) - toLight;
% now normalize toLight
denom = sum(toLight.^2, 3).^0.5;
toLight = toLight ./ repmat(denom, [1,1,3]);

imOut = renderBRDF(clr, toLight, params.unitVectorToObserver, ...
    params.imageNormal, params.brdf);

% now set to zero any image pixels that are obstructed by the wall.  Check
% the sign of the y-value where the line from the image to the light source
% intersects the x=0 plane.  THIS ALGEBRA ASSUMES z=0.
% c = pos(1); d = pos(2);  a = imx; b = imy;   where image pt is (a,b) and
% light source point is (c,d)

% make a non-zero mask.  then set the denom to 1 when it is zero, but
% remembering the mask
denom = (pos(1) - imx);
zeroMask = denom == 0;
denom = denom + zeroMask;

imtest = (pos(2) - imy) .* abs(imx ./ denom) + imy;
imtest = double(~zeroMask .* imtest + zeroMask);
imOut = imOut .* repmat(double(imtest > 0),[1,1,3]);

end


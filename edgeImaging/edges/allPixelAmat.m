function [amat, x0, y0] = allPixelAmat(corner, maxr, nsamples, thetas)
if nargin < 4
    thetas = [0, pi/2];
end
angles = linspace(thetas(1), thetas(2), nsamples);
% make x and y grid in the right direction, grid goes out from the corner
% corner is in the (1,1) position in grid
if abs(cos(thetas(1))) < 1e-4 % starting from k * pi/2
    xdir = sign(cos(thetas(2)));
    ydir = sign(sin(thetas(1)));
else % starting from k * pi
    xdir = sign(cos(thetas(1)));
    ydir = sign(sin(thetas(2)));
end
xmax = (maxr - 1) * xdir;
ymax = (maxr - 1) * ydir;
[yy, xx] = ndgrid(0:ydir:ymax, 0:xdir:xmax);
x0 = xx + floor(corner(1));
y0 = yy + floor(corner(2));

% find the angle each point makes with the wall and corner
theta = atan2(double(yy), double(xx));
theta(1) = thetas(2);
[nrows, ncols] = size(yy);
theta = reshape(theta, [nrows*ncols, 1]);
amat = zeros([nrows*ncols, nsamples]);
adelta = angles(2) - angles(1);
adir = sign(adelta);
for i = 1:nrows*ncols
    if adir > 0
        idx = sum(angles <= theta(i)); % idx of greatest angle < theta(i)
    else
        idx = sum(angles >= theta(i)); % idx of smallest angle > theta(i)
    end
    amat(i,1:idx) = 1;
    if idx < nsamples
        amat(i,idx+1) = (theta(i) - angles(idx))/adelta;
    end
end
end
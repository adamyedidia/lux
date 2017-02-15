function [amat, x0, y0] = allPixelAmat(corner, maxr, nsamples, thetas)
angles = linspace(thetas(1), thetas(2), nsamples);
% make x and y grid in the right direction, grid goes out from the corner
% corner is in the (1,1) position in grid
if abs(cos(thetas(1))) < 1e-4 % starting from k * pi/2 (k odd)
    xdir = sign(cos(thetas(2)));
    ydir = sign(sin(thetas(1)));
else % starting from k * pi
    xdir = sign(cos(thetas(1)));
    ydir = sign(sin(thetas(2)));
end
xmax = (maxr - 1) * xdir;
ymax = (maxr - 1) * ydir;
[yy, xx] = ndgrid(0:ydir:ymax, 0:xdir:xmax);
x0 = xx + round(corner(1));
y0 = yy + round(corner(2));

% find the angle each point makes with the wall and corner
adelta = angles(2) - angles(1);
adir = sign(adelta); % dir of angles away from wall

theta = atan2(double(yy), double(xx));
theta(1) = thetas(2);
% theta = (theta - thetas(1)) * adir;
[nrows, ncols] = size(yy);
theta = reshape(theta, [nrows*ncols, 1]);
amat = zeros([nrows*ncols, nsamples]);

for i = 1:nrows*ncols
    idx = sum((theta(i) - angles)*adir >= 0); % idx of largest <= theta(i)    
    d = (theta(i) - angles(idx))/adelta;
    amat(i,1:idx-1) = 1;
    amat(i,idx) = 0.5*(2-d)*d + 0.5;
    if idx < nsamples
        amat(i,idx+1) = 0.5*d^2;
    end
end
amat = amat / (40/nsamples);
end
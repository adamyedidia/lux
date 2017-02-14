function amat = cornerAmat(floorx, floory, scenex, sceney, side)
% the floorx and floory define the corner location and the region
% extending beyond the corner. the side defines which side, x or y,
% the wall is on. if nothing is passed in, it assumes 1, the wall is
% on the x side (same x coords, opposite dir in y). if -1, the wall is
% on the y side (same y coords, opposite dir in x).

if nargin < 5
    side = 1;
end
% cornerloc and pointloc are [x, y, z] coords
% floorx(1), floory(1) is the corner, shift all coords so corner is (0, 0)
[obsy, obsx] = ndgrid(floory - floory(1), floorx - floorx(1));
xdir = sign(floorx(2) - floorx(1));
ydir = sign(floory(2) - floory(1));
[imy, imx] = ndgrid(sceney - floory(1), scenex - floorx(1));

amat = zeros([numel(obsx), numel(imx)]);
for i = 1:numel(imx)
    if side == 1 && sign(imx(i)) == xdir
        continue
    else if side == -1 && sign(imy(i)) == ydir
            continue
        end
    end
    if (side == -1 && sign(imx(i)) == xdir)...
            || (side == 1 && sign(imy(i)) == ydir);
        amat(:,i) = 1;
        continue;
    end
    % calculate the line from image cut by the corner
    ytest = obsy - imy(i) .* (obsx ./ imx(i));
    amat(:,i) = reshape(ytest * side >= 0, [size(amat,1),1]);
%     if mod(i,length(scenex)) == 0
%         imagesc(floorx, floory, ytest * side >= 0); title(imx(i)); pause(1e-4);
%     end
end
end
function amat = cornerAmat(floorx, floory, scenex, sceney)
% cornerloc and pointloc are [x, y, z] coords
% floorx(1), floory(1) is the corner, shift all coords so corner is (0, 0)
[obsy, obsx] = ndgrid(floory - floory(1), floorx - floorx(1));
[imy, imx] = ndgrid(sceney - floory(1), scenex - floorx(1));

amat = zeros([numel(obsx), numel(imx)]);
for i = 1:numel(imx)
    % calculate the line from image cut by the corner
    ytest = obsy - imy(i) .* (obsx ./ imx(i));
    amat(:,i) = reshape(ytest >= 0, [size(amat,1),1]);
%     imagesc(floorx, floory, ytest >= 0); title(imx(i)); pause(5e-4);
end
end
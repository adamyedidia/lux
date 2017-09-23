function [outframes, angles, params] = doCornerRecon(params, moviefile, calimgfile)

addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));

load(params.corner_datafile, 'avg_img');

% get A matrix
[amat_out, angles, params] = getAmat(params);
amat = amat_out{1};
if ~strcmp(params.amat_method, 'interp')   
    params.crop_idx = amat_out{2};
else
    params.lambda = params.lambda / 4;
end

% throw away nan rows
[~, nanrows] = getObsVec(avg_img, params);
amat(nanrows,:) = [];

% see where we're looking on the floor
figure; imagesc(avg_img(:,:,1)); hold on;
plot(params.corner(1), params.corner(2), 'ro');
if strcmp(params.amat_method, 'interp')
    x0 = cos(angles) * params.rs + params.corner(1);
    y0 = sin(angles) * params.rs + params.corner(2);
    plot(x0, y0, 'y');
    plot(x0(1,:), y0(1,:), 'g');
    plot(x0(end,:), y0(end,:), 'g');
else
    % plot a box around the area we're looking at
    [y0, x0] = ind2sub(params.framesize, params.crop_idx);
    plot([x0(1,1), x0(1,end), x0(1,end), x0(1,1), x0(1,1)],...
         [y0(1,1), y0(1,1), y0(end,1), y0(end,1), y0(1,1)], 'y');
end

if nargin > 2
    saveas(gcf, calimgfile);
end

% spatial prior
bmat = eye(size(amat,2)) - diag(ones([size(amat,2)-1,1]), 1);
bmat = bmat(1:end-1,:);
bmat(1,:) = 0; % don't use the constant light to smooth

switch params.inf_method
    case 'spatial_smoothing'
        outframes = spatialSmoothingRecon(moviefile, params, amat, bmat);
    case 'online'
        outframes = onlineRecon(moviefile, params, amat, bmat);
    case 'kalman_filter'
        fmat = eye(size(amat,2));
        outframes = kalmanFilterRecon(moviefile, params, amat, bmat, fmat);
    case 'kalman_smoothing'
        fmat = eye(size(amat,2));
        outframes = kalmanSmoothingRecon(moviefile, params, amat, bmat, fmat);
    case 'kalman_smoothing_priors'
        fmat = eye(size(amat,2));
        outframes = kalmanSmoothingPriorsRecon(moviefile, params, amat, bmat, fmat);
    otherwise % default to naive corner cam
        outframes = noSmoothingRecon(moviefile, params, amat);
end

if size(angles, 2) ~= 1 % not a column vector
    angles = angles';
end
angles = smoothSamples(angles, params.smooth_up);
end

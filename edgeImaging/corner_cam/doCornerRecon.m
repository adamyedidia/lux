function outframes = doCornerRecon(params, moviefile, outfile)

addpath(genpath('../utils/pyr'));
addpath(genpath('../rectify'));


% get A matrix
[amat_out, params] = getAmat(params);
amat = amat_out{1};
if ~strcmp(params.amat_method, 'interp')   
    params.crop_idx = amat_out{2};
end

% find and remove rows corresponding to nan obs pixels
load(params.corner_datafile, 'avg_img');
[~, nanrows] = getObsVec(avg_img, params);
amat(nanrows,:) = [];

% spatial prior
bmat = eye(size(amat,2)) - diag(ones([size(amat,2)-1,1]), 1);
bmat = bmat(1:end-1,:);
bmat(1,:) = 0; % don't use the constant light to smooth

switch params.inf_method
    case 'spatial_smoothing'
        outframes = spatialSmoothingRecon(moviefile, params, amat, bmat);
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

outframes = outframes(:,2:end,:); % throwing away the constant light

if nargin > 2
    save(outfile);
end
end
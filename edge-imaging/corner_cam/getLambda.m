function lambda = getLambda(params)
% modify these params rn for getting the part of the variance
% image we're interested in -- this won't change the actual params!
if strcmp(params.amat_method, 'interp')
    params.amat_method = 'allpix';
    params.outr = max(params.rs);
end

amat_out = getAmat(params);
params.crop_idx = amat_out{2};

% load in the snr image and rectify it appropriately
load(params.mean_datafile, 'variance');
load(params.cal_datafile, 'iold', 'jold', 'ii', 'jj');

var_rect = rectify_image(variance, iold, jold, ii, jj);
var_rect = blurDnClr(var_rect, params.downlevs, params.filt);
var_cropped = getObsVec(var_rect, params);
% variance reduced by number of frames we average over
lambda = median(median(mean(var_cropped, 3))) / params.navg;
end

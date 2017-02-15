function outframes = kalmanFilterRecon(moviefile, caldata, params, amat, bmat, fmat)
% load from saved calibration
load(caldata);
v = VideoReader(moviefile);

frameidx = params.start:params.step:params.endframe;
nrows = params.framesize(1);
ncols = params.framesize(2);
nchans = params.framesize(3);
mean_img = repmat(mean_pixel, [nrows, ncols]);

if params.sub_mean
    back_img = avg_img;
else if ~params.sub_background
    back_img = zeros(size(background));
    mean_img = zeros(size(mean_img));
    end
end

nout = length(frameidx);
nsamples = params.nsamples;

% initialize filter variables
cur_mean = zeros([nsamples, nchans]);
pred_mean = zeros([nsamples, nchans]);
cur_cov = zeros([nsamples, nsamples, nchans]);
prior_cov = inv(bmat' * bmat / params.sigma^2);
pred_cov = repmat(prior_cov, [1, 1, 3]);

rmat = params.lambda * eye(size(amat,1)); % independent pixel noise
qmat = params.alpha * eye(nsamples); % independent process noise

outframes = zeros([nout, (params.nsamples-1)*params.smooth_up+1, nchans]);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Iteration %i\n', n);
    framen = rectify_image(double(read(v, n)), iold, jold, ii, jj);
    framen = blurDnClr(framen - back_img, params.downlevs, params.filt) - mean_img;

    y = getObsVec(framen, params);

    % kalman forward filtering
    for c = 1:nchans
        % update step
        gain = pred_cov(:,:,c) * amat' * inv(amat*pred_cov(:,:,c)*amat' + rmat);
        cur_mean(:,c) = pred_mean(:,c) + gain * (y(:,c) - amat * pred_mean(:,c));
        cur_cov(:,:,c) = pred_cov(:,:,c) - gain * amat * pred_cov(:,:,c);
        
        % next predict step
        pred_mean(:,c) = fmat * cur_mean(:,c);
        pred_cov(:,:,c) = fmat * cur_cov(:,:,c) * fmat' + qmat;
    end
    toc;

    outframes(i,:,:) = smoothSamples(cur_mean, params.smooth_up);
end
outframes(outframes<params.minclip) = params.minclip;
outframes(outframes>params.maxclip) = params.maxclip;
outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end
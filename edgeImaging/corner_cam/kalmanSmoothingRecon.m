function outframes = kalmanSmoothingRecon(moviefile, caldata, params, amat, bmat, fmat)
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
cur_mean = zeros([nsamples, nchans, nout]);
pred_mean = zeros([nsamples, nchans, nout+1]);
cur_cov = zeros([nsamples, nsamples, nchans, nout]);
pred_cov = zeros([nsamples, nsamples, nchans, nout+1]);
prior_cov = inv(bmat' * bmat / params.sigma^2);
pred_cov(:,:,:,1) = repmat(prior_cov, [1, 1, 3]); % pred 1|0

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
        gain = pred_cov(:,:,c,i) * amat' * inv(amat*pred_cov(:,:,c,i)*amat' + rmat);
        cur_mean(:,c,i) = pred_mean(:,c,i) + gain * (y(:,c) - amat * pred_mean(:,c,i));
        cur_cov(:,:,c,i) = pred_cov(:,:,c,i) - gain * amat * pred_cov(:,:,c,i);
        
        % next predict step, i+1|i
        pred_mean(:,c,i+1) = fmat * cur_mean(:,c,i);
        pred_cov(:,:,c,i+1) = fmat * cur_cov(:,:,c,i) * fmat' + qmat;
    end
    toc;
end

% backward smoothing
out_mean = cur_mean;
for c = 1:nchans
    out_mean(:,c,nout+1) = pred_mean(:,c,nout+1);
    for i = nout-1:-1:1
        res = out_mean(:,c,i+1) - pred_mean(:,c,i+1);
        fi_res = cur_cov(:,:,c,i)*fmat'*(pred_cov(:,:,c,i+1)\res);
        out_mean(:,c,i) = cur_mean(:,c,i) + fi_res;
    end
end

for i = 1:nout
    outframes(i,:,:) = smoothSamples(out_mean(:,:,i), params.smooth_up);
end

outframes(outframes<params.minclip) = params.minclip;
outframes(outframes>params.maxclip) = params.maxclip;
outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end
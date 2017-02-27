function outframes = kalmanSmoothingPriorsRecon(moviefile, params, amat, bmat, fmat)
% reconstruct with kalman smoothing, using spatial regularization matrix
% bmat at every timestep, and transition matrix fmat
v = VideoReader(moviefile);

% load from saved datafiles
load(params.cal_datafile, 'iold', 'jold', 'ii', 'jj');
load(params.mean_datafile, 'mean_pixel');
load(params.corner_datafile, 'avg_img');

frameidx = params.start:params.step:params.endframe;
[nrows, ncols, nchans] = size(avg_img);
mean_img = repmat(mean_pixel, [nrows, ncols]);

if params.sub_mean
    back_img = avg_img;
else 
    back_img = zeros(size(avg_img));
    mean_img = zeros(size(mean_img));
end

nout = length(frameidx);

% initialize filter variables

% y(t) = Ax(t) + w(t); w(t) ~ N(0, rmat)
% x(t) = Fx(t-1) + v(t); v(t) ~ N(0, qmat), x(0) ~ N(0, prior_cov)

xdim = size(amat,2);
pred_mean = zeros([xdim, nchans, nout+1]);
pred_cov = zeros([xdim, xdim, nchans, nout+1]);
% prior mean is 0
prior_cov = inv((bmat'*bmat + eye(xdim)*params.eps) / params.sigma^2);
pred_cov(:,:,:,1) = repmat(prior_cov, [1, 1, 3]); % pred 1|0

all_obs = zeros([size(amat,1), nchans, nout]);

rmat = params.lambda * eye(size(amat,1)); % independent pixel noise
qmat = params.alpha * eye(size(amat,2)); % independent process noise

outframes = zeros([nout, (size(amat,2)-1)*params.smooth_up+1, nchans]);
tic;

for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Forward frame %i\n', n);
    framen = rectify_image(double(read(v, n)), iold, jold, ii, jj);
    framen = (blurDnClr(framen, params.downlevs, params.filt) - back_img) + mean_img;

    y = getObsVec(framen, params);
    all_obs(:,:,i) = y;

    for c = 1:nchans
        % update step
        gain = pred_cov(:,:,c,i) * amat' * inv(amat*pred_cov(:,:,c,i)*amat' + rmat);
        cur_mean = pred_mean(:,c,i) + gain * (y(:,c) - amat * pred_mean(:,c,i));
        cur_cov = pred_cov(:,:,c,i) - gain * amat * pred_cov(:,:,c,i);
        
        % next predict step, i+1|i, with the prior
        cur_mean = fmat * cur_mean;
        cur_cov = fmat * cur_cov * fmat' + qmat;
        
        prior_factor = prior_cov * inv(prior_cov + cur_cov);
        pred_mean(:,c,i+1) = prior_factor * cur_mean;
        pred_cov(:,:,c,i+1) = prior_factor * cur_cov;
    end
    toc;
end

% backward update initializations
back_mean = zeros(size(pred_mean));
back_cov = zeros(size(pred_cov));

% do the first update now -- 
% we want the i|i updates in the end, not the i|i+1predictions
y = all_obs(:,:,nout);
for c = 1:nchans
    gain = prior_cov * amat' * inv(amat*prior_cov*amat' + rmat);
    back_mean(:,c,nout) = gain * y(:,c);
    back_cov(:,:,c,nout) = prior_cov - gain * amat * prior_cov;
end

% predictions and updates from nout-1 to 1
for i = nout-1:-1:1
    y = all_obs(:,:,i);
    for c = 1:nchans
        % prediction, back update i|i+1
        gain = prior_cov*fmat'*inv(qmat + back_cov(:,:,c,i+1) + fmat*prior_cov*fmat');
        temp_mean = gain * back_mean(:,c,i+1);
        temp_cov = prior_cov - gain * fmat * prior_cov;
        
        % update i|i
        gain = temp_cov * amat' * inv(amat*temp_cov*amat' + rmat);
        back_mean(:,c,i) = temp_mean + gain * (y(:,c) - amat*temp_mean);
        back_cov(:,:,c,i) = temp_cov - gain * amat * temp_cov;
    end
end

% combining into marginals
out_mean = zeros(size(pred_mean));
for i = 1:nout
    for c = 1:nchans
        % pred_cov i|i-1, back_cov i|i
        const = inv(back_cov(:,:,c,i) + pred_cov(:,:,c,i));
        out_mean(:,c, i) = back_cov(:,:,c,i) * const * pred_mean(:,c,i)...
            + pred_cov(:,:,c,i) * const * back_mean(:,c,i);
    end
end

for i = 1:nout
    outframes(i,:,:) = smoothSamples(out_mean(:,:,i), params.smooth_up);
end

% outframes(outframes<params.minclip) = params.minclip;
% outframes(outframes>params.maxclip) = params.maxclip;
% outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end
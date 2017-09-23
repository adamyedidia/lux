function outframes = kalmanFilterRecon(moviefile, params, amat, bmat, fmat)
% reconstruct with kalman filter, using spatial regularization matrix
% bmat, and transition matrix fmat
if params.isvid > 0
    vsrc = VideoReader(moviefile);
else
    vsrc = moviefile;
end

% load from saved datafiles
load(params.cal_datafile, 'iold', 'jold', 'ii', 'jj');
if params.use_median
    load(params.median_datafile, 'med_img');
    mean_pixel = mean(mean(med_img));
    if params.sub_mean
        back_img = med_img;
    else
        back_img = zeros(size(med_img));
        mean_pixel = zeros(size(mean_pixel));
    end
else
    load(params.mean_datafile, 'mean_pixel', 'avg_img');
    if params.sub_mean
        back_img = avg_img;
    else 
        back_img = zeros(size(avg_img));
        mean_pixel = zeros(size(mean_pixel));
    end
end

frameidx = params.start:params.step:params.endframe;
nchans = params.framesize(3);
nout = length(frameidx);

% initialize filter variables
% y(t) = Ax(t) + w(t); w(t) ~ N(0, rmat)
% x(t) = Fx(t-1) + v(t); v(t) ~ N(0, qmat), x(0) ~ N(0, prior_cov)

xdim = size(amat,2);
cur_mean = zeros([xdim, nchans]);
pred_mean = zeros([xdim, nchans]);
cur_cov = zeros([xdim, xdim, nchans]);
prior_cov = inv((bmat'*bmat + eye(xdim)*params.eps) / params.sigma^2);
pred_cov = repmat(prior_cov, [1, 1, 3]);

rmat = params.lambda * eye(size(amat,1)); % independent pixel noise
qmat = params.alpha * eye(xdim); % independent process noise

outframes = zeros([nout, (size(amat,2)-2)*params.smooth_up+1, nchans]);
orig_size = size(back_img);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Frame %i\n', n);
    framen = rectify_image(getFrame(params.isvid, vsrc, n,...
        params.navg, orig_size) - back_img, iold, jold, ii, jj);
    framen = bsxfun(@plus, blurDnClr(framen, params.downlevs, params.filt), mean_pixel);

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

    outframes(i,:,:) = smoothSamples(cur_mean(2:end,:), params.smooth_up);
end
% outframes(outframes<params.minclip) = params.minclip;
% outframes(outframes>params.maxclip) = params.maxclip;
% outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end

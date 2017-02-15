function outframes = spatialSmoothingRecon(moviefile, caldata, params, amat, bmat)
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
outframes = zeros([nout, (params.nsamples-1)*params.smooth_up+1, nchans]);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Iteration %i\n', n);
    framen = rectify_image(double(read(v, n)), iold, jold, ii, jj);
    framen = blurDnClr(framen - back_img, params.downlevs, params.filt) - mean_img;

    y = getObsVec(framen, params);

    % using a spatial prior
    out = zeros([params.nsamples, nchans]);
    for c = 1:nchans
        out(:,c) = (amat'*amat/params.lambda + bmat'*bmat/params.sigma^2)...
            \(amat'*y(:,c)/params.lambda);
    end
    toc;
    outframes(i,:,:) = smoothSamples(out, params.smooth_up);
end
outframes(outframes<params.minclip) = params.minclip;
outframes(outframes>params.maxclip) = params.maxclip;
outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end
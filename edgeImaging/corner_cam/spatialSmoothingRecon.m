function outframes = spatialSmoothingRecon(moviefile, params, amat, bmat)
% reconstruct using a spatial smoothness regularizer given by bmat
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

cmat = eye(size(bmat, 2));
cmat(1,:) = 0;

nout = length(frameidx);
outframes = zeros([nout, (size(amat,2)-1)*params.smooth_up+1, nchans]);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Frame %i\n', n);
    framen = rectify_image(double(read(v, n)), iold, jold, ii, jj);
    framen = (blurDnClr(framen, params.downlevs, params.filt) - back_img) + mean_img;

    y = getObsVec(framen, params);

    % using a spatial prior
    out = zeros([size(amat,2), nchans]);
    for c = 1:nchans
        out(:,c) = (amat'*amat/params.lambda + (bmat'*bmat + cmat)/params.sigma^2)...
            \(amat'*y(:,c)/params.lambda);
    end
    toc;
    outframes(i,:,:) = smoothSamples(out, params.smooth_up);
end
outframes(outframes<params.minclip) = params.minclip;
outframes(outframes>params.maxclip) = params.maxclip;
outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end
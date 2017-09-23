function outframes = onlineRecon(moviefile, params, amat, bmat)
% reconstruct using a spatial smoothness regularizer given by bmat
if params.isvid > 0
    vsrc = VideoReader(moviefile);
else
    vsrc = moviefile;
end

% load from saved datafiles
load(params.cal_datafile, 'iold', 'jold', 'ii', 'jj');

% when online, we subtract from first frame
orig_size = [size(ii), 3];
back_img = getFrame(params.isvid, vsrc,...
    params.start, params.navg, orig_size);
mean_pixel = mean(mean(back_img, 1), 2);

% start processing from the second desired frame
frameidx = params.start+1:params.step:params.endframe;
nchans = params.framesize(3);

cmat = eye(size(bmat, 2))*params.eps;
cmat(1,:) = 0;

nout = length(frameidx);
outframes = zeros([nout, (size(amat,2)-2)*params.smooth_up+1, nchans]);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Frame %i\n', n);
    frame = getFrame(params.isvid, vsrc, n, params.navg, orig_size);
    framen = rectify_image(frame - back_img, iold, jold, ii, jj);
    framen = bsxfun(@plus, blurDnClr(framen, params.downlevs, params.filt), mean_pixel);

    y = getObsVec(framen, params);

    % using a spatial prior
    out = zeros([size(amat,2), nchans]);
    for c = 1:nchans
        out(:,c) = (amat'*amat/params.lambda + (bmat'*bmat + cmat)/params.sigma^2)...
            \(amat'*y(:,c)/params.lambda);
    end
    toc;
    outframes(i,:,:) = smoothSamples(out(2:end,:), params.smooth_up);

    if params.sub_mean
        % update our running mean
        back_img = back_img * i/(i+1) + frame/(i+1);
        mean_pixel = mean(mean(back_img, 1), 2);
    end % otherwise just use the first frame
end
% outframes(outframes<params.minclip) = params.minclip;
% outframes(outframes>params.maxclip) = params.maxclip;
% outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end

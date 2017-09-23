function outframes = noSmoothingRecon(moviefile, params, amat)
% reconstruct the corner naively with least squares
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

outframes = zeros([nout, (size(amat, 2)-2)*params.smooth_up+1, nchans]);
orig_size = size(back_img);
tic;
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Frame %i\n', n);
    framen = rectify_image(getFrame(params.isvid, vsrc, n,...
        params.navg, orig_size) - back_img, iold, jold, ii, jj);
    framen = bsxfun(@plus, blurDnClr(framen, params.downlevs, params.filt), mean_pixel);

    y = getObsVec(framen, params);

    out = zeros([size(amat, 2), nchans]);
    for c = 1:nchans
        out(:,c) = (amat'*amat/params.lambda)\(amat'*y(:,c)/params.lambda);
    end
    toc;
    outframes(i,:,:) = smoothSamples(out(2:end,:), params.smooth_up);
end
% outframes(outframes<params.minclip) = params.minclip;
% outframes(outframes>params.maxclip) = params.maxclip;
% outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end

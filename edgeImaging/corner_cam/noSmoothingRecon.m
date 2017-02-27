function outframes = noSmoothingRecon(moviefile, params, amat)
% reconstruct the corner naively with least squares
addpath(genpath('../rectify'));

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
outframes = zeros([nout, (size(amat, 2)-1)*params.smooth_up+1, nchans]);
tic;
try
for i = 1:nout
    n = frameidx(i); % using the nth frame
    fprintf('Frame %i\n', n);
    framen = rectify_image(double(v.read(n)), iold, jold, ii, jj);
    framen = blurDnClr(framen - back_img, params.downlevs, params.filt) - mean_img;

    y = getObsVec(framen, params);

    out = zeros([size(amat, 2), nchans]);
    for c = 1:nchans
        out(:,c) = (amat'*amat/params.lambda)\(amat'*y(:,c)/params.lambda);
    end
    toc;
    outframes(i,:,:) = smoothSamples(out, params.smooth_up);
end
catch
end
% outframes(outframes<params.minclip) = params.minclip;
% outframes(outframes>params.maxclip) = params.maxclip;
% outframes = (outframes - params.minclip)/(params.maxclip - params.minclip);
end

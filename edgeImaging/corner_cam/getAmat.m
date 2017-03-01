function [output, params] = getAmat(params)
if strcmp(params.amat_method, 'interp')
    nsamples = params.nsamples;
    ncircles = length(params.rs);
    
    ydim = nsamples;
    xdim = nsamples+1;
    amat = zeros([ydim*ncircles, xdim]);
    for i = 1:ncircles
        si = ydim*(i-1) + 1;
        ei = ydim*i;
        amat(si:ei,2:end) = tril(ones(nsamples));
    end
    amat(:,1) = 1;
    output{1} = amat;
else % default use all pixels
    [amat, x0, y0, maxr] = allPixelAmat(params.corner, params.framesize,...
        params.outr,params.nsamples, params.theta_lim);
    params.outr = maxr;
    crop_idx = sub2ind(params.framesize, y0, x0);
    output{1} = amat;
    output{2} = crop_idx;
end
end
function output = getAmat(params)
if strcmp(params.amat_method, 'interp')
    nsamples = params.nsamples;
    ncircles = length(params.rs);
    
    amat = zeros([nsamples*ncircles, nsamples+1]);
%     amat = zeros([nsamples*ncircles, nsamples]);
    for i = 1:ncircles
        si = nsamples*(i-1) + 1;
        ei = nsamples*i;
        amat(si:ei,2:end) = tril(ones(nsamples));
    end
    amat(:,1) = 1;
    output{1} = amat;
else % default use all pixels
    [amat, x0, y0] = allPixelAmat(params.corner,...
        params.outr,params.nsamples, params.theta_lim);
    crop_idx = sub2ind(params.framesize, y0, x0);
    output{1} = amat;
    output{2} = crop_idx;
end
end
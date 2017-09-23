function outframe = smoothSamples(x, upfactor)
[nsamples, nchans] = size(x);
outframe = zeros([(nsamples-1)*upfactor+1, nchans]);
left_frac = linspace(1, 0, upfactor+1)';
left_frac = repmat(left_frac(1:end-1), [nsamples-1, 1]);
right_frac = linspace(0, 1, upfactor+1)';
right_frac = repmat(right_frac(1:end-1), [nsamples-1, 1]);
for c = 1:nchans
    left = repmat(x(1:end-1,c)', [upfactor,1]);
    left = left(:);
    right = repmat(x(2:end,c)', [upfactor,1]);
    right = right(:);
    outframe(1:end-1,c) = left_frac .* left + right_frac .* right;
    outframe(end,c) = x(end,c);
end
end
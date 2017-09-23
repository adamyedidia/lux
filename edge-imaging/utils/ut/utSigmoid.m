
function res = utSigmoid(data,center,sharp);
res = (1 + exp(-(data-center)/sharp)).^(-1);


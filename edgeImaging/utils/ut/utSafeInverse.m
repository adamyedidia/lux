
function invm = utSafeInverse(m,wantEigenFlag)

[v,d] = eig(m);
v = real(v);
d = diag(max(real(diag(d)),eps));
inv = v * diag(diag(d).^(-1)) * v';

if (~exist('wantEigenFlag'))  
     invm=inv;
else invm{1}=inv; invm{2}=v; invm{3}=prod(diag(d));
end;

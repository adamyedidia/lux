
function v = utNormalize(u)

% utNormalize(u) : quick vector normalization utility
% 
% This returns (u/norm(u)) unless norm(u) is 0, in which
% case it returns u.  This is a simple convenience function.

nu = norm(u);
if (nu == 0) nu=1; end;

v = u / nu; 

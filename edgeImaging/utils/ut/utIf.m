
function val = utIf(bool, val1, val2)

% utIf: functional "if" statement
%
%  X = utIf(Y, A,B) returns A if Y is true (nonzero), and B otherwise.
%  This is basically the much-loved "?:" operator that you find in C++.
%

if (bool) val = val1;
     else val = val2; end;


%
% bug #87382
%

function prod = utSafeMult(f1,f2)

inner = size(f1,2);
%%leap  = 50000;
leap  = 5000;

if (inner <= leap) prod = f1 * f2;
else

   s=1; prod = zeros(size(f1,1),size(f2,2));

   while(inner > leap) 
      prod = prod + f1(:,s:(s+leap-1)) * f2(s:(s+leap-1),:);
      s = s + leap;
      inner = inner - leap;
   end;
   prod = prod + f1(:,s:size(f1,2)) * f2(s:size(f2,1),:);

end;





function res = utIsScalarInteger (value)

%  function res = utIsScalarInteger (value)

res = 0;

if (isnumeric(value))
   if (max(size(value))==1)
      if (round(value)==value)
	 res = 1;
      end;
   end;
end;

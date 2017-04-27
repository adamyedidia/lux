
function grPrint(filename)
global figlist;

if (figlist.printFlag) 
  orient tall; 
  eval(sprintf('print %s%s -dpsc',figlist.printTag,filename'));
end;




function grSetPrinting(varargin)
global figlist;

[printFlag, printTag] = utParseArgs(varargin, {
  { 'printFlag', 1 }
  { 'printTag', '' }
});

figlist.printFlag = printFlag;
figlist.printTag  = printTag;


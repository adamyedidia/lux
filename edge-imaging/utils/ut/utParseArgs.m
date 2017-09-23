
% function [varargout] = utParseArgs(argsIn, argsWanted, lenientFlag)
%
% This function may help make the ubiquitous varargin-parsing code 
% more comprehensible, more consistent, and easier to type.
%
% Pass 'varargin' in as the first parameter, and for the second
% parameter pass a cell array of (1x2) cell arrays.  
% Each (1x2) cell array specifies a parameter:
%   the first entry is a string, the parameter's name, 
%   the second entry is the default value if no matching arg is found.
%
% The arglist may contain positional arguments, (ie, { arg1, arg2, arg3 })
%   folled by key-value pairs (ie, { 'key1', value1, 'key2', value2 }).
% It may also contain one or more arguments with the key 'args', whose value
%   will be recursively interpreted as an argument cell-array.
%
% For example, the code for "showIm" begins as follows:
%
% - - - - - - - - - - - - - - - - - - - - - - 
% function imRet = showIm(im, varargin);
%
% [range, filter, text_flag] = utParseArgs(varargin,{
%    { 'range',     [] },
%    { 'filter',    [] },
%    { 'text_flag', 1  }
% });
% - - - - - - - - - - - - - - - - - - - - - - 
%
% After the utParseArgs call, the variables <range>, <filter>, 
% and <text_flag> are all defined, set to either the default
% values, or values provided by the caller.
%
% The caller can provide the arguments positionally, e.g,
%
%   showIm(im, 1,2,3)
%
% would assign 1, 2, and 3 to <range>, <filter>, and <text_flag>
% respectively, or use the 'name, value' convention, e.g,
%
%   showIm(im, 'filter',17,'range',23)
%
% to assign 23, 17, and 1 (the default) to <range>, <filter>, 
% and <text_flag> respectively.
%
%


function [varargout] = utParseArgs(argsIn, argsWanted, lenientFlag)

lenient = exist('lenientFlag');  % If lenient, will not throw error on
                                 %   unsupported 'key'-'value' pairs.
 

keys = cell(1,length(argsWanted));
for i = 1:length(argsWanted)
  keys{i} = argsWanted{i}{1};
end;

[found, values] = utScanArglist(argsIn, keys, lenient);

varargout = cell(1,length(argsWanted));
for i = 1:length(argsWanted) 
  varargout{i} = utIf(found(i), values{i}, argsWanted{i}{2});
end;

% ----------------------------------------------------------------

function [found, values] = utScanArglist(argsIn, keys, lenient)

found  = zeros(1,length(keys));
values = cell(1,length(keys));
 
nKeys = length(keys);
nIn   = length(argsIn);
positionalsDone = 0;

i = 1;
while (i <= nIn)
   arg = argsIn{i};

   keyValueFound = 0;
   if (isstr(arg) & (i < length(argsIn)))  
      for j = 1:nKeys
         if (strcmp(arg,keys{j})) 
            i = i + 1;
            values{j}=argsIn{i}; found(j)=1;
            keyValueFound   = 1;
            break;
         end;
      end;
      if (~keyValueFound) & (strcmp(arg,'args'))
         i = i + 1;
         [nFound, nValues] = utScanArglist(argsIn{i},keys, lenient);
         for j = 1:length(nFound) if nFound(j)
            values{j} = nValues{j};  found(j)=1;
         end; end;
         keyValueFound = 1;
      end;
      if (keyValueFound) positionalsDone = 1; end;
   end;
   if (~positionalsDone) values{i}=arg; found(i)=1; end;

   if (positionalsDone & ~keyValueFound & ~lenient) 
      disp(sprintf('Error in variable argument list at [%s]\n',arg));
      disp('Allowed arguments are: ');
      for j = 1:nKeys
         disp(sprintf(' %3d: ''%s''',j,keys{j}));
      end;
      disp('-----------------------');
      error('Bad Option');
   end;

   i = i + 1;
   if (i > length(values)) positionalsDone = 1; end;
end;


function [filename, extension, directory] = utFilename(fName)

% utFilename: Parsing a pathname
%
%   [filename, extension, directory] = utFilename(fName)
%
% removes the directory and extension components from a pathname.
% For example,
%
%   utFilename('../blorf/foobar.mat') => ['foobar', 'mat', '../blorf/']
%   utFilename('files.tar.gz')        => ['files', 'tar.gz', './']
%

filename  = '';
extension = '';
directory = '';

i = length(fName);
while ((i>0) & (fName(i)~='/') & (fName(i)~='\')) i=i-1; end;

if (i>0) filename=fName(i+1:length(fName)); directory=fName(1:i);          
    else filename=fName;                    directory='./';
end;

i = 1;
while ((i<=length(filename)) & (filename(i)~='.')) i=i+1; end;

if (i>0) extension=filename(i+1:length(filename)); filename=filename(1:i-1); end;

if (length(filename)==0) filename=''; end;
if (length(extension)==0) extension=''; end;
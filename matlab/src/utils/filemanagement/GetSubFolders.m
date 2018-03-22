function SubFolders = GetSubFolders(DirPath)
%GetSubFolders  returns all the sub directories of a directory.
%
% inputs
%   DirPath  the path to the directory.
%
% outputs
%   SubFolders  the name of the sub directories in a cell list.
%

folder = dir(DirPath);
isub = [folder(:).isdir];
SubFolders = {folder(isub).name}';
SubFolders(ismember(SubFolders,{'.', '..'})) = [];

end

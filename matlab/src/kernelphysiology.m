AbsolutePath = mfilename('fullpath');

[dirpath, ~, ~] = fileparts(AbsolutePath);

addpath(genpath(dirpath));
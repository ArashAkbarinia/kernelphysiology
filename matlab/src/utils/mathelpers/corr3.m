function corr3coef = corr3(mat1, mat2)
%CORR3  Computes the correlation along the 3rd dimension.
%
% inputs
%   mat1  the input matrix 1
%   mat2  the input matrix 2
%
% outputs
%   corr3coef  3-D correlation coefficient
%

avg1 = mean(mat1, 3);
avg2 = mean(mat2, 3);

diff1 = mat1 - avg1;
diff2 = mat2 - avg2;

corr3coef = sum(diff1 .* diff2, 3) ./ sqrt(sum(diff1 .^ 2, 3) .* sum(diff2 .^ 2, 3));

end

function [ colorDiff, diff ] = imgDifference( img1, img2 )

diff_tmp = img1 - img2;
diff_tmp = diff_tmp.^2;
diff = zeros(size(diff_tmp, 1), size(diff_tmp, 2));

for i = 1: size(diff_tmp, 1)
    for j = 1: size(diff_tmp, 2)
        diff (i, j) = sum(diff_tmp(i, j, :));
    end
end

diff = floor(diff / max(diff(:)) * 255);

%colormap jet
%cmap = colormap;
colorDiff = ind2rgb(diff, jet);

%imshow(visualDiff)

end


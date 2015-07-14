function [ colorDiff, err ] = imgDifference( img1, img2 )

diff_tmp = img1 - img2;
diff_tmp = diff_tmp.^2;
diff = zeros(size(diff_tmp, 1), size(diff_tmp, 2));

for i = 1: size(diff_tmp, 1)
    for j = 1: size(diff_tmp, 2)
        diff (i, j) = sum(diff_tmp(i, j, :));
    end
end

diff = sqrt(diff);
err = sum(diff(:)) / (size(diff_tmp, 1) * size(diff_tmp, 2));
diff = floor(diff);

colorDiff = ind2rgb(diff, jet);

imshow(colorDiff)

end


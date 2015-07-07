dir = '../output/OptFlow_HS/';
test_set = 'rubberWheel';

filenameB = ['warpto1_' test_set];

a = imread(['../input/' test_set '_02.png']);
b = imread([dir filenameB '.bmp']);

[colorDiff, diff] = imgDifference(a, b);
avgDiff = sum(diff(:)) / (size(diff, 1) * size(diff, 2))

imwrite(colorDiff, [dir 'diffBetweenWarp_' test_set '.bmp']);
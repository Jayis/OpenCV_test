dir = '../output/OptFlow_tv1/';
test_set = 'rubberWheel';

filenameB = ['warpto1_' test_set];

a = imread(['../input/' test_set '_02.png']);
b = imread([dir filenameB '.bmp']);

[colorDiff, diff] = imgDifference(a, b);
diff

imwrite(colorDiff, [dir 'diffBetweenWarp_' test_set '.bmp']);
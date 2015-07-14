clear all

test_set{1} = 'bb1Crop2000';
test_set{2} = 'bb2Crop2000';
test_set{3} = 'res1Crop2000';
test_set{4} = 'res2Crop2000';
test_set{5} = 'shakeCrop2000';
test_set{6} = 'instroeCrop2000';


for i = 1:length(test_set)
   file_name = [test_set{i} '_01.bmp'];
   
   im  = imread(['../input/' file_name]);
   
   
   up_scale = 2;
   model = 'model/9_5_5_ImageNet/x2.mat'; 
   % up_scale = 4;
   % model = 'model/9_5_5_ImageNet/x4.mat'; 
   
   demo_SR
   
   imwrite(im_h, ['x2/' test_set{i} '_x2.bmp']);
end
a=dir('*.png');
a= struct2cell(a);
b= a(1,:);
disp(b);


for i=drange(b)
    name = i{1};
    x = imread(name);
    noisy = imnoise(x,'salt & pepper',0.02);
    images = strcat('hnoisy',name);
    %[file,path] = uiputfile(images,'Save file name');
    imwrite(noisy,images);
end


for i=drange(b)
    name = i{1};
    x = imread(name);
    rotated = rot90(x,3);
    images = strcat('hrotated',name);
    imwrite(rotated,images);
    
end

for i=drange(b)
    name = i{1};
    x = imread(name);
    flipped = flip(x);
    images = strcat('hflipped',name);
    imwrite(flipped,images);
    
end

for i=drange(b)
    name = i{1};
    x = imread(name);
    flippedlr = fliplr(x);
    images = strcat('hflippedlr',name);
    imwrite(flippedlr,images);
    
end

for i=drange(b)
    name = i{1};
    x = imread(name);
    flippedud = flipud(x);
    images = strcat('hflippedud',name);
    imwrite(flippedud,images);
    
end





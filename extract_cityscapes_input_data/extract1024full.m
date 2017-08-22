clear all;
lst=strsplit(ls('leftImg8bit/train/*/*.png'));
for i=1:numel(lst)-1
    im=im2double(imread(lst{i}));
    im=imresize(im,[1024 2048]);
    if(size(im,3)==1)
        im=repmat(im,[1 1 3]);
    end
    imwrite(im,sprintf('RGB1024Full/%08d.png',i));
end

lst3=strsplit(ls('leftImg8bit/val/*/*.png'));
for i=1:numel(lst3)-1
    im=im2double(imread(lst3{i}));
    im=imresize(im,[1024 2048]);
    if(size(im,3)==1)
        im=repmat(im,[1 1 3]);
    end
    imwrite(im,sprintf('RGB1024Full/%08d.png',i+100000));
end
for i=1:numel(lst)-1
    im=im2double(imread(strrep(strrep(lst{i},'leftImg8bit.png','gtFine_color.png'),'leftImg8bit','gtFine')));
    im=imresize(im,[1024 2048],'nearest');
    imwrite(im,sprintf('Label1024Full/%08d.png',i));
end

for i=1:numel(lst3)-1
    im=im2double(imread(strrep(strrep(lst3{i},'leftImg8bit.png','gtFine_color.png'),'leftImg8bit','gtFine')));
    im=imresize(im,[1024 2048],'nearest');
    imwrite(im,sprintf('Label1024Full/%08d.png',i+100000));
end

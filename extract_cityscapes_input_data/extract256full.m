clear all;
lst=strsplit(ls('leftImg8bit/train/*/*.png'));
parfor i=1:numel(lst)-1
    im=im2double(imread(lst{i}));
    im=imresize(im,[256 512]);
    if(size(im,3)==1)
        im=repmat(im,[1 1 3]);
    end
    imwrite(im,sprintf('RGB256Full/%08d.png',i));
end
lst3=strsplit(ls('leftImg8bit/val/*/*.png'));
parfor i=1:numel(lst3)-1
    im=im2double(imread(lst3{i}));
    im=imresize(im,[256 512]);
    if(size(im,3)==1)
        im=repmat(im,[1 1 3]);
    end
    imwrite(im,sprintf('RGB256Full/%08d.png',i+100000));
end

parfor i=1:numel(lst)-1
    im=im2double(imread(strrep(strrep(lst{i},'leftImg8bit.png','gtFine_color.png'),'leftImg8bit','gtFine')));
    im=imresize(im,[256 512],'nearest');
    imwrite(im,sprintf('Label256Full/%08d.png',i));
end

parfor i=1:numel(lst3)-1
    im=im2double(imread(strrep(strrep(lst3{i},'leftImg8bit.png','gtFine_color.png'),'leftImg8bit','gtFine')));
    im=imresize(im,[256 512],'nearest');
    imwrite(im,sprintf('Label256Full/%08d.png',i+100000));
end

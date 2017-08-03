clear all;
lst=strsplit(ls('../leftImg8bit/train/*/*.png'));
parfor i=1:numel(lst)-1
    im=im2double(imread(lst{i}));
    im=imresize(im,[256 512]);
    if(size(im,3)==1)
        im=repmat(im,[1 1 3]);
    end
    
    rgb = min(max(im,0),1).^0.75;
    cform = makecform('srgb2lab');
    lab = applycform(rgb, cform);
    L = lab(:,:,1);

    %% Filter
    tic
    L0 = wlsFilter(L, 0.125, 1.2);
    L1 = wlsFilter(L, 0.50,  1.2);
    toc

    %% Fine
    val0 = 25;
    val1 = 1;
    val2 = 1;
    exposure = 1.0;
    saturation = 1.1;
    gamma = 1.0;

    fine = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %% Medium
    val0 = 1;
    val1 = 40;
    val2 = 1;
    exposure = 1.0;
    saturation = 1.1;
    gamma = 1.0;

    med = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    %% Coarse
    val0 = 4;
    val1 = 1;
    val2 = 15;
    exposure = 1.10;
    saturation = 1.1;
    gamma = 1.0;

    coarse = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

    im2=(coarse+med+fine)/3;
    
    imwrite(im2,sprintf('../RGB256Full_vivid/%08d.png',i));
end

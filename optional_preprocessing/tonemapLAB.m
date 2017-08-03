function res = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation)

% DESCR:
% This function gets an image in the CIELAB color
% space and tone maps it according to the parameters.
%
% PARAMS:
% lab is the image in CIELAB color space
% L0, L1 are smoothed versions of L of LAB
%
% val0-val3 compression/expansion params in [-1, 1] range
% exposure is in [0,inf) range
% gamma is in (0,1] range
% saturation is in [0,inf) range

L = lab(:,:,1);

if val0==0
    diff0 = L-L0;
else
    if val0>0
        diff0 = sigmoid((L-L0)/100,val0)*100;
    else
        diff0 = (1+val0)*(L-L0);
    end
end

if val1==0
    diff1 = L0-L1;
else
    if val1>0
        diff1 = sigmoid((L0-L1)/100,val1)*100;
    else
        diff1 = (1+val1)*(L0-L1);
    end
end

if val2==0
    base = exposure*L1;
else
    if val2>0
        base = (sigmoid((exposure*L1-56)/100,val2)*100)+56;
    else
        base = (1+val2)*(exposure*L1-56) + 56;
    end
end


if gamma == 1
    res = base + diff1 + diff0;
else
    maxBase = max(base(:));
    res = (zeroone(base).^gamma)*maxBase + diff1 + diff0;
end

if saturation == 0
    lab(:,:,1) = res;
else
    lab(:,:,1) = res;
    lab(:,:,2) = lab(:,:,2) * saturation;
    lab(:,:,3) = lab(:,:,3) * saturation;
end

cform = makecform('lab2srgb');
res = applycform(lab, cform);

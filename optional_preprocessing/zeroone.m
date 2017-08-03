function out = zeroone(in)

maxIn = max(in(:));
minIn = min(in(:));

out = (in-minIn)/(maxIn-minIn);
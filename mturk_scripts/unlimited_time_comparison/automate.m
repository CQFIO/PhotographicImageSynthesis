input=fopen('input.txt');
C = textscan(input, '%s', 'delimiter', '\n');
fclose(input);
content = char(C{1});
ouf=fopen('output.txt','w');
for i=1:110
    for k=1:size(content,1)
        output=strrep(content(k,:),'choice',['choice_' num2str(i)]);
        output=strrep(output,'image_A_url',['image_A_url_' num2str(i)]);
        output=strrep(output,'image_B_url',['image_B_url_' num2str(i)]);
        output=strrep(output,'<strong>Image A</strong>','<strong>Image A is more realistic</strong>');
        output=strrep(output,'<strong>Image B</strong>','<strong>Image B is more realistic</strong>');
        fprintf(ouf,'%s\n',strtrim(output));
    end
    fprintf(ouf,'\n');
end
fclose(ouf);

choice='GAN' %the method to compare to
rng(0)
order=randi(2,[1 500]);
rng(0)
lst=randperm(500);
ouf=fopen(sprintf('data_%s.csv',choice),'w');
a='';
b='';
add=zeros(1,500);
for i=1:5
	add(randperm(100,10)+(i-1)*100)=1;
end
add_order=randi(2,[1 500]);
cnt=0;
for i=1:500
    cnt=mod(cnt,110)+1;
    a=[a 'image_A_url_' num2str(cnt) ',' 'image_B_url_' num2str(cnt) ','];
    if(order(i)==1)
        b=[b sprintf('http://web.stanford.edu/~cqf/mturk/ours/%06d.jpg',lst(i)+3050) ',' sprintf('http://web.stanford.edu/~cqf/mturk/%s/%06d.jpg',choice,lst(i)) ','];
    else
        b=[b sprintf('http://web.stanford.edu/~cqf/mturk/%s/%06d.jpg',choice,lst(i)) ',' sprintf('http://web.stanford.edu/~cqf/mturk/ours/%06d.jpg',lst(i)+3050) ','];
    end
    if(add(i)==1)
        cnt=mod(cnt,110)+1;
        a=[a 'image_A_url_' num2str(cnt) ',' 'image_B_url_' num2str(cnt) ','];
        k=randi(500);
        if(add_order(i)==1)
            b=[b sprintf('http://web.stanford.edu/~cqf/mturk/cityscapes/%06d.jpg',k+100000) ',' sprintf('http://web.stanford.edu/~cqf/mturk/%s/%06d.jpg',choice,k) ','];
        else
            b=[b sprintf('http://web.stanford.edu/~cqf/mturk/%s/%06d.jpg',choice,k) ',' sprintf('http://web.stanford.edu/~cqf/mturk/cityscapes/%06d.jpg',k+100000) ','];
        end
    end
    if(mod(i,100)==0)
        if(i==100)
            fprintf(ouf,'%s\n',a(1:end-1));
        end
        fprintf(ouf,'%s\n',b(1:end-1));
        a=[];
        b=[];
    end
end
fclose(ouf);

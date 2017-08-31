input=fopen('input_timed.txt');
C = textscan(input, '%s', 'delimiter', '\n');
fclose(input);
content = char(C{1});
ouf=fopen('output_timed.txt','w');
for i=1:110
    for k=1:size(content,1)
        output=strrep(content(k,:),'choice',['choice_' num2str(i)]);
        output=strrep(output,'image_A_url',['image_A_url_' num2str(i)]);
        output=strrep(output,'image_B_url',['image_B_url_' num2str(i)]);
        output=strrep(output,'time',['time_' num2str(i)]);
        output=strrep(output,'hiddenA',['hiddenA_' num2str(i)]);
        output=strrep(output,'hiddenB',['hiddenB_' num2str(i)]);
        fprintf(ouf,'%s\n',strtrim(output));
    end
    fprintf(ouf,'\n');
end
fclose(ouf);

rng(0)
ouf=fopen('data_timed_cityscapes.csv','w');
tm=[125 250 500 1000 2000 4000 8000];
%tm=[64 64 64 64 64 64 64 64];
%tm=[125 125 125 125 125 125 125 125];
for j=1:110
    fprintf(ouf,['image_A_url_' num2str(j) ',' 'image_B_url_' num2str(j) ',' 'time_' num2str(j)]);
    if(j==110)
        fprintf(ouf,'\n');
    else
        fprintf(ouf,',');
    end
end
for i=1:10
	lst=randperm(110);
	for j=1:110
        c=randi(500);
        if(rand()<0.5)
            order=[1 2];
        else
            order=[2 1];
        end
        if(lst(j)>105)
            c=randi(500);
            for k=order
                if(k==1)
                    fprintf(ouf,'http://web.stanford.edu/~cqf/mturk/cityscapes/%06d.jpg,',c+100000);                    
                end
                if(k==2)
                    fprintf(ouf,'http://web.stanford.edu/~cqf/mturk/L1/%06d.jpg,',c);                    
                end
            end
            fprintf(ouf,'%d',4000);
        else
            a=mod(lst(j)-1,3)+1;
            b=mod((lst(j)-a)/3,7)+1;
            for k=order
                if((a==1||a==3)&&k==1)
                    fprintf(ouf,'http://web.stanford.edu/~cqf/mturk/ours/%06d.jpg,',c+3050);
                end
                if((a==2&&k==1)||(a==3&&k==2))
                    fprintf(ouf,'http://web.stanford.edu/~cqf/mturk/L1cGAN/%06d.jpg,',c);
                end
                if((a==1||a==2)&&k==2)
                    fprintf(ouf,'http://web.stanford.edu/~cqf/mturk/cityscapes/%06d.jpg,',c+100000);
                end
            end
            fprintf(ouf,'%d',tm(b));
        end
        if(j==110)
            fprintf(ouf,'\n');
        else
            fprintf(ouf,',');
        end        
	end
end
fclose(ouf);

%DATA PREPERATION%

type netflix_data.csv

filename = "netflix_data.csv";
%filename = "politics_data.csv";
%filename = "war_data.csv";

chr = fileread(filename);

chr = strrep( chr, '"positive"', '1');
%chr = strrep( chr, '"negative"', '-1');
%chr = strrep( chr, '"neutral"', '0');

fid = fopen('netflix_data.csv','w');

fprintf(fid,'%s',chr)

fclose(fid);

type netflix_data.csv
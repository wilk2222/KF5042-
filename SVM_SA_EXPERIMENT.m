%%ALL DATA IN THIS EXPERIMENT CAN BE FOUND AT:
%%https://twitter-sentiment-csv.herokuapp.com

%clear existing data to prevent data dublicating
clc;
clear;

emb = fastTextWordEmbedding;

data = readLexicon;

idx = data.Label == "Positive";
head(data(idx,:))

idx = data.Label == "Negative";
head(data(idx,:))

idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = []; 

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

mdl = fitcsvm(XTrain,YTrain);

wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest);

figure
confusionchart(YTest,YPred);

figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(wordsTest(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")


filename = "netflix_data.csv";
%filename = "politics_data.csv";
%filename = "war_data.csv";

tbl = readtable(filename,'TextType','string');
textData = tbl.text;
textData(1:10)

documents = preprocessText(textData);

idx = ~isVocabularyWord(emb,documents.Vocabulary);
documents = removeWords(documents,idx);

words = documents.Vocabulary;
words(ismember(words,wordsTrain)) = [];

vec = word2vec(emb,words);
[YPred,scores] = predict(mdl,vec);

%plot predicted positive sentiment
figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")

%plot predicted negative sentiment
subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")

for i = 1:numel(documents)
    words = string(documents(i));
    vec = word2vec(emb,words);
    [~,scores] = predict(mdl,vec);
    sentimentScore(i) = mean(scores(:,1));
end

table(sentimentScore', textData)

sentimentScore(sentimentScore > 0) = 1;
sentimentScore(sentimentScore < 0) = -1;

%Transpose Matrix tbl.sentiment 1x294 matrix for acc operation.
actualScoreMatrix = tbl.sentiment;
actualScore = actualScoreMatrix';

notfound = sum(sentimentScore == 0);
covered = numel(sentimentScore) - notfound;

tp = sentimentScore((sentimentScore > 0) & ( actualScore > 0));
tn = sentimentScore((sentimentScore < 0) & ( actualScore == 0));

%coverage

fprintf("Coverage: %2.2f%%, found %d, missed: %d\n", (covered * 100) / numel(sentimentScore), covered, notfound);

%Calculate Accuracy
acc = (sum(tp) - sum(tn)) / sum(covered);

fprintf("Accuracy: %2.2f%%, tp: %d, tn; %d\n", acc*100, sum(tp), -sum(tn));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RESULTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                               %
%netflix --> [44.00%,38.00%,41.00%,35.00%,43.00%] -> Avg.Acc.40.20%
%politics -> [38.78%,37.76%,38.78%,38.78%,38.78%] -> Avg.Acc.38.56%
%war ------> [45.00%,53.00%,47.00%,51.00%,48.00%] -> Avg.Acc.48.80%
%                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
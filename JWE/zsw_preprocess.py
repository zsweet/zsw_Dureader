import  json

inputFile = '../DuReader/data/DuReader_v2.0_preprocess/preprocessed/trainset/search.train.json'
outputFile = '../DuReader/data/DuReader_v2.0_preprocess/preprocessed/trainset/search.chinese.train.json'
DEBUG = 0
def get_Chinese_words(sentence,tmp):
    sentenceList = []
    if DEBUG:
        sentenceList.append(tmp)
    for word in sentence:
        tmpFlag = 1
        for chara in word:
            if not (u'\u4e00' <= chara <= u'\u9fff'):
               # print('delete:', word, '\n')
                tmpFlag = 0
                break
        if tmpFlag == 1:
            sentenceList.append(word)
    return sentenceList

def write_file(segmentWordList):
    content = ''
    for sentence in segmentWordList:
        for word in sentence:
            content+=(word+' ')
        content+='\n'
    with open(outputFile,'a') as wfile:
        wfile.write(content)


if __name__ =="__main__":
  with open(inputFile,'rb') as file:
     # line = file.readline()
     # all_uni = line.decode("utf-8")
     # print(all_uni)

     count = 1
     while 1 :
        if count%100 ==0:
             print (count)
        count+=1
        line = file.readline()
        if not line:break
        all_uni = line.decode("utf-8")
        sampleJson = json.loads(all_uni)
        # if not sampleJson:
        #     print('error')
        # else:
        #     print(sampleJson['segmented_answers'],'\n',all_uni)

        segmentedWordsList = []

        try:
            segmentedWordsList.append(get_Chinese_words(sampleJson['segmented_question'],'segmented_question'))
        except Exception as e :
            print('缺失segmented_question')

        try:
            for sentence in sampleJson['segmented_answers']:
                segmentedWordsList.append(get_Chinese_words(sentence,'segmented_answers'))
        except Exception as e:
            print('缺失segmented_question')

        try:
            for paragraph in sampleJson['documents']:
                    segmentedWordsList.append(get_Chinese_words(paragraph['segmented_title'],'segmented_title'),)
                    for sentence in paragraph['segmented_paragraphs']:
                        segmentedWordsList.append(get_Chinese_words(sentence,'segmented_paragraphs'))
        except Exception as e :
            print('缺失segmented_question')

        write_file(segmentedWordsList)


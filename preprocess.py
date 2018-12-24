def preprocess(path):
    starttime = time.time()
    dataset = pd.read_csv(path)
    if('target' in dataset):
        print('Preprocessing train dataset')
        print(dataset.groupby(['target']).size())
        savepath = 'dataset/processed_train.csv'
    else:
        print('Preprocessing test dataset')
        savepath = 'dataset/processed_test.csv'
    stop = set(stopwords.words('english'))
    print(dataset.shape)
    print(dataset.head())
    dataset = dataset.drop(['qid'],axis=1) 
    ps = PorterStemmer()
    for idx,row in dataset.iterrows():
        nval = ''
        for val in row['question_text'].split(' '):
            val = re.sub('[^A-Za-z]+',' ',val)
            if(val != ' '):
                if val.lower() not in stop:
                    nval = nval + ' ' + ps.stem(val.lower())
        dataset.at[idx,'question_text'] = nval
    print(dataset.shape)
    print(dataset.head())
    dataset.to_csv(savepath,sep=',')
    endtime = time.time()
    print("Time for preprocessing")
    print(endtime - starttime)
    
def create_vocabulary(min_frequency=0):
    starttime = time.time()
    dataset = pd.read_csv('dataset/processed_train.csv')
    dataset = dataset.drop(dataset.columns[0],axis=1)
    print(dataset.shape)
    print(dataset.head())
    max_length = 0
    for idx,row in dataset.iterrows():
        length = len(str(row['question_text']).strip().split())
        if length > max_length:
            max_length = length
    print("Max length:- ")
    print(max_length)
    with open('processed/max_length.txt','w') as file:
        file.write(str(max_length))  
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length,min_frequency=min_frequency)
    for idx,row in dataset.iterrows():
        vocab_processor.fit(str(row['question_text']))
    vocab_processor.save('processed/vocab')
    endtime = time.time()
    print('Time to create vocabulary')
    print(endtime - starttime)

def get_batch(data,labels,lengths,batch_size,epochs):
    assert len(data) == len(labels) == len(lengths)
    no_batches = len(data) // batch_size
    for i in range(1,epochs):
        for j in range(1,no_batches):
            start_index = j * batch_size
            end_index = start_index + batch_size
            x_data = data[start_index:end_index]
            y_data = labels[start_index:end_index]
            length_data = lengths[start_index:end_index]
            yield x_data,y_data,length_data

def get_processed_batch_data(data,vocab_processor,count):
    starttime = time.time()
    lengths = []
    labels = []
    processed_data = []
    for idx,row in data.iterrows():
        lengths.append(len(str(row['question_text']).strip().split(' ')))
        labels.append(row['target'])
        processed_data.append(list(vocab_processor.transform(str(row['question_text']))))
    lengths = np.array(lengths)
    labels = np.array(labels)
    processed_data = np.array(processed_data)
    endtime = time.time()
    print('Time to create batch {:g}'.format(count))
    print(endtime - starttime)
    return data,labels,lengths

def save_testing_data(x,y,lengths):
    

def preprocess():
    if(not os.path.isfile('dataset/processed_train.csv')):
        preprocess('dataset/train.csv')
    if(not os.path.isfile('dataset/processed_test.csv')):
        preprocess('dataset/test.csv')
    if(not os.path.isfile('processed/vocab')):
        create_vocabulary()

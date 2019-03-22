import numpy as np 

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def data_cifar10():
	dataset_folder = '../dataset/cifar-10-batches-py/'
	dataset = {}
	for i in range(1,6):
		train_filename = 'data_batch_'+str(i)
		train_data = unpickle(dataset_folder+train_filename)
		if i == 1:
			dataset['training_features'] = train_data['data']
			dataset['training_labels'] = np.array(train_data['labels'])
		else:
			dataset['training_features'] = np.concatenate((dataset['training_features'],train_data['data']))
			dataset['training_labels'] = np.concatenate((dataset['training_labels'],np.array(train_data['labels'])))
	test_filename = 'test_batch'
	test_data = unpickle(dataset_folder+test_filename)
	dataset['test_features'] = test_data['data']
	dataset['test_labels'] = np.array(test_data['labels'])
	return dataset
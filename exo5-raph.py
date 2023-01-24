
###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np
from matplotlib import pyplot

category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    # Convert data to encoding
    x_train_encodings = tokenizer(x_train, padding=True, truncation=True, return_tensors='pt')
    x_test_encodings = tokenizer(x_test, padding=True, truncation=True, return_tensors='pt')
    # Create train and test datasets
    train_dataset = Dataset(x_train_encodings, y_train)
    test_dataset = Dataset(x_test_encodings, y_test)
    # Create the trainer and train the model
    training_args = TrainingArguments(output_dir='./results',
                                      eval_steps=1000,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      num_train_epochs=5,
                                      weight_decay=0.01,
                                      learning_rate=5e-5)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
                      compute_metrics=compute_metrics)
    trainer.train()
    result = trainer.evaluate()
    return result['eval_accuracy']


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    candidate_labels = list(category_dict.values())
    predictions = clf(x_test, candidate_labels=candidate_labels)
    predictions = [p['labels'][0] for p in predictions]
    y_test = [list(category_dict.values())[i] for i in y_test]
    accuracy = accuracy_score(y_test, predictions)
    print("\nZero-shot result:")
    return accuracy


def graphs(result, portions, ylabel, xlabel, title):
    pyplot.title(title)
    pyplot.plot(portions, result, label="Accuracy")
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.0]
    result_linear_classification = []
    # Q1
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        result = linear_classification(p)
        result_linear_classification.append(result)
        print(result)
    graphs(result_linear_classification, portions, "Accuracy", "Portion",
           "Model 1 : accuracy results as a function of the portion of the data")

    result_transformer_classification = []
    # Q2
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        result = transformer_classification(portion=p)
        result_transformer_classification.append(result)
        print(result)
    for i in range(len(result_transformer_classification)):
        print("Portion ", portions[i], " : ", result_transformer_classification[i])
    graphs(result_transformer_classification, portions, "Accuracy", "Portion",
           "Model 2 : accuracy results as a function of the portion of the data")

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())

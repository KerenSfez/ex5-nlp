

###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np

# subset of categories that we will use
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
    train_len = int(portion*len(data_train.data))
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

    classifier = LogisticRegression()

    encoded_train_text = tf.fit_transform(x_train)
    encoded_test_text = tf.transform(x_test)

    classifier.fit(encoded_train_text, y_train)
    results = classifier.predict(encoded_test_text)
    return accuracy_score(y_test, results)


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

    encoded_train_text = tokenizer(x_train, padding=True, truncation=True, return_tensors='pt')
    train_dataset = Dataset(encoded_train_text, y_train)

    encoded_test_text = tokenizer(x_test, padding=True, truncation=True, return_tensors='pt')
    eval_dataset = Dataset(encoded_test_text, y_test)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=5e-5)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
    trainer.train()
    return trainer.evaluate()['eval_accuracy']


    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator
    return


def get_labels_prediction(predictions):
    labels_prediction = []
    for prediction in predictions:
        labels_prediction.append(prediction['labels'][0])
    return labels_prediction


def get_labels_test(y_test):
    labels_test = []
    for y in y_test:
        labels_test.append(y)
    return labels_test

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
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
    candidate_labels = list(category_dict.values())

    results = clf(x_test, candidate_labels=candidate_labels)
    labels_prediction = get_labels_prediction(results)
    labels_test = get_labels_test(y_test)
    return accuracy_score(labels_test, labels_prediction)


def draw_plot(plot_title, portions, accuracy_results):
    plt.title(plot_title)
    plt.plot(portions, accuracy_results)
    plt.xlabel('Portion of data')
    plt.ylabel('Accuracy')
    plt.show()


import matplotlib.pyplot as plt

if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]

    # Q1
    accuracy_results = []
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        accuracy = linear_classification(p)
        accuracy_results.append(accuracy)
        print(accuracy)
    draw_plot("Plot Logistic regression: ", portions, accuracy_results)

    # Q2
    accuracy_results = []
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        accuracy = transformer_classification(portion=p)
        accuracy_results.append(accuracy)
        print(accuracy)
    draw_plot("Plot Finetune: ", portions, accuracy_results)

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())

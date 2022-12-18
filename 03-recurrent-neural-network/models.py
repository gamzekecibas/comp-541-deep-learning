# models.py

import numpy as np
import collections
import torch
import random
import time
#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")

class RNNModel(torch.nn.Module):
    def __init__(self, dict_size):
        super(RNNModel, self).__init__()
        embed_dim = 27
        self.embedding = torch.nn.Embedding(dict_size, embed_dim)
        self.hidden_num = 30
        self.lstm_layer = torch.nn.LSTM(embed_dim, self.hidden_num)
        self.linear = torch.nn.Linear(self.hidden_num, 1)

    def forward(self, input):
        embeds = self.embedding(input)
        embeds = embeds.unsqueeze(1)
        first_hidden = (torch.from_numpy(np.zeros(self.hidden_num)).unsqueeze(0).unsqueeze(1).float(),
                        torch.from_numpy(np.zeros(self.hidden_num)).unsqueeze(0).unsqueeze(1).float())
        _, (hidden_rnn, _) = self.lstm_layer(embeds, first_hidden)
        lstm_out = self.linear(hidden_rnn)
        lstm_out = lstm_out.squeeze(1)
        return lstm_out

class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def predict(self, context):
        ## raise Exception("Implement me")
        input = [self.vocab_index.index_of(char) for char in context]
        context = torch.LongTensor(input)
        prediction = int(self.model(context) >=0)
        return prediction


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    #raise Exception("Implement me")
    cons_exs_len = len(train_cons_exs)
    train_exs = train_cons_exs + train_vowel_exs
    train_data = []

    for i, train_ex in enumerate(train_exs):
        train_label = 0 if i < cons_exs_len else 1
        train_label = torch.FloatTensor([[train_label]])
        train_char_idx = [vocab_index.index_of(char) for char in train_ex]
        train_ex = torch.LongTensor(train_char_idx)
        train_data.append((train_ex, train_label))
    
    model = RNNModel(len(vocab_index)).train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    epochs_num = 10
    
    start_time_epochs = time.time()
    for epoch in range(epochs_num):
        print('## EPOCH: ', epoch+1, ' ##')
        random.shuffle(train_data)
        for ex, label in train_data:
            start_time_each_epoch = time.time()
            o = model(ex)
            loss = loss_function(o, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('LOSS: ', loss.item(), '|| EPOCH TIME: ', time.time() - start_time_each_epoch, 'seconds \n')
    model.eval()
    print('TOTAL TRAINING TIME: ', time.time() - start_time_epochs, 'seconds \n')
    return RNNClassifier(model, vocab_index)

#####################
# MODELS FOR PART 2 #
#####################

class LMRNNModel(torch.nn.Module):
    def __init__(self, dict_size):
        super(LMRNNModel, self).__init__()
        embed_dim = 30
        self.embedding = torch.nn.Embedding(dict_size, embed_dim)
        self.hidden_num = 50
        self.lstm_layer = torch.nn.LSTM(embed_dim, self.hidden_num)
        self.linear = torch.nn.Linear(self.hidden_num, dict_size)

    def forward(self, input):
        embeds = self.embedding(input)
        embeds = embeds.unsqueeze(1)
        first_hidden = (torch.from_numpy(np.zeros(self.hidden_num)).unsqueeze(0).unsqueeze(1).float(),
                        torch.from_numpy(np.zeros(self.hidden_num)).unsqueeze(0).unsqueeze(1).float())
        out, (_, _) = self.lstm_layer(embeds, first_hidden)
        out = out.permute(1, 0, 2)
        return self.linear(out)[0]

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        #raise Exception("Implement me")
        self.model = model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        #raise Exception("Implement me")
        context_idx = torch.LongTensor([self.vocab_index.index_of(char) for char in context])
        output = self.model(context_idx)[-1]
        output = torch.nn.LogSoftmax(dim=0)(output)
        return output.detach().numpy()
    
    def get_log_prob_sequence(self, next_chars, context):
        #raise Exception("Implement me")
        log_prob =0
        for i, char in enumerate(next_chars):
            log_prob += self.get_next_char_log_probs(context)[self.vocab_index.index_of(char)]
            context += char
        return float(log_prob)

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    #raise Exception("Implement me")
    train_data = []
    chunk_size = 15

    for i in range(len(train_text) - chunk_size):
        train_ex = train_text[i:i+chunk_size]
        train_label = train_text[i+1:i+chunk_size+1]
        train_ex = torch.LongTensor([vocab_index.index_of(char) for char in train_ex])
        train_label = torch.LongTensor([vocab_index.index_of(label) for label in train_label])
        train_data.append((train_ex, train_label))
    
    model = LMRNNModel(len(vocab_index)).train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    epochs_num = 10

    start_time_epochs = time.time()

    for epoch in range(epochs_num):
        print('## EPOCH: ', epoch+1, ' ##')
        random.shuffle(train_data)
        for ex, label in train_data:
            start_time_each_epoch = time.time()
            pred = model(ex)
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('LOSS: ', loss.item(), '|| EPOCH TIME: ', time.time() - start_time_each_epoch, 'seconds \n')
    model.eval()

    print('TOTAL TRAINING TIME: ', time.time() - start_time_epochs, 'seconds \n')
    
    return RNNLanguageModel(model, vocab_index)
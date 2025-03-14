import os
import re
import codecs
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from preprocessing import textPreprocessing
import ast
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

def load_movies(path2data):
    alltxts = []
    labs = []
    cpt = 0
    for cl in os.listdir(path2data):
        class_dir = os.path.join(path2data, cl)
        # Check if it is a directory
        if not os.path.isdir(class_dir):
            continue
        try:
            for f in os.listdir(class_dir):
                file_path = os.path.join(class_dir, f)
                with open(file_path, encoding='utf-8') as file:
                    txt = file.read()
                alltxts.append(txt)
                labs.append(cpt)
            cpt += 1
        except Exception as e:
            print(f"Error reading files in {class_dir}: {e}")
    return alltxts, labs

# Chargement des données:
def load_pres(fname):
  alltxts = []
  alllabs = []
  s=codecs.open(fname, 'r','utf-8') # pour régler le codage
  
  while True:
      txt = s.readline()
      
      if(len(txt))<5:
          break
      #
      lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
      txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
      
      if lab.count('M') >0:
          alllabs.append(-1)
      else:
          alllabs.append(1)

      alltxts.append(txt)
  return alltxts, alllabs


def wordCounter(texts):
    wc = Counter()
    for text in texts:
        wc.update(text.split())

    return wc

# Document frequency of each word
def document_frequency(texts):
    doc_freq = Counter()
    for text in texts:
        words = set(text.split())  # Un mot est compté une fois par document
        doc_freq.update(words)
    return doc_freq

def calculate_odds_ratios(texts, labels):
    unique_labels = sorted(set(labels))
    class_freq = {label: Counter() for label in unique_labels}

    # Count word frequencies per class
    for text, label in zip(texts, labels):
        words = text.split()
        class_freq[label].update(words)

    # Extract words across both classes
    vocab = set(class_freq[unique_labels[0]].keys()).union(class_freq[unique_labels[1]].keys())

    # Compute odds ratios
    odds_ratios = {}
    for word in vocab:
        freq_class0 = class_freq[unique_labels[0]].get(word, 1)  # Avoid zero division
        freq_class1 = class_freq[unique_labels[1]].get(word, 1)
        odds_ratios[word] = np.log((freq_class1 / freq_class0))

    return odds_ratios


def select_subset(alltxts, alllabs):

    # Convert to NumPy arrays for easy indexing
    alltxts = np.array(alltxts)
    alllabs = np.array(alllabs)

    # Find indices of class 0 and class 1
    class0_indices = np.where(alllabs == 0)[0]  # Indices where label is 0
    if len(class0_indices) == 0:
        class0_indices = np.where(alllabs == -1)[0]  # Indices where label is -1
    class1_indices = np.where(alllabs == 1)[0]  # Indices where label is 1

    # Select 10 samples from each class (if available)
    selected_class0 = class0_indices[:100]  # First 10 samples of class 0
    selected_class1 = class1_indices[:50]  # First 10 samples of class 1

    # Combine selected indices
    selected_indices = np.concatenate([selected_class0, selected_class1])

    # Extract subset
    alltxts_subset = alltxts[selected_indices].tolist()  # Convert back to list
    alllabs_subset = alllabs[selected_indices].tolist()  # Convert back to list

    return alltxts_subset, alllabs_subset

def best_oversampler_undersampler(basic_results_under, basic_results_over):
    # Retreive model scores
    model_result_under = basic_results_under[:, 1:4]
    model_result_over = basic_results_over[:, 1:4]

    # Average results
    mean_result_under = np.mean(model_result_under)
    mean_result_over = np.mean(model_result_over)

    # Return sampling parameters
    if mean_result_under > mean_result_over:
        return True, False
    
    return False, True


def get_best_ngram(n_grams_basic_results, titles_n_grams, sampling=False):

    best_score = -np.inf
    best_ngram = None
    sampler = 'None'

    for i, basis_result_ngram in enumerate(n_grams_basic_results):
        avg_score = np.mean(basis_result_ngram[:, 1:4])
        
        if avg_score > best_score:
            best_score = avg_score
            best_ngram = titles_n_grams[i]

            if sampling:
                ngram, sampler = best_ngram.split(' - ')
                best_ngram_tuple = ast.literal_eval(ngram)
            else:
                best_ngram_tuple = ast.literal_eval(best_ngram)

    if sampler.strip() == 'under':
        return best_ngram_tuple, True, False
    elif sampler.strip() == 'over':
        return best_ngram_tuple, False, True
    
    return best_ngram_tuple, False, False


def optimal_model_CV(nb_scores, lr_scores, svm_scores):
    
    nb_scores = np.array(nb_scores)
    lr_scores = np.array(lr_scores)
    svm_scores = np.array(svm_scores)

    # Compute the mean over processes for each parameter setting
    nb_mean = np.mean(nb_scores, axis=0)
    lr_mean = np.mean(lr_scores, axis=0)
    svm_mean = np.mean(svm_scores, axis=0)
    
    # Extract the best (maximum) average score for each model
    best_nb = np.max(nb_mean)
    best_lr = np.max(lr_mean)
    best_svm = np.max(svm_mean)
    
    # Determine which model has the highest best average score,
    if best_nb > best_lr and best_nb > best_svm:
        return 'NB', np.argmax(nb_mean)
    elif best_lr > best_nb and best_lr > best_svm:
        return 'LR', np.argmax(lr_mean)
    else:
        return 'SVM', np.argmax(svm_mean)
    
def retreive_tuned_param(proccess, model, results_max_features, results_max_df, results_min_df):

    def get_best_parameter_for_model(results_array, process, model):
        # Map model name to the corresponding column index
        model_col = {'NB': 1, 'LR': 2, 'SVM': 3}[model]
        
        # Filter rows where the process column equals the given process value
        filtered = results_array[results_array[:, 4] == process]
        
        # Find the row with the maximum score in the chosen model's score column
        best_row = filtered[np.argmax(filtered[:, model_col])]
        
        # Return the parameter from that row
        return best_row[0]

    # Retreive best parameters
    max_features_opt = get_best_parameter_for_model(results_max_features, proccess, model)
    max_df_opt = get_best_parameter_for_model(results_max_df, proccess, model)
    min_df_opt = get_best_parameter_for_model(results_min_df, proccess, model)

    return max_features_opt, max_df_opt, min_df_opt


def final_model_evaluation(max_features_cv, max_df_cv, min_df_cv, 
                           model_type, vectorizer_type, ngram_range, 
                           train_texts, test_texts, train_labels, test_labels,
                           undersampling=False, oversampling=False):

    # 1. Instantiate the vectorizer based on type and optimized parameters.
    if vectorizer_type.lower() == "count":
        vectorizer = CountVectorizer(max_features=max_features_cv,
                                     max_df=max_df_cv,
                                     min_df=min_df_cv,
                                     ngram_range=ngram_range)
    elif vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features_cv,
                                     max_df=max_df_cv,
                                     min_df=min_df_cv,
                                     ngram_range=ngram_range)
    else:
        raise ValueError("Invalid vectorizer_type. Choose 'count' or 'tfidf'.")
    
    # 2. Fit the vectorizer on training data, then transform both train and test texts.
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    # Apply oversampling, undersampling
    X_train, train_labels = balance_dataset(X_train, train_labels, undersampling=undersampling, oversampling=oversampling)

    # 3. Initialize the classifier based on the chosen model.
    if model_type.upper() == "LR":
        clf = LogisticRegression(random_state=0, solver='sag', max_iter=50000, tol=1e-8, C=1.0)
    elif model_type.upper() == "NB":
        clf = MultinomialNB()
    elif model_type.upper() == "SVM":
        clf = LinearSVC(random_state=0, tol=1e-8, max_iter=50000)
    else:
        raise ValueError("Invalid model_type. Choose 'LR', 'NB', or 'SVM'.")
    
    # 4. Train the classifier on the training data.
    clf.fit(X_train, train_labels)
    
    # 5. Predict on the test data.
    y_pred = clf.predict(X_test)
    
    # 6. Compute performance metrics.
    acc   = accuracy_score(test_labels, y_pred)
    prec  = precision_score(test_labels, y_pred, average='weighted')
    rec   = recall_score(test_labels, y_pred, average='weighted')
    f1    = f1_score(test_labels, y_pred, average='weighted')
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # 7. Create an array of the metrics.
    metrics = np.array([acc, prec, rec, f1])
    
    return metrics, conf_matrix


def process_txts(alltxts, process):
    
    # Preprocess text
    textpreprocessor = textPreprocessing()

    if process == 0:
        return alltxts
    elif process == 1:
        # Stemming
        alltxts_clean1 = [textpreprocessor.preprocess_text(text, stemming=True, stop_words=True,
                                                        languages=['english']) for text in alltxts]
        return alltxts_clean1
    elif process == 2: 
        # Lemmatize
        alltxts_clean2 = [textpreprocessor.preprocess_text(text, lemmatize=True, stop_words=True,
                                                        languages=['english']) for text in alltxts]
        return alltxts_clean2
    elif process == 3:
        # Lemmatize & last sentence
        alltxts_clean3 = [textpreprocessor.preprocess_text(text, keep_first_n=0, keep_last_n=2, lemmatize=True,
                                                        stop_words=True, languages=['english']) for text in alltxts]
        return alltxts_clean3
    
    # Lemmatize & last/first 9 sentence
    n_first, n_last = 9, 9
    alltxts_clean4 = [textpreprocessor.preprocess_text(text, keep_first_n=n_first, keep_last_n=n_last,
                                    lemmatize=True, stop_words=True, languages=['english']) for text in alltxts]
    return alltxts_clean4


def process_txts_pres(alltxts, process):
    # Preprocess text
    textpreprocessor = textPreprocessing()

    if process == 0:
        return alltxts
    elif process == 1:
        # Stemming
        alltxts_clean1 = [textpreprocessor.preprocess_text(text, stemming=True, stop_words=True,
                                                   languages=['french']) for text in alltxts]
        return alltxts_clean1

    # Lemmatize
    alltxts_clean2 = [textpreprocessor.preprocess_text(text, lemmatize=True, stop_words=True,
                                                   languages=['french']) for text in alltxts]
    return alltxts_clean2


def balance_dataset(X, y, undersampling=False, oversampling=False, strategy=0.8, random_state=42):
    
    if oversampling and undersampling:
        # Apply SMOTE + Tomek Links (Hybrid method)
        sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state)
    elif oversampling:
        # Apply SMOTE (Oversampling)
        sampler = SMOTE(sampling_strategy=strategy, random_state=random_state, k_neighbors=2)
    elif undersampling:
        # Apply Random Undersampling
        sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
    else:
        return X, y

    # Apply chosen resampling method
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled


def best_model(tune_maxf, tune_max_df, tune_min_df):
    
    def results_best_scores(tune_result):
        # Extract models
        processes = np.unique(tune_result[:, -1])
        # Retreive maximum tuning scores for every process
        results = []
        for process in processes:
            # Filter rows for the current model
            process_rows = tune_result[tune_result[:, -1] == process]
            # Extract scores and compute max per column (axis=0 keeps columns)
            max_scores = np.max(process_rows[:, 1:4].astype(float), axis=0)
            results.append([process, *max_scores])

        return np.array(results)
    
    r_maxf = results_best_scores(tune_maxf)
    r_max_df = results_best_scores(tune_max_df)
    r_min_df = results_best_scores(tune_min_df)

    # Stack all arrays along a new axis
    stacked = np.stack([r_maxf[:, 1:], r_max_df[:, 1:], r_min_df[:, 1:]], axis=2)

    # Compute the mean across the third dimension
    average_scores = np.mean(stacked, axis=2)

    # Combine process IDs with averaged scores
    final_tune = np.hstack([r_min_df[:, [0]], average_scores])

    # Transform results
    final_tune = np.array(final_tune, dtype=object)
    numeric_scores = final_tune[:, 1:4].astype(float)

    # Find the best model on average in the process
    mean = np.mean(numeric_scores, axis=0)
    model_ind = np.argmax(mean)
    # Choose model
    models = {0: "NB", 1:"LR", 2:"SVM"}
    model = models[model_ind]

    # Choose process
    process = np.argmax(final_tune[:,model_ind+1])
    
    return model, process


def plot_confusion_matrix(conf_matrix, labels, figure_name, figure_folder):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='black')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(figure_folder, figure_name))
    plt.close()

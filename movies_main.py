import os
import json
from preprocessing import textPreprocessing
from vocabularyAnalysis import vocabularyAnalysis
from BasicTextClassificationModel import BasicTextClassificationModel
from nlp_tools import *

print('step 0: loading data')
# Retreive data
current_dir = os.getcwd()
target_path = os.path.join(current_dir, "data", "movies", "movies_big", "json_pol.json")

with open(target_path,encoding="utf-8") as f:
    data = json.load(f)

data_train, data_test = data["train"], data["test"]
alltxts_train = [text for text,pol in data_train]
llabs_train = [pol for text,pol in data_train]
alltxts_test = [text for text,pol in data_test]
llabs_test = [pol for text,pol in data_test]

alltxts, alllabs = alltxts_train, llabs_train
y_test = llabs_test

print('step 1: processing data')
# Preprocess text
textpreprocessor = textPreprocessing()
# Stemming
alltxts_clean1 = [textpreprocessor.preprocess_text(text, stemming=True, stop_words=True,
                                                   languages=['english']) for text in alltxts]
# Lemmatize
alltxts_clean2 = [textpreprocessor.preprocess_text(text, lemmatize=True, stop_words=True,
                                                   languages=['english']) for text in alltxts]
# Lemmatize & last/first sentence
n_first, n_last = 4, 4
alltxts_clean4 = [textpreprocessor.preprocess_text(text, keep_first_n=n_first, keep_last_n=n_last,
                                  lemmatize=True, stop_words=True, languages=['english']) for text in alltxts]

# Save different processes with its titles in lists
titles = ['No process', 'stem', 'lemma', f'lemma F{n_first} - L{n_last}']
processed_txts = [alltxts, alltxts_clean1, alltxts_clean2, alltxts_clean4]

figures_path = os.path.join(current_dir, "results", "movies", "figures")
results_path = os.path.join(current_dir, "results", "movies")

print('step 2: visualization data')
# Start vocabulary analysis
analysisVocabMovies = vocabularyAnalysis(processed_txts, titles, figures_folder=figures_path)
# Vocab size
analysisVocabMovies.vocabSize()
# wordclouds
analysisVocabMovies.create_wordclouds(frequency_function=wordCounter, frequency_f_name='wordcounter')
analysisVocabMovies.create_wordclouds(frequency_function=document_frequency, frequency_f_name='document_freq')
analysisVocabMovies.create_wordclouds(alllabs=alllabs, frequency_function=calculate_odds_ratios, frequency_f_name='odds_ratio')
# Frequency distribution    
analysisVocabMovies.plot_freq_distribution(frequency_function=wordCounter)
analysisVocabMovies.freq_distribution_vectorizer()
# N-gram vocab size
analysisVocabMovies.Ngram_vocabSize()
# Data-set bias
analysisVocabMovies.plot_label_distribution(alllabs)

analysis_movie = BasicTextClassificationModel(processed_txts, alllabs, titles, figures_folder=figures_path, result_folder=results_path)

print('step 3.1: tuning classification models - count vectorizer')

# defeault parameter results count vectorizer
if os.path.exists(os.path.join(results_path, "results_basic_count.txt")):
    default_cv = np.genfromtxt(os.path.join(results_path, "results_basic_count.txt"),
                                  delimiter=",", dtype=float, filling_values=np.nan)
else:
    default_cv = analysis_movie.basic_classification()

# Plot results classification default parameters
analysis_movie.plot_basic_results([default_cv], ["CountVectorizer"], title_figure='countvectorizer')

# Model results with max_features tuning
if os.path.exists(os.path.join(results_path, "results_count_max_features.txt")):
    tune_maxf = np.loadtxt(os.path.join(results_path, "results_count_max_features.txt"), delimiter=",")
else:
    tune_maxf = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(4000,35000,500), parameter_name='max_features')

# Model results with max_df tuning
if os.path.exists(os.path.join(results_path, "results_count_max_df.txt")):
    tune_max_df = np.loadtxt(os.path.join(results_path, "results_count_max_df.txt"), delimiter=",")
else:
    tune_max_df = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.03), parameter_name='max_df')

# Model results with min_df tuning
if os.path.exists(os.path.join(results_path, "results_count_min_df.txt")):
    tune_min_df = np.loadtxt(os.path.join(results_path, "results_count_min_df.txt"), delimiter=",")
else:
    tune_min_df = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(0.002,0.08,0.002), parameter_name='min_df')

# Plot results classification tuning parameters
analysis_movie.plot_results_tuning([tune_maxf, tune_max_df, tune_min_df], ['max_features', 'max_df', 'min_df'], figure_name='count')

# Save best final model for countvectorizer
best_model_cv, process_cv = best_model(tune_maxf, tune_max_df, tune_min_df)

print('step 3.2: tuning classification models - tfidf vectorizer')

# defeault parameter results tfidf vectorizer
if os.path.exists(os.path.join(results_path, "results_basic_tfidf.txt")):
    default_tfidf = np.genfromtxt(os.path.join(results_path, "results_basic_tfidf.txt"),
                                  delimiter=",", dtype=float, filling_values=np.nan)
else:
    default_tfidf = analysis_movie.basic_classification(vectorizer_type='tfidf')

analysis_movie.plot_basic_results([default_tfidf], ["TFIDF Vectorizer"], title_figure='tfidf')

# Evaluate tfidf parameters for best initial results
use_idf, smooth_idf, sublinear_tf = analysis_movie.evaluate_tfidf_variations()

# Model results with max_features tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_max_features.txt")):
    maxf_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_max_features.txt"), delimiter=",")
else:
    maxf_tfidf = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(5000,35000,500), vectorizer_type='tfidf',
                                                                    parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                                    parameter_name='max_features')
                                                               
# Model results with max_df tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_max_df.txt")):
    max_df_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_max_df.txt"), delimiter=",")
else:
    max_df_tfidf = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.03), vectorizer_type='tfidf',
                                                                    parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                                    parameter_name='max_df')
                                                                    
# Model results with min_df tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_min_df.txt")):
    min_df_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_min_df.txt"), delimiter=",")
else:
    min_df_tfidf = analysis_movie.vectorizer_parameter_analysis(parameters_loop=(0.002,0.08,0.002), vectorizer_type='tfidf',
                                                                        parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                                        parameter_name='min_df')
                                                                    
analysis_movie.plot_results_tuning([maxf_tfidf, max_df_tfidf, min_df_tfidf],
                                   ['max_features', 'max_df', 'min_df'], figure_name='tf_idf')

# Retreive model and text process which give best results
best_model_tfidf, process_tfidf = best_model(maxf_tfidf, max_df_tfidf, min_df_tfidf)

print('step 3.3: tuning classification models - n-grams')

# Classification with default vectorizer values for ngrams 
ngrams = [(1,2), (2,2), (1,3), (2,3), (3,3)]
n_grams_basic_results = []
for ngram in ngrams:
    n_grams_basic_results.append(analysis_movie.basic_classification(vectorizer_type='tfidf', n_gram_range=ngram))

# Plot Results
titles_n_grams  = ["(1,2)", "(2,2)", "(1,3)", "(2,3)", "(3,3)"]
analysis_movie.plot_basic_results(n_grams_basic_results, titles_n_grams, title_figure='ngrams')

# Get best performing n-gram from basic result
ngram_opt, _, _ = get_best_ngram(n_grams_basic_results, titles_n_grams)
n1, n2 = ngram_opt

# Model results with max_features tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_max_features_ngram{n1}-{n2}.txt")):
    maxf_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_max_features_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    maxf_ngram = analysis_movie.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(7000,40000,500),
                                                               n_gram_range=ngram_opt,
                                                               parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                               parameter_name='max_features')
    
# Model results with max_df tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_max_df_ngram{n1}-{n2}.txt")):
    max_df_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_max_df_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    max_df_ngram = analysis_movie.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.2,0.9,0.1),
                                                                    n_gram_range=ngram_opt,
                                                                    parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                                    parameter_name='max_df')

# Model results with min_df tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_min_df_ngram{n1}-{n2}.txt")):
    min_df_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_min_df_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    min_df_ngram = analysis_movie.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.002,0.08,0.002),
                                                                    n_gram_range=ngram_opt,
                                                                    parameters_tf_idf=(use_idf, smooth_idf, sublinear_tf),
                                                                    parameter_name='min_df')

# Plot results
analysis_movie.plot_results_tuning([maxf_ngram, max_df_ngram, min_df_ngram], ['max_features', 'max_df', 'min_df'],
                                   figure_name='ngram')

# Save best final model for ngram
best_model_ngram, process_ngram = best_model(maxf_ngram, max_df_ngram, min_df_ngram)

print('step 4: Obtained tuned parameter for best peforming model and process')

# Get optimized parameters for each model
max_features_cv, max_df_cv, min_df_cv = retreive_tuned_param(process_cv, best_model_cv,
                                                                tune_maxf, tune_max_df, tune_min_df)

max_features_tfidf, max_df_tfidf, min_df_tfidf = retreive_tuned_param(process_tfidf, best_model_tfidf,
                                                                maxf_tfidf, max_df_tfidf, min_df_tfidf)

max_features_ngram, max_df_ngram, min_df_ngram = retreive_tuned_param(process_ngram, best_model_ngram,
                                                                maxf_ngram, max_df_ngram, min_df_ngram)


print('tuned param cv: ', max_features_cv, max_df_cv, min_df_cv)
print('tuned param tfidf: ', max_features_tfidf, max_df_tfidf, min_df_tfidf)
print('tuned param ngram: ', max_features_ngram, max_df_ngram, min_df_ngram)

# final model evaluation cv
X_train_texts = processed_txts[process_cv]
X_test_texts = process_txts(alltxts_test, process_cv)
metrics_cv, conf_cv  = final_model_evaluation(int(max_features_cv), max_df_cv, min_df_cv,
                                                 best_model_cv, "count", (1,1),
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test)


plot_confusion_matrix(conf_cv, np.unique(y_test), "confusion_cv.png", figures_path)

# final model evaluation tfidf
X_train_texts = processed_txts[process_tfidf]
X_test_texts = process_txts(alltxts_test, process_tfidf)
metrics_tfidf, conf_ctfidf  = final_model_evaluation(int(max_features_tfidf), max_df_tfidf, min_df_tfidf,
                                                 best_model_tfidf, "tfidf", (1,1),
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test)

plot_confusion_matrix(conf_ctfidf, np.unique(y_test), "confusion_tfidf.png", figures_path)


# final model evaluation n-gram
X_train_texts = processed_txts[process_ngram]
X_test_texts = process_txts(alltxts_test, process_ngram)
metrics_ngram, conf_ngram  = final_model_evaluation(int(max_features_ngram), max_df_ngram, min_df_ngram,
                                                 best_model_ngram, "tfidf", ngram_opt,
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test)

plot_confusion_matrix(conf_ngram, np.unique(y_test), "confusion_ngram.png", figures_path)

# Append final results
metrics_cv = np.append(metrics_cv, [best_model_cv, process_cv, max_features_cv, max_df_cv,
                                    min_df_cv, '(1,1)' ,'count'])
metrics_tfidf = np.append(metrics_tfidf, [best_model_tfidf, process_tfidf, max_features_tfidf, max_df_tfidf,
                                          min_df_tfidf, '(1,1)', 'tfidf'])
metrics_ngram = np.append(metrics_ngram, [best_model_ngram, process_ngram, max_features_ngram, max_df_ngram,
                                          min_df_ngram, str(ngram_opt),'ngram'])
final_results = np.vstack([metrics_cv, metrics_tfidf, metrics_ngram])

# Save final model results
np.savetxt(os.path.join(results_path, f"final_results_model.txt"), final_results, fmt='%s', delimiter=",")
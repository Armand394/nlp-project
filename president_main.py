import os
import json
from preprocessing import textPreprocessing
from vocabularyAnalysis import vocabularyAnalysis
from BasicTextClassificationModel import BasicTextClassificationModel
from nlp_tools import *
from sklearn.model_selection import train_test_split

print('step 0: loading data')
# Retreive data
current_dir = os.getcwd()
target_path = os.path.join(current_dir, "data", "AFDpresidentutf8", "corpus.tache1.learn.utf8.txt")
alltxts,alllabs = load_pres(target_path)

alltxts_train, alltxts_test, y_train, y_test = train_test_split(alltxts, alllabs, test_size=0.20 random_state=32, stratify=alllabs)
alltxts, alllabs = alltxts_train, y_train

# Print split
print("Train/Valid size:", len(alltxts))
print("Test size:", len(alltxts_test))

print('step 1: processing data')
# Preprocess text
textpreprocessor = textPreprocessing()
# Stemming
alltxts_clean1 = [textpreprocessor.preprocess_text(text, stemming=True, stop_words=True,
                                                   languages=['french']) for text in alltxts]
# Lemmatize
alltxts_clean2 = [textpreprocessor.preprocess_text(text, lemmatize=True, stop_words=True,
                                                   languages=['french']) for text in alltxts]

# Save different processes with its titles in lists
titles = ['No process', 'stem', 'lemma']
processed_txts = [alltxts, alltxts_clean1, alltxts_clean2]

# Path to save results
figures_path = os.path.join(current_dir, "results", "president", "figures")
results_path = os.path.join(current_dir, "results", "president")

# Create folder if they don't exist
os.makedirs(results_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

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

analysis_pres = BasicTextClassificationModel(processed_txts, alllabs, titles, figures_folder=figures_path, result_folder=results_path)

print('step 3.1: tuning classification models - count vectorizer')

# defeault parameter results count vectorizer under & over
if os.path.exists(os.path.join(results_path, "results_basic_count_over.txt")):
    basic_results_under = np.genfromtxt(os.path.join(results_path, "results_basic_count_under.txt"), delimiter=",",
                                        dtype=float, filling_values=np.nan)
    basic_results_over = np.genfromtxt(os.path.join(results_path, "results_basic_count_over.txt"), delimiter=",",
                                        dtype=float, filling_values=np.nan)
else:
    basic_results_under = analysis_pres.basic_classification(undersampling=True)
    basic_results_over = analysis_pres.basic_classification(oversampling=True)

# Plot results default
analysis_pres.plot_basic_results([basic_results_under, basic_results_over], ["Undersampling", "Oversampling"], title_figure='countvectorizer')

# Get best sampling initial results
undersampling_cv, oversampling_cv = best_oversampler_undersampler(basic_results_under, basic_results_over)

# Model results with max_features tuning
if os.path.exists(os.path.join(results_path, "results_count_max_features.txt")):
    tune_maxf = np.loadtxt(os.path.join(results_path, "results_count_max_features.txt"), delimiter=",")
else:
    tune_maxf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(1000,10000,1000), parameter_name='max_features',
                                                            undersampling=oversampling_cv, oversampling=oversampling_cv)

# Model results with max_df tuning
if os.path.exists(os.path.join(results_path, "results_count_max_df.txt")):
    tune_max_df = np.loadtxt(os.path.join(results_path, "results_count_max_df.txt"), delimiter=",")
else:
    tune_max_df = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.1), parameter_name='max_df',
                                                            undersampling=oversampling_cv, oversampling=oversampling_cv)

# Model results with min_df tuning
if os.path.exists(os.path.join(results_path, "results_count_min_df.txt")):
    tune_min_df = np.loadtxt(os.path.join(results_path, "results_count_min_df.txt"), delimiter=",")
else:
    tune_min_df = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.01,0.08,0.01), parameter_name='min_df',
                                                            undersampling=oversampling_cv, oversampling=oversampling_cv)

# Plot grid search results
analysis_pres.plot_results_tuning([tune_maxf, tune_max_df, tune_min_df], ['max_features', 'max_df', 'min_df'], figure_name='count')

# Save best final model for countvectorizer
best_model_cv, process_cv = best_model(tune_maxf, tune_max_df, tune_min_df)

print('step 3.2: tuning classification models - tfidf vectorizer')
# defeault parameter results tfidf vectorizer
if os.path.exists(os.path.join(results_path, "results_basic_tfidf_over.txt")):
    basic_tfidf_under = np.genfromtxt(os.path.join(results_path, "results_basic_tfidf_under.txt"), delimiter=",",
                                        dtype=float, filling_values=np.nan)
    basic_tfidf_over = np.genfromtxt(os.path.join(results_path, "results_basic_tfidf_over.txt"), delimiter=",",
                                        dtype=float, filling_values=np.nan)
else:    
    basic_tfidf_under = analysis_pres.basic_classification(vectorizer_type='tfidf', undersampling=True)
    basic_tfidf_over = analysis_pres.basic_classification(vectorizer_type='tfidf', oversampling=True)

analysis_pres.plot_basic_results([basic_tfidf_under, basic_tfidf_over], ["Undersampling", "Oversampling"],
                                 title_figure='tfidf')
undersampling_tf, oversampling_tf = best_oversampler_undersampler(basic_tfidf_under, basic_tfidf_over)

# Evaluate tfidf parameters for best initial results
tfidf_params = analysis_pres.evaluate_tfidf_variations()

# Model results with max_features tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_max_features.txt")):
    maxf_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_max_features.txt"), delimiter=",")
else:
    maxf_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(1000,10000,1000), vectorizer_type='tfidf',
                                                                parameters_tf_idf=tfidf_params, parameter_name='max_features',
                                                                undersampling=undersampling_tf, oversampling=oversampling_tf)
                                                               
# Model results with max_df tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_max_df.txt")):
    max_df_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_max_df.txt"), delimiter=",")
else:
    max_df_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.1), vectorizer_type='tfidf',
                                                                  parameters_tf_idf=tfidf_params, parameter_name='max_df',
                                                                  undersampling=undersampling_tf, oversampling=oversampling_tf)
                                                                  
# Model results with min_df tuning - tfidf
if os.path.exists(os.path.join(results_path, "results_tfidf_min_df.txt")):
    min_df_tfidf = np.loadtxt(os.path.join(results_path, "results_tfidf_min_df.txt"), delimiter=",")
else:
    min_df_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.01,0.08,0.01), vectorizer_type='tfidf',
                                                                    parameters_tf_idf=tfidf_params, parameter_name='min_df',
                                                                    undersampling=undersampling_tf, oversampling=oversampling_tf)

# Plot results tuning                                                                    
analysis_pres.plot_results_tuning([maxf_tfidf, max_df_tfidf, min_df_tfidf],
                                   ['max_features', 'max_df', 'min_df'], figure_name='tf_idf')

# Retreive model and text process which give best results
best_model_tfidf, process_tfidf = best_model(maxf_tfidf, max_df_tfidf, min_df_tfidf)

print('step 3.3: tuning classification models - n-grams')
# Results with default tfidf 
basic_results_1_2_under = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,2), undersampling=True)
basic_results_1_2_over = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,2), oversampling=True)
basic_results_1_3_under = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,3), undersampling=True)
basic_results_1_3_over = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,3), oversampling=True)

# Plot Results
n_grams_basic_results  = [basic_results_1_2_under, basic_results_1_2_over, basic_results_1_3_under, basic_results_1_3_over]
titles_n_grams  = ["(1,2) - under ", "(1,2) - over", "(1,3) - under", "(1,3) - over"]
analysis_pres.plot_basic_results(n_grams_basic_results, titles_n_grams, title_figure='ngrams')

# Get best performing n-gram from basic result
ngram_opt, undersampling_n , oversampling_n = get_best_ngram(n_grams_basic_results, titles_n_grams, sampling=True)
n1, n2 = ngram_opt

# Evaluate tfidf parameters for best initial results
tfidf_params = analysis_pres.evaluate_tfidf_variations(undersampling=undersampling_n, oversampling=oversampling_n)

# Model results with max_features tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_max_features_ngram{n1}-{n2}.txt")):
    maxf_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_max_features_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    maxf_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(1000,10000,1000),
                                                               n_gram_range=ngram_opt, parameters_tf_idf=tfidf_params,
                                                               parameter_name='max_features',
                                                             undersampling=undersampling_n, oversampling=oversampling_n)
# Model results with max_df tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_max_df_ngram{n1}-{n2}.txt")):
    max_df_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_max_df_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    max_df_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.2,0.9,0.1),
                                                                  n_gram_range=ngram_opt, parameters_tf_idf=tfidf_params,
                                                                  parameter_name='max_df',
                                                                  undersampling=undersampling_n, oversampling=oversampling_n)

# Model results with min_df tuning - ngram
if os.path.exists(os.path.join(results_path, f"results_tfidf_min_df_ngram{n1}-{n2}.txt")):
    min_df_ngram = np.loadtxt(os.path.join(results_path, f"results_tfidf_min_df_ngram{n1}-{n2}.txt"), delimiter=",")
else:
    min_df_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.01,0.08,0.01),
                                                                   n_gram_range=ngram_opt, parameters_tf_idf=tfidf_params,
                                                                   parameter_name='min_df',
                                                                   undersampling=undersampling_n, oversampling=oversampling_n)

# Plot tuning results
analysis_pres.plot_results_tuning([maxf_ngram, max_df_ngram, min_df_ngram], ['max_features', 'max_df', 'min_df'],
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
X_test_texts = process_txts_pres(alltxts_test, process_cv)
metrics_cv, conf_cv  = final_model_evaluation(int(max_features_cv), max_df_cv, min_df_cv,
                                                 best_model_cv, "count", (1,1),
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test,
                                                 undersampling=undersampling_cv, oversampling=oversampling_cv)

plot_confusion_matrix(conf_cv, np.unique(y_test), "confusion_cv.png", figures_path)

# final model evaluation tfidf
X_train_texts = processed_txts[process_tfidf]
X_test_texts = process_txts_pres(alltxts_test, process_tfidf)
metrics_tfidf, conf_ctfidf  = final_model_evaluation(int(max_features_tfidf), max_df_tfidf, min_df_tfidf,
                                                 best_model_tfidf, "tfidf", (1,1),
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test,
                                                 undersampling=undersampling_tf, oversampling=oversampling_tf)

plot_confusion_matrix(conf_ctfidf, np.unique(y_test), "confusion_tfidf.png", figures_path)

# final model evaluation n-gram
X_train_texts = processed_txts[process_ngram]
X_test_texts = process_txts_pres(alltxts_test, process_ngram)
metrics_ngram, conf_ngram  = final_model_evaluation(int(max_features_ngram), max_df_ngram, min_df_ngram,
                                                 best_model_ngram, "tfidf", ngram_opt,
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test,
                                                 undersampling=undersampling_n, oversampling=oversampling_n)

plot_confusion_matrix(conf_ngram, np.unique(y_test), "confusion_ngram.png", figures_path)

# Append final results
metrics_cv = np.append(metrics_cv, [best_model_cv, process_cv, max_features_cv, max_df_cv,
                                    min_df_cv,'(1,1)', undersampling_cv, oversampling_cv, 'count'])
metrics_tfidf = np.append(metrics_tfidf, [best_model_tfidf, process_tfidf, max_features_tfidf, max_df_tfidf,
                                          min_df_tfidf, '(1,1)', undersampling_tf, oversampling_tf, 'tfidf'])
metrics_ngram = np.append(metrics_ngram, [best_model_ngram, process_ngram, max_features_ngram, max_df_ngram,
                                          min_df_ngram, str(ngram_opt), undersampling_n, oversampling_n, 'ngram'])
final_results = np.vstack([metrics_cv, metrics_tfidf, metrics_ngram])

# Save final model results
np.savetxt(os.path.join(results_path, f"final_results_model.txt"), final_results, fmt='%s', delimiter=",")
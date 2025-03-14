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

# alltxts,alllabs = select_subset(alltxts,alllabs)

alltxts_train, alltxts_test, y_train, y_test = train_test_split(alltxts, alllabs, test_size=0.2, random_state=32, stratify=alllabs)
alltxts, alllabs = alltxts_train, y_train

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

figures_path = os.path.join(current_dir, "results", "president", "figures")
results_path = os.path.join(current_dir, "results", "president")

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
# Basic results
basic_results_under = analysis_pres.basic_classification(undersampling=True)
basic_results_over = analysis_pres.basic_classification(oversampling=True)
analysis_pres.plot_basic_results([basic_results_under, basic_results_over], ["Undersampling", "Oversampling"], title_figure='countvectorizer')
undersampling_cv, oversampling_cv, basic_results = best_oversampler_undersampler(basic_results_under, basic_results_over)

# Model results with Parameter Tuning of countVector and save plot
result_mf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(4000,35000,300), parameter_name='max_features',
                                                        undersampling=oversampling_cv, oversampling=oversampling_cv)
result_m_df = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.015), parameter_name='max_df',
                                                        undersampling=oversampling_cv, oversampling=oversampling_cv)
result_min_df = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.002,0.08,0.002), parameter_name='min_df',
                                                        undersampling=oversampling_cv, oversampling=oversampling_cv)
analysis_pres.plot_results_tuning([result_mf, result_m_df, result_min_df], ['max_features', 'max_df', 'min_df'], figure_name='count')

# Acquire mean scores of every model for every process using cross validation
nb_scores_mf, lr_scores_mf, svm_scores_mf = analysis_pres.run_cv_opt_param(result_mf, parameter='max_features', cfold=7,
                                                                           undersampling=oversampling_cv, oversampling=oversampling_cv)
nb_scores_max_df, lr_scores_max_df, svm_scores_max_df = analysis_pres.run_cv_opt_param(result_m_df, parameter='max_df', cfold=7,
                                                                                       undersampling=oversampling_cv, oversampling=oversampling_cv)
nb_scores_min_df, lr_scores_min_df, svm_scores_min_df = analysis_pres.run_cv_opt_param(result_min_df, parameter='min_df', cfold=7,
                                                                                       undersampling=oversampling_cv, oversampling=oversampling_cv)

# Define models and corresponding score lists for plotting
parameters = ['min_df', 'max_df', 'max_features']
nb_scores = [nb_scores_min_df, nb_scores_max_df, nb_scores_mf]
lr_scores = [lr_scores_min_df, lr_scores_max_df, lr_scores_mf]
svm_scores = [svm_scores_min_df, svm_scores_max_df, svm_scores_mf]
analysis_pres.plot_results_cv(basic_results, nb_scores, lr_scores, svm_scores, parameters, figure_name='count')

# Save best final model for countvectorizer
best_model_cv, process_cv = optimal_model_CV(nb_scores, lr_scores, svm_scores)

print('step 3.2: tuning classification models - tfidf vectorizer')
# Results with tfidf vectorizer (all default values)
basic_tfidf_under = analysis_pres.basic_classification(vectorizer_type='tfidf', undersampling=True)
basic_tfidf_over = analysis_pres.basic_classification(vectorizer_type='tfidf', oversampling=True)

analysis_pres.plot_basic_results([basic_tfidf_under, basic_tfidf_over], ["Undersampling", "Oversampling"],
                                 title_figure='tfidf')
undersampling_tf, oversampling_tf, basic_results_tfidf = best_oversampler_undersampler(basic_tfidf_under, basic_tfidf_over)

analysis_pres.plot_tfidf_variations_results()

# Model results with Parameter Tuning of TFIDF and save plot
result_mf_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(4000,35000,300), vectorizer_type='tfidf',
                                                                parameters_tf_idf=(True, True, True), parameter_name='max_features',
                                                                undersampling=undersampling_tf, oversampling=oversampling_tf)
                                                               

result_m_df_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.2,0.9,0.015), vectorizer_type='tfidf',
                                                                  parameters_tf_idf=(True, True, True), parameter_name='max_df',
                                                                  undersampling=undersampling_tf, oversampling=oversampling_tf)
                                                                  

result_min_df_tfidf = analysis_pres.vectorizer_parameter_analysis(parameters_loop=(0.002,0.08,0.002), vectorizer_type='tfidf',
                                                                    parameters_tf_idf=(True, True, True), parameter_name='min_df',
                                                                    undersampling=undersampling_tf, oversampling=oversampling_tf)
                                                                    

analysis_pres.plot_results_tuning([result_mf_tfidf, result_m_df_tfidf, result_min_df_tfidf],
                                   ['max_features', 'max_df', 'min_df'], figure_name='tf_idf')

# Acquire scores of every model for every process using cross validation
nb_scores_mf, lr_scores_mf, svm_scores_mf = analysis_pres.run_cv_opt_param(result_mf_tfidf, vectorizer_type='tfidf', cfold=7,
                                                                            parameter='max_features',
                                                                            undersampling=undersampling_tf, oversampling=oversampling_tf)
nb_scores_max_df, lr_scores_max_df, svm_scores_max_df = analysis_pres.run_cv_opt_param(result_m_df_tfidf, vectorizer_type='tfidf', cfold=7,
                                                                                        parameter='max_df',
                                                                                        undersampling=undersampling_tf, oversampling=oversampling_tf)
nb_scores_min_df, lr_scores_min_df, svm_scores_min_df = analysis_pres.run_cv_opt_param(result_min_df_tfidf, vectorizer_type='tfidf', cfold=7,
                                                                                        parameter='min_df',
                                                                                        undersampling=undersampling_tf, oversampling=oversampling_tf)

# Define models and corresponding score lists for plotting
parameters = ['min_df', 'max_df', 'max_features']
nb_scores = [nb_scores_min_df, nb_scores_max_df, nb_scores_mf]
lr_scores = [lr_scores_min_df, lr_scores_max_df, lr_scores_mf]
svm_scores = [svm_scores_min_df, svm_scores_max_df, svm_scores_mf]
analysis_pres.plot_results_cv(basic_results, nb_scores, lr_scores, svm_scores, parameters, figure_name='tfidf')

# Save best final model for tfidf
best_model_tfidf, process_tfidf = optimal_model_CV(nb_scores, lr_scores, svm_scores)

print('step 3.3: tuning classification models - n-grams')
# Results with tfidf vectorizer (all default values)
basic_results_1_2_under = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,2), undersampling=True)
basic_results_1_2_over = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,2), oversampling=True)
basic_results_1_3_under = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,2), undersampling=True)
basic_results_1_3_over = analysis_pres.basic_classification(vectorizer_type='tfidf', n_gram_range=(1,3), oversampling=True)
# Plot Results
n_grams_basic_results  = [basic_results_1_2_under, basic_results_1_2_over, basic_results_1_3_under, basic_results_1_3_over]
titles_n_grams  = ["(1,2) - under ", "(1,2) - over", "(1,3) - under", "(1,3) - over"]
analysis_pres.plot_basic_results(n_grams_basic_results, titles_n_grams, title_figure='ngrams')
# Get best performing n-gram from basic result
ngram_opt, undersampling_n , oversampling_n = get_best_ngram(n_grams_basic_results, titles_n_grams, sampling=True)

# Model results with Parameter Tuning of N-grams and save plot
result_mf_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(6000,40000,500),
                                                               n_gram_range=ngram_opt, parameters_tf_idf=(True, True, True),
                                                               parameter_name='max_features',
                                                               undersampling=undersampling_n, oversampling=oversampling_n)
result_m_df_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.2,0.9,0.015),
                                                                  n_gram_range=ngram_opt, parameters_tf_idf=(True, True, True),
                                                                  parameter_name='max_df',
                                                                  undersampling=undersampling_n, oversampling=oversampling_n)
result_min_df_ngram = analysis_pres.vectorizer_parameter_analysis(vectorizer_type='tfidf', parameters_loop=(0.002,0.08,0.002),
                                                                   n_gram_range=ngram_opt, parameters_tf_idf=(True, True, True),
                                                                   parameter_name='min_df',
                                                                   undersampling=undersampling_n, oversampling=oversampling_n)

analysis_pres.plot_results_tuning([result_mf_ngram, result_m_df_ngram, result_min_df_ngram], ['max_features', 'max_df', 'min_df'],
                                   figure_name='ngram')



# Acquire scores of every model for every process using cross validation
nb_scores_mf, lr_scores_mf, svm_scores_mf = analysis_pres.run_cv_opt_param(result_mf_ngram, vectorizer_type='tfidf', cfold=7,
                                                                            parameter='max_features', n_grame_range=ngram_opt,
                                                                            undersampling=undersampling_n, oversampling=oversampling_n)
nb_scores_max_df, lr_scores_max_df, svm_scores_max_df = analysis_pres.run_cv_opt_param(result_m_df_ngram, vectorizer_type='tfidf', cfold=7,
                                                                                     n_grame_range=ngram_opt, parameter='max_df',
                                                                                     undersampling=undersampling_n,
                                                                                     oversampling=oversampling_n)
nb_scores_min_df, lr_scores_min_df, svm_scores_min_df = analysis_pres.run_cv_opt_param(result_min_df_ngram, vectorizer_type='tfidf', cfold=7,
                                                                                     n_grame_range=ngram_opt, parameter='min_df',
                                                                                     undersampling=undersampling_n,
                                                                                     oversampling=oversampling_n)

# Define models and corresponding score lists
parameters = ['min_df', 'max_df', 'max_features']
nb_scores = [nb_scores_min_df, nb_scores_max_df, nb_scores_mf]
lr_scores = [lr_scores_min_df, lr_scores_max_df, lr_scores_mf]
svm_scores = [svm_scores_min_df, svm_scores_max_df, svm_scores_mf]
analysis_pres.plot_results_cv(basic_results, nb_scores, lr_scores, svm_scores, parameters, figure_name='ngram')

# Save best final model for N-gram
best_model_ngram, process_ngram = optimal_model_CV(nb_scores, lr_scores, svm_scores)

print('step 4: Obtained tuned parameter for best peforming model and process')

max_features_cv, max_df_cv, min_df_cv = retreive_tuned_param(process_cv, best_model_cv,
                                                                result_mf, result_m_df, result_min_df)

max_features_tfidf, max_df_tfidf, min_df_tfidf = retreive_tuned_param(process_tfidf, best_model_tfidf,
                                                                result_mf_tfidf, result_m_df_tfidf, result_min_df_tfidf)

max_features_ngram, max_df_ngram, min_df_ngram = retreive_tuned_param(process_ngram, best_model_ngram,
                                                                result_mf_ngram, result_m_df_ngram, result_min_df_ngram)


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

# final model evaluation tfidf
X_train_texts = processed_txts[process_tfidf]
X_test_texts = process_txts_pres(alltxts_test, process_tfidf)
metrics_tfidf, conf_ctfidf  = final_model_evaluation(int(max_features_tfidf), max_df_tfidf, min_df_tfidf,
                                                 best_model_tfidf, "tfidf", (1,1),
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test,
                                                 undersampling=undersampling_tf, oversampling=oversampling_tf)

# final model evaluation n-gram
X_train_texts = processed_txts[process_ngram]
X_test_texts = process_txts_pres(alltxts_test, process_ngram)
metrics_ngram, conf_ngram  = final_model_evaluation(int(max_features_ngram), max_df_ngram, min_df_ngram,
                                                 best_model_ngram, "tfidf", ngram_opt,
                                                 X_train_texts, X_test_texts,
                                                 alllabs, y_test,
                                                 undersampling=undersampling_n, oversampling=oversampling_n)

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
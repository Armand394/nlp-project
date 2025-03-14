from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter

# Apply global settings
plt.rcParams.update({
    'axes.spines.right': False,   # Enable right spine (solid)
    'axes.spines.top': False,     # Enable top spine (solid)
    'axes.grid': True,           # Enable grid
    'grid.alpha': 0.4,           # Make the grid transparent (adjust alpha)
    'xtick.direction': 'out',     # Tickmarks on x-axis (inside)
    'ytick.direction': 'out',     # Tickmarks on y-axis (inside)
    'grid.linestyle': '--',      # Dashed grid (can be changed)
    'axes.edgecolor': 'black',   # Ensure spines are visible
    'axes.linewidth': 1.2,        # Make spines slightly thicker
    'axes.labelsize': 11
})

# Ignore ConvergenceWarning from LinearSVC
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class BasicTextClassificationModel:

    def __init__(self, processed_txts, alllabs, titles_prcs, result_folder=None, figures_folder=None):
        self.processed_txts = processed_txts
        self.alllabs = alllabs
        self.titles_prcs = titles_prcs
        self.result_folder = result_folder
        self.figures_folder = figures_folder

    def accuracy_models(self, X, vectorizer=CountVectorizer(), rs=42, t_lr=1e-8, C_lr=100.00, t_svc=1e-8,
                        undersampling=False, oversampling=False):

        # Classifier
        nb_clf = MultinomialNB()
        lr_clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=500, tol=t_lr, C=C_lr)
        svm_clf = LinearSVC(random_state=0)

        # Perform 5-Fold Cross-Validation using bow_cross_validation
        scores_nb = self.bow_cross_validation(X, nb_clf, cfold=5, vectorizer=vectorizer,
                                              undersampling=undersampling, oversampling=oversampling)
        scores_lr = self.bow_cross_validation(X, lr_clf, cfold=5, vectorizer=vectorizer,
                                              undersampling=undersampling, oversampling=oversampling)
        scores_svm = self.bow_cross_validation(X, svm_clf, cfold=5, vectorizer=vectorizer,
                                               undersampling=undersampling, oversampling=oversampling)
        
        # Mean Scores of cross-val
        mean_score_nb = np.mean(scores_nb)
        mean_score_lr = np.mean(scores_lr)
        mean_score_svm = np.mean(scores_svm)

        # Return Accuracy scores each respective models
        return mean_score_nb, mean_score_lr, mean_score_svm

    def basic_classification(self, vectorizer_type='count', n_gram_range=(1,1), undersampling=False,
                             oversampling=False, verbose=0, save_plot=False):
        
        # Decide which vectorizer class to use
        if vectorizer_type.lower() == 'count':
            VectorizerClass = CountVectorizer
        elif vectorizer_type.lower() == 'tfidf':
            VectorizerClass = TfidfVectorizer

        else:
            raise ValueError("Invalid vectorizer_type. Choose 'count' or 'tfidf'.")
        
        results = []

        for i, txts in enumerate(self.processed_txts):
            vectorizer = VectorizerClass(ngram_range=n_gram_range)
            score_nb, score_lr, score_svm = self.accuracy_models(txts, vectorizer=vectorizer,
                                                                  undersampling=undersampling, oversampling=oversampling)
            results.append([None, score_nb, score_lr, score_svm, i])  # No parameter

        filename = self.produce_file_name(vectorizer_type, None, n_gram_range, True, undersampling, oversampling)

        # Save results to file
        self.save_results(filename, results)

        if verbose == 1:
            print(f'Finished basic classification for vectorizer={vectorizer_type} for model accuracy')
            print()

        return np.array(results)
    
    def vectorizer_parameter_analysis(self, vectorizer_type='count', n_gram_range=(1,1), parameters_loop=None, parameter_name=None,
                                       parameters_tf_idf=None, undersampling=False, oversampling=False, verbose=0):
        
        # Decide which vectorizer class to use
        if vectorizer_type.lower() == 'count':
            VectorizerClass = CountVectorizer
        elif vectorizer_type.lower() == 'tfidf':
            VectorizerClass = TfidfVectorizer
        else:
            raise ValueError("Invalid vectorizer_type. Choose 'count' or 'tfidf'.")

        results = []

        if verbose == 1:
            print(f'Start tuning parameter={parameter_name} for model accuracy...')

        # Loop parameters for getting tuning results
        min_loop, max_loop, step_loop = parameters_loop

        for i, txts in enumerate(self.processed_txts):
            for parameter in np.arange(min_loop, max_loop, step_loop):
                # Common parameters
                common_params = {
                    parameter_name: int(parameter) if parameter_name == "max_features" else parameter,
                    "ngram_range": n_gram_range
                }
                
                if parameter_name in ['max_features', 'min_df', 'max_df']:
                    common_params[parameter_name] = parameter

                # TF-IDF–specific parameters
                tfidf_params = {}
                if vectorizer_type == 'tfidf' and parameters_tf_idf is not None:
                    use_idf, smooth_idf, sublinear_tf = parameters_tf_idf
                    
                    tfidf_params['use_idf'] = use_idf
                    tfidf_params['smooth_idf'] = smooth_idf
                    tfidf_params['sublinear_tf'] = sublinear_tf

                # Merge into a single param dict
                param_dict = {**common_params, **tfidf_params}

                # Instantiate the vectorizer
                vectorizer = VectorizerClass(**param_dict)

                # Fit for given countVectorizer object and compute results
                score_nb, score_lr, score_svm = self.accuracy_models(txts, vectorizer=vectorizer,
                                                                    undersampling=undersampling, oversampling=oversampling)
                results.append([parameter, score_nb, score_lr, score_svm, i])

        filename = self.produce_file_name(vectorizer_type, parameter_name, n_gram_range)

        # Save results to file
        self.save_results(filename, results)

        if verbose == 1:
            print(f'Finished tuning parameter={parameter_name} for vectorizer={vectorizer_type} for model accuracy')
            print()

        return np.array(results)

    def evaluate_tfidf_variations(self, undersampling=False, oversampling=False):
        # Define baseline TF-IDF parameters.
        baseline_params = {"use_idf": False, "smooth_idf": False, "sublinear_tf": False}
        
        # Define the three variations.
        variation1 = {"use_idf": True,  "smooth_idf": False, "sublinear_tf": False}
        variation2 = {"use_idf": True,  "smooth_idf": True,  "sublinear_tf": False}
        variation3 = {"use_idf": True,  "smooth_idf": True,  "sublinear_tf": True}
        variations = [baseline_params, variation1, variation2, variation3]
        variation_labels = ["( )",
                            "(use_idf)", 
                            "(use_idf, smooth_idf)", 
                            "(use_idf, smooth_idf, sublinear_tf)"]
        
        # Evaluate performance for each variation.
        var_svm_scores = []
        for params in variations:
            svm_scores = []
            for txt in self.processed_txts:
                vectorizer = TfidfVectorizer(**params)
                _, _, acc_svm = self.accuracy_models(txt, vectorizer=vectorizer, oversampling=oversampling, undersampling=undersampling)
                svm_scores.append(acc_svm)

            var_svm_scores.append(svm_scores)
        
        self.plot_tfidf_variations_results(var_svm_scores, variation_labels)
        
        # transform in np array
        var_svm_scores = np.array(var_svm_scores)
        
        # Retreive best tfidf variation
        average_score_var = np.mean(var_svm_scores, axis=1)
        opt_var = np.argmax(average_score_var)

        return tuple(variations[opt_var].values())

    def build_vectorizers(self, parameter, opt_params, vectorizer_type='count', n_gram_range=(1,1), use_max_feature=False, max_feature_value=8000,
                        use_idf=True, smooth_idf=True, sublinear_tf=True):

        def build_vectorizer(opt_value):
            # Common kwargs for Count or Tfidf
            kwargs = {"ngram_range": n_gram_range}

            # Set the tuned parameter if present
            if parameter == "max_features":
                if opt_value is not None:
                    kwargs["max_features"] = int(opt_value)
            elif parameter == "max_df":
                # param could be float or int
                kwargs["max_df"] = opt_value
                if use_max_feature:
                    kwargs["max_features"] = max_feature_value
            elif parameter == "min_df":
                if opt_value > 1.0:
                    kwargs["min_df"] = int(opt_value)
                else:
                    kwargs["min_df"] = opt_value

                if use_max_feature:
                    kwargs["max_features"] = max_feature_value

            # If we're using TfidfVectorizer, add TF–IDF–specific parameters
            if vectorizer_type == 'tfidf':
                kwargs["use_idf"] = use_idf
                kwargs["smooth_idf"] = smooth_idf
                kwargs["sublinear_tf"] = sublinear_tf

                return TfidfVectorizer(**kwargs)
            else:
                # Default is CountVectorizer
                return CountVectorizer(**kwargs)

        # Build vectorizers for each model using the provided opt_params dict
        vectorizer_nb = build_vectorizer(opt_params.get("nb"))
        vectorizer_lr = build_vectorizer(opt_params.get("lr"))
        vectorizer_svm = build_vectorizer(opt_params.get("svm"))

        return vectorizer_nb, vectorizer_lr, vectorizer_svm

    def balance_dataset(self, X, y, undersampling=False, oversampling=False, strategy=0.8, random_state=42):
        
        if oversampling and undersampling:
            # Apply SMOTE + Tomek Links (Hybrid method)
            sampler = SMOTETomek(sampling_strategy=strategy, random_state=random_state)
        elif oversampling:
            # Apply SMOTE (Oversampling)
            sampler = SMOTE(sampling_strategy=strategy, random_state=random_state)
        elif undersampling:
            # Apply Random Undersampling
            sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
        else:
            return X, y

        # Apply chosen resampling method
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        return X_resampled, y_resampled

    def bow_cross_validation(self, txts, clf, cfold=5, vectorizer=CountVectorizer(), undersampling=False, oversampling=False):

        kf = KFold(n_splits=cfold, shuffle=True, random_state=42)
        scores = []
    
        for train_idx, test_idx in kf.split(txts):
            # Split raw texts and labels
            X_train_texts = [txts[i] for i in train_idx]
            X_test_texts  = [txts[i] for i in test_idx]
            y_train = np.array(self.alllabs)[train_idx]
            y_test  = np.array(self.alllabs)[test_idx]
            
            # Vectorizers
            X_train = vectorizer.fit_transform(X_train_texts)
            X_test = vectorizer.transform(X_test_texts)

            # Sampling
            X_train, y_train = self.balance_dataset(X_train, y_train, undersampling, oversampling)

            # Fit classifier on training data.
            clf.fit(X_train, y_train)

            # Predict on test set.
            y_pred = clf.predict(X_test)
            
            # Compute score.
            if undersampling or oversampling:
                score = f1_score(y_test, y_pred, average='weighted')
            else:
                score = accuracy_score(y_test, y_pred)
            
            scores.append(score)
        
        return scores

    def plot_results_tuning(self, results_list, parameter_names, figure_name='count'):
    
        scores = ['NB', 'LR', 'SVM']
        colors = ['darkturquoise', 'limegreen','firebrick']
        k = len(results_list)         # Number of tuned parameter sets (rows)
        n_titles = len(self.titles_prcs)        # Number of processed texts (columns)
        
        if len(parameter_names) != k:
            raise ValueError("The length of parameter_names must equal the number of result arrays in results_list.")
        
        # Create a grid of subplots: k rows (one per tuned parameter set) and n_titles columns.
        fig, axes = plt.subplots(k, n_titles, figsize=(5 * n_titles, 4 * k), squeeze=False)
        
        # Loop through each result array (each row)
        for row_idx, result in enumerate(results_list):
            # Loop through each processed text (each column)
            for col_idx in range(n_titles):
                ax = axes[row_idx, col_idx]
                
                # Filter data for the current text process (text index in column 4)
                process_data = result[result[:, 4] == col_idx]
                if process_data.size == 0:
                    # In case no data exists for this process, set labels and continue.
                    ax.set_title(f"{self.titles_prcs[col_idx]}")
                    ax.set_xlabel(parameter_names[row_idx])
                    ax.set_ylabel("Score")
                    continue
                
                # Extract parameter values (x-axis) and score data (columns 1-3)
                param_values = process_data[:, 0]
                score_data = process_data[:, 1:4]
                
                # Plot each score series
                for s in range(score_data.shape[1]):
                    ax.plot(param_values, score_data[:, s], label=scores[s], color=colors[s])
                    
                    # Find and annotate the maximum value for the current score series
                    max_idx = np.argmax(score_data[:, s])
                    max_x = param_values[max_idx]
                    max_y = score_data[max_idx, s]
                    
                    # Plot the maximum point and annotate it
                    ax.scatter(max_x, max_y, color='black', zorder=2)
                    ax.text(max_x, max_y, f"{max_y:.2f}", fontsize=8,
                            color='black', ha='right', va='bottom')
                
                # Customize the subplot: title, axis labels, legend, and grid
                ax.set_title(f"{self.titles_prcs[col_idx]}", fontsize=12)
                ax.set_xlabel(parameter_names[row_idx], fontsize=10)
                ax.set_ylabel("Score", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True)
        
        # Adjust layout and display the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"{figure_name}_vectorizer_tuning.jpeg"))
        plt.close()

    def produce_file_name(self, vectorizer_type, parameter_name, n_gram_range, basic=False, undersampling=False, oversampling=False):
        
        if basic == False:
            if n_gram_range != (1,1):
                return f"results_{vectorizer_type}_{parameter_name}_ngram{n_gram_range[0]}-{n_gram_range[1]}.txt"

            return f"results_{vectorizer_type}_{parameter_name}.txt"
        
        if (n_gram_range != (1,1)) & (undersampling == True):
            return f"results_basic_{vectorizer_type}_under_ngram{n_gram_range[0]}-{n_gram_range[1]}.txt"
        elif (n_gram_range != (1,1)) & (oversampling == True):
            return f"results_basic_{vectorizer_type}_over_ngram{n_gram_range[0]}-{n_gram_range[1]}.txt"
        elif n_gram_range != (1,1):
            return f"results_basic_{vectorizer_type}_ngram{n_gram_range[0]}-{n_gram_range[1]}.txt"
 
        if undersampling == True:
            return f"results_basic_{vectorizer_type}_under.txt"
        elif oversampling == True:
            return f"results_basic_{vectorizer_type}_over.txt"
        
        return f"results_basic_{vectorizer_type}.txt"

    def save_results(self, filename, results):
        # Save results to file
        file_path = os.path.join(self.result_folder, filename)

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(results)

    def plot_tfidf_variations_results(self, var_svm_scores, variation_labels):

        # Plot
        tfidf_var_results = np.array(var_svm_scores)
        
        plt.figure(figsize=(10,6))
        colors = ["tomato", "darkorange", "khaki", "yellowgreen"]
        x = np.arange(tfidf_var_results.shape[1])
        for i in range(tfidf_var_results.shape[0]):
            plt.bar(x + i*0.2, tfidf_var_results[i], width=0.2, label=variation_labels[i], color=colors[i])

        
        plt.xticks(x + 0.2, self.titles_prcs, rotation=45)
        plt.ylim((0.6,1.0))
        plt.legend(loc="upper left")
        plt.title("TF-IDF Variations: SVM")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"tfidf_variation_results.jpeg"))
        plt.close()

    def plot_basic_results(self, results_arrays, titles_list, title_figure='vectorizer'):

        num_plots = len(results_arrays)

        # Define figure and subplots
        fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(8 * num_plots, 6), sharey=True)

        # Ensure axes is iterable (if only one subplot, make it a list)
        if num_plots == 1:
            axes = [axes]

        # Loop through each result array and create a subplot
        for idx, (results_array, title, ax) in enumerate(zip(results_arrays, titles_list, axes)):
            # Extract scores
            nb_scores = results_array[:, 1]
            lr_scores = results_array[:, 2]
            svm_scores = results_array[:, 3]

            # Define x-axis positions
            x = np.arange(len(self.titles_prcs))
            bar_width = 0.25

            # Create bar chart
            ax.bar(x - bar_width, nb_scores, width=bar_width, label='Naïve Bayes', color='darkturquoise', alpha=0.8)
            ax.bar(x, lr_scores, width=bar_width, label='Logistic Regression', color='limegreen', alpha=0.8)
            ax.bar(x + bar_width, svm_scores, width=bar_width, label='SVM', color='firebrick', alpha=0.8)

            # Labels and formatting
            if len(titles_list) > 1:
                ax.set_title(title, fontsize=14)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.titles_prcs, rotation=15, ha="right")
            ax.set_ylim(0.6, 1.0)

            # Add legend only to the first subplot
            if idx == 0:
                ax.set_ylabel("Score", fontsize=12)
                ax.legend()

            ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"basic_result_{title_figure}.jpeg"))
        plt.close()

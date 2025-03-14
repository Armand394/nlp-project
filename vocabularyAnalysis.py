import os
import matplotlib.pyplot as plt
from collections import Counter
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

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

class vocabularyAnalysis():
    
    def __init__(self, processed_txts, titles_txts, figures_folder=None):
        self.processed_txts = processed_txts
        self.titles_txts = titles_txts
        self.figures_folder = figures_folder

    def vocabSize(self):
        
        vocabsize_result = {}
        
        for i, txts in enumerate(self.processed_txts):
            #Taille d'origine du vocabulaire pour les films
            vocab = set()

            for text in txts:
                words = text.split()
                vocab.update(words)

            vocabsize_result[self.titles_txts[i]] = len(vocab)

        # Convert results to lists for plotting
        techniques = list(vocabsize_result.keys())   # Processing technique names
        vocab_sizes = list(vocabsize_result.values())  # Corresponding vocab sizes

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(4 + 0.5 * len(techniques), 4))  # Adjust width dynamically

        ax.bar(techniques, vocab_sizes, color='skyblue', alpha=0.8)

        # Labels and formatting
        ax.set_xlabel("Preprocessing Techniques", fontsize=11)
        ax.set_ylabel("Vocabulary Size", fontsize=11)
        tick_positions = range(len(techniques))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(techniques, rotation=15, ha="right")

        # Add values on top of bars
        for index, value in enumerate(vocab_sizes):
            ax.text(index, value + 0.02 * max(vocab_sizes), str(value), ha='center', fontsize=10)

        # Display the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"vocab_size.jpeg"))
        plt.close()
        return vocabsize_result
    

    def create_wordclouds(self, alllabs=None, top_frequency=100, frequency_function=None, frequency_f_name='freq'):

        word_counter_dictionaries = []

        for txts in self.processed_txts:

            if alllabs == None:
                current_count = frequency_function(txts)
                current_dict = dict(current_count.most_common(top_frequency))
            else:
                current_count = frequency_function(txts, alllabs)
                current_dict = dict(sorted(current_count.items(), key=lambda x: abs(x[1]), reverse=True)[:top_frequency])

            word_counter_dictionaries.append(current_dict)

        self.generate_multiple_wordclouds(word_counter_dictionaries, frequency_f_name)


    def generate_multiple_wordclouds(self, frequencies_list, frequency_f_name):

        if len(frequencies_list) != len(self.titles_txts):
            raise ValueError("The length of frequencies_list and titles must be the same.")
        
        n = len(frequencies_list)  # Number of word clouds
        
        # Set up the plot grid
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))  # Dynamically adjust the figure size
        if n == 1:
            axes = [axes]  # Ensure axes is iterable when there is only one word cloud

        # Generate and plot each word cloud
        for i, (frequencies, title, ax) in enumerate(zip(frequencies_list, self.titles_txts, axes)):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')  # Turn off the axis
            ax.set_title(title, fontsize=14, fontweight='bold')  # Bold title
            
            # Add a border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"wordcloud_{frequency_f_name}.jpeg"))
        plt.close()

    
    def plot_freq_distribution(self, top_frequency=1000, frequency_function=None, log=True):

        plt.figure(figsize=(6, 4))  # Create a new figure with a specified size
        palette = sns.color_palette("husl", len(self.processed_txts))

        # Loop through each text dataset and plot its frequencies
        for idx, (txts, title) in enumerate(zip(self.processed_txts, self.titles_txts)):
            current_wc = frequency_function(txts)  # Calculate word counts
            freq = [f for w, f in current_wc.most_common(top_frequency)]  # Get top frequencies
            plt.plot(freq[:top_frequency],
                    label=title,
                    alpha=0.8,               # Adjust transparency
                    linestyle='--',         # Dashed lines
                    color=palette[idx])      # Distinct color for each line


        # Add legend, labels, and title to the plot
        if log == True:
            plt.yscale('log')  # Apply logarithmic scale to y-axis

        plt.legend()
        plt.xlim(0,top_frequency)
        plt.ylim(0.1)
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.figures_folder, f"frequency_distribution.jpeg"))
        plt.close()

    def freq_distribution_vectorizer(self):
        # Create a figure with 3 subplots in a single row
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Define a list of different linestyles
        linestyles = ['-', '--', '-.', ':']

        # Subplot 1: Varying min_df
        for idx, min_df in enumerate(np.arange(0.02, 0.1, 0.02)):
            linestyle = linestyles[idx % len(linestyles)]
            top_frequencies = self.get_word_freq_CV(min_df=min_df)
            axes[0].plot(top_frequencies, linestyle=linestyle, label=f'min_df={min_df:.2f}')

        axes[0].legend()
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Frequency')

        # Subplot 2: Varying max_df
        for idx, max_df in enumerate(np.arange(0.4, 0.6, 0.05)):
            linestyle = linestyles[idx % len(linestyles)]
            top_frequencies = self.get_word_freq_CV(max_df=max_df)
            axes[1].plot(top_frequencies, linestyle=linestyle, label=f'max_df={max_df:.2f}')

        axes[1].legend()
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Frequency')

        # Subplot 3: Varying max_features
        for idx, max_features in enumerate(range(500, 2000, 250)):
            linestyle = linestyles[idx % len(linestyles)]
            current_m = 2000 - max_features
            top_frequencies = self.get_word_freq_CV(max_features=current_m)
            axes[2].plot(top_frequencies, linestyle=linestyle, label=f'max_features={current_m}')

        axes[2].legend()
        axes[2].set_xlabel('Rank')
        axes[2].set_ylabel('Frequency')

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"frequency_distribution_vectorizer.jpeg"))
        plt.close()


    def get_word_freq_CV(self, top_words=1000, min_df=1, max_df=1.0, max_features=None):
        
        alltxts = self.processed_txts[0]

        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)

        X = vectorizer.fit_transform(alltxts)

        feature_names = vectorizer.get_feature_names_out()

        # Get the word frequencies (sums across all documents)
        word_frequencies = np.asarray(X.sum(axis=0)).flatten()

        # Combine words with their frequencies
        word_freq_dict = dict(zip(feature_names, word_frequencies))

        # Sort the dictionary by frequency (descending)
        sorted_word_freq = dict(sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True))

        return list(sorted_word_freq.values())[:top_words]
    

    def Ngram_vocabSize(self):

        alltxts = self.processed_txts[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sizes = []
        for i in range(1,6):
            ngram_range = (1,i)
            vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer='word')
            X = vectorizer.fit_transform(alltxts)
            dict_size = X.shape[1]
            sizes.append(dict_size)

        sizes2 = []
        for i in range(1,6):
            ngram_range = (i,i)
            vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer='word')
            X = vectorizer.fit_transform(alltxts)
            dict_size = X.shape[1]
            sizes2.append(dict_size)

        axes[0].bar(range(1, len(sizes) + 1), sizes, color='springgreen', edgecolor='black')
        axes[0].set_xlabel('N-gram (1,i)')
        axes[0].set_ylabel('Vocabulary Size')
        axes[0].set_xticks(range(1, len(sizes) + 1))

        axes[1].bar(range(1, len(sizes2) + 1), sizes2, color='mediumseagreen', edgecolor='black')
        axes[1].set_xlabel('N-gram (i,i)')
        axes[1].set_ylabel('Vocabulary Size')
        axes[1].set_xticks(range(1, len(sizes2) + 1))

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"ngram_vocab_size.jpeg"))
        plt.close()


    def plot_label_distribution(self, alllabs):

        # Count occurrences of each label (0 or 1)
        label_counts = Counter(alllabs)
        labels, frequencies = zip(*label_counts.items())  # Unpack labels and their frequencies

        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size
        ax.barh(labels, frequencies, color=['blue', 'red'], alpha=0.7)

        # Labels and formatting
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Label")
        ax.set_yticks(labels)
        ax.set_yticklabels([f"Class {l}" for l in labels])

        # Display value on bars
        for i, v in enumerate(frequencies):
            ax.text(v + 0.02 * max(frequencies), labels[i], str(v), va='center', fontsize=10)

        # Show the plot
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_folder, f"bias_distribution.jpeg"))
        plt.close()
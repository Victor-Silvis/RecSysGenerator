import pandas as pd
import numpy as np
import math
from scipy.stats import ks_2samp, entropy, chi2_contingency
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time


# EVALUATOR SCRIPT
# -----------------------------------------------------------------------------------------------
# This script contains the evaluation logic. This script calculates all the key metrics
# and graphs as described in the literature review of the report. Finally, it takes these
# metrics and graphs and outputs them into a HTML report which by default can be found in 
# the Reports folder
# -----------------------------------------------------------------------------------------------


'''
Contents of Script
    1. DataQuality Evaluator + HTML report generator
    2. SimulationQuality Evaluator + HTML report generator
    3. Item DataQuality Evaluator + HTML report generator
'''

class DataQuality:

    def __init__(self, discrete_columns, continous_columns, folder='Reports', contents_folder='contents', id_field=None):
        self.continous_columns = continous_columns
        self.discrete_columns = discrete_columns
        self.contents_folder = contents_folder
        if id_field is not None:
            self.discrete_columns.remove(id_field)
        self.graphs = []
        self.metrics_discrete = {
            'Category_Entropy' : [],
            'Chi-Sqaure avg P-Value' : [],
        }
        self.metrics_continuous = {
            'Kolmogorov Smirnov' : [],
            'Mean Score' : [],
        }
        self.avg_scores_discrete = {
            'Category_Entropy' : [],
            'Chi-Sqaure avg P-Value' : [],
        }
        self.avg_scores_continuous = {
            'Kolmogorov Smirnov' : [],
            'Mean Score' : [],
        }
        self.count_reports = []
        self.path_to_report(folder)

    def path_to_report(self, folder):
        dir = os.getcwd()
        self.report_folder = os.path.join(dir, folder)

    def calc_scores(self):
        for key, values in self.metrics_discrete.items():
            if values:
                value = np.mean(values)
            else:
                value = 0
            self.avg_scores_discrete[key].append(value)
            self.metrics_discrete[key] = []
        
        for key, values in self.metrics_continuous.items():
            if values:
                value = np.mean(values)
            else:
                value = 0
            self.avg_scores_continuous[key].append(value)
            self.metrics_continuous[key] = []


    def evaluate(self, real_df, fake_df, number=1):
        '''Main Function that can be called during simulation, to evaluate data quality'''
        discrete, continuous = [], []
        self.real_df = real_df
        self.fake_df = fake_df
        self.number = number

        if self.discrete_columns:
            for col in self.discrete_columns:
                real, fake = real_df[col], fake_df[col]
                real = real.reset_index(drop=True)
                fake = fake.reset_index(drop=True)
                coverage = self.Category_Coverage(real, fake)
                adherence = self.Category_Adherence(real, fake)
                entropy = self.Category_Entropy(real, fake)
                dist_match = self.Category_Distribution_Similarity(real, fake)
                mv_similarity = self.Missing_Value_Similarity(real, fake)
                chi_square, p_value = self.chi_square(real, fake)
                discrete.append({'Column': col, 'Chi-Square (p-value)': str(chi_square)+' ('+str(p_value)+')', 'Category Coverage': coverage, 'Category Adherence': adherence,
                                'Category Entropy': entropy, 'Category Distribution Match': dist_match,
                                'Missing Value Similarity': mv_similarity})
        
        if self.continous_columns:
            for col in self.continous_columns:
                real, fake = real_df[col], fake_df[col]
                kl_div = self.KL_Divergence(real, fake)
                ks_stat, p_value = self.KS_Test(real, fake)
                stats = self.Stat_Similarity(real, fake)
                mv_similarity = self.Missing_Value_Similarity(real, fake)
                continuous.append({'Column': col, 'Kullback Leibler': kl_div, 'Kolmogorov Smirnov': (ks_stat, p_value),
                                   'Mean Score': stats[0], 'Median Score': stats[1], 'Std Score': stats[2],
                                   'Missing Value Similarity': mv_similarity})
        
        self.discrete_df = pd.DataFrame(discrete) if discrete else pd.DataFrame({'None': [0]})
        self.continuous_df = pd.DataFrame(continuous) if continuous else pd.DataFrame({'None': [0]})

        self.calc_scores()
        self.generate_contents()
        self.generate_html()
        self.graphs = [] #Reset graphs
        self.discrete_df = None
        self.continuous_df = None


    ### METRICS SECTION ###

    def KL_Divergence(self, real, fake, bins=10):
        hist_real, _ = np.histogram(real, bins=bins, density=True)
        hist_fake, _ = np.histogram(fake, bins=bins, density=True)
        hist_real[hist_real == 0], hist_fake[hist_fake == 0] = 1e-10, 1e-10
        kl_div = np.sum(hist_real * np.log(hist_real / hist_fake))
        kl_div = np.round(kl_div, 3)
        return kl_div

    def KS_Test(self, real, fake):
        ks_stat, p_value = ks_2samp(real, fake)
        self.metrics_continuous['Kolmogorov Smirnov'].append(np.round(ks_stat, 3))
        return np.round(ks_stat, 3), np.round(p_value, 3)

    def Stat_Similarity(self, real, fake):
        stats = ['mean', 'median', 'std']
        scores = []
        for stat in stats:
            real_val, fake_val = getattr(real, stat)(), getattr(fake, stat)()
            score = np.round(1 - (abs(real_val - fake_val) / (real.max() - real.min())), 2)
            scores.append(np.clip(score, 0, 1))
        self.metrics_continuous['Mean Score'].append(scores[0])
        return scores

    def Category_Coverage(self, real, fake):
        unique_real, unique_fake = set(real), set(fake)
        coverage = len(unique_fake.intersection(unique_real)) / len(unique_real)
        return str(np.round(coverage * 100, 1)) + '%'

    def Category_Distribution_Similarity(self, real, fake):
        real_counts = pd.Series(real).value_counts(normalize=True)
        fake_counts = pd.Series(fake).value_counts(normalize=True)
        all_cats = set(real_counts.index).union(set(fake_counts.index))
        diffs = [abs(real_counts.get(cat, 0) - fake_counts.get(cat, 0)) for cat in all_cats]
        diff_norm = np.round(1 - np.mean(diffs), 2)
        return diff_norm

    def Category_Adherence(self, real, fake):
        unique_real = set(real)
        valid_fake = [val for val in fake if val in unique_real]
        return str(np.round(len(valid_fake) / len(fake) * 100, 1)) + '%'

    def Category_Entropy(self, real, fake):
        real_counts = pd.Series(real).value_counts(normalize=True, sort=False)
        fake_counts = pd.Series(fake).value_counts(normalize=True, sort=False)
        all_cats = set(real_counts.index).union(set(fake_counts.index))
        real_probs = np.array([real_counts.get(cat, 0) for cat in all_cats])
        fake_probs = np.array([fake_counts.get(cat, 0) for cat in all_cats])
        ent = np.round(abs(entropy(real_probs) - entropy(fake_probs)), 4)
        if ent < 1:
            self.metrics_discrete['Category_Entropy'].append(ent)
        return ent

    def Missing_Value_Similarity(self, real, fake):
        real_prop, fake_prop = real.isna().sum() / len(real), fake.isna().sum() / len(fake)
        return 1 - (fake_prop - real_prop)
    
    def chi_square(self, real, fake):
        contingency_table = pd.crosstab(real, fake)
        chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
        self.metrics_discrete['Chi-Sqaure avg P-Value'].append(p_value)
        return np.round(chi2_statistic,3), np.round(p_value,3)
    

    ### GRAPHS SECTION ###

    def KDE_graph(self, real, fake, title, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = None

        sns.kdeplot(real, color="#2C3E50", fill=True, alpha=0.9, ax=ax, label='Real Data')
        sns.kdeplot(real, color="#424949", fill=False, linewidth=2, ax=ax)
        sns.kdeplot(fake, color="skyblue", fill=True, alpha=0.5, ax=ax, label='Synthetic Data')
        sns.kdeplot(fake, color="#3498DB", fill=False, linewidth=2, alpha=0.5, ax=ax)

        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        return ax

    def Bar_plot(self, real, fake, title, ax=None):
        unique_categories_real = set(real)
        unique_categories_fake = set(fake)
        category_coverage = len(unique_categories_fake.intersection(unique_categories_real)) / len(unique_categories_real)
        category_counts = {}

        if len(unique_categories_real) < 7:
            category_counts_real = Counter(real)
            category_counts_synthetic = Counter(fake)

            for category in unique_categories_real:
                category_counts[category] = {
                    'real_count': category_counts_real[category],
                    'synthetic_count': category_counts_synthetic[category]
                }

            categories = list(category_counts.keys())
            real_counts = [category_counts[category]['real_count'] for category in categories]
            synthetic_counts = [category_counts[category]['synthetic_count'] for category in categories]

            width = 0.35
            ind = range(len(categories))
            ax.bar(ind, real_counts, width, label='Real Data', color='#2C3E50', alpha=0.9)
            ax.bar([i + width for i in ind], synthetic_counts, width, label='Synthetic Data', color='skyblue', alpha=0.5)

            ax.set_xlabel('Categories')
            ax.set_ylabel('Counts')
            ax.set_title(title)
            ax.set_xticks([i + width / 2 for i in ind])
            ax.set_xticklabels(categories)
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            return ax
        else:
            return None  # Return None if there are too many categories

    def generate_contents(self):
        # KDE Plots
        if self.continous_columns:
            num_kde_cols = 3
            num_kde_rows = math.ceil(len(self.continous_columns) / num_kde_cols)
            fig, axs = plt.subplots(num_kde_rows, num_kde_cols, figsize=(18, 5 * num_kde_rows))
            axs = axs.flatten()

            for i, cont_column in enumerate(self.continous_columns):
                real = self.real_df[cont_column]
                fake = self.fake_df[cont_column]
                self.KDE_graph(real, fake, title=cont_column, ax=axs[i])

            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            current_time = time.time()
            filename = os.path.join(self.report_folder, self.contents_folder, 'KDE_plot'+str(current_time)+'.png')
            plt.savefig(filename)
            self.graphs.append(('KDE_plot'+str(current_time)+'.png','Kullback-Leibler (KL)' ))
            plt.close(fig)

        # Bar Plots
        if self.discrete_columns:
            num_bar_cols = 3
            num_bar_rows = math.ceil(len(self.discrete_columns) / num_bar_cols)
            fig, axs = plt.subplots(num_bar_rows, num_bar_cols, figsize=(18, 5 * num_bar_rows))
            axs = axs.flatten()
            next_ax = 0  # Start index for the next plot

            for disc_column in self.discrete_columns:
                real = self.real_df[disc_column]
                fake = self.fake_df[disc_column]
                ax = self.Bar_plot(real, fake, title=disc_column, ax=axs[next_ax])
                if ax is not None:  # Only move to next position if plot is generated
                    next_ax += 1

            for j in range(next_ax, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            current_time = time.time()
            filename = os.path.join(self.report_folder, self.contents_folder, 'Bar_plot'+str(current_time)+'.png')
            plt.savefig(filename)
            self.graphs.append(('Bar_plot'+str(current_time)+'.png', 'Category Coverage'))
            plt.close(fig)
        
        #History plot
        iteration = range(1, len(self.avg_scores_discrete['Category_Entropy']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Metrics over all iterations')

        # Define linestyles
        linestyles = ['-', '--', '-.', ':']

        # Combine discrete and continuous metrics into a single list for easier indexing
        all_metrics = list(self.avg_scores_discrete.items()) + list(self.avg_scores_continuous.items())

        # Plot each metric in a separate subplot
        for i, (key, values) in enumerate(all_metrics):
            row, col = divmod(i, 2)
            ax = axs[row, col]
            linestyle = linestyles[i % len(linestyles)]
            
            if key in self.avg_scores_discrete:
                ax.plot(iteration, values, label=f'Discrete: {key}', linestyle=linestyle, marker='o', color='#2C3E50', alpha=0.9, linewidth=2)
            else:
                ax.plot(iteration, values, label=f'Continuous: {key}', linestyle=linestyle, marker='s', color='skyblue', linewidth=2)
                
            ax.set_xlabel('Interval')
            ax.set_ylabel('Scores')
            ax.set_xticks(np.arange(0, len(iteration) + 2, step=1))
            ax.legend()
            ax.set_title(f'Metric: {key}')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to make room for the suptitle

        current_time = time.time()
        filename = os.path.join(self.report_folder, self.contents_folder, 'History' + str(current_time) + '.png')
        plt.savefig(filename)
        self.graphs.append(('History' + str(current_time) + '.png', 'Scores per iteration'))
        plt.close(fig)

        
    def generate_html(self):
        current_time = time.time()
        assets_file = os.path.join(self.report_folder, self.contents_folder)

        if self.discrete_df is not None:
            discrete_table_file = os.path.join(assets_file, 'results_table_discrete'+str(current_time)+'.html')       
            self.discrete_df.to_html(discrete_table_file, index=False)
            with open(discrete_table_file, 'r') as file:
                results_table_discrete_html = file.read()

        if self.continuous_df is not None:
            continuous_table_file = os.path.join(assets_file,'results_table_continuous'+str(current_time)+'.html')
            self.continuous_df.to_html(continuous_table_file, index=False)
            with open(continuous_table_file, 'r') as file:
                results_table_continuous_html = file.read()
        

        #Generate HTML Content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report Simulated User{self.number}</title>
            <style>
                body {{
                    margin:0;
                    padding:0;
                    background-color: #F2F4F4;
                    align-items:center;
                }}
                .header {{
                    width: 100%;
                    height: 100px;
                    background-color: #424242;
                    color: white;
                    padding: 10px 6%;
                    font-family: Tahoma, sans-serif;
                    font-size: 20px;
                }}
                .main-container {{
                    padding: 20px;
                    width:90%;
                    margin: auto;
                    height: 100%;
                    font-family: Tahoma, sans-serif;
                }}
                h2 {{
                    font-size: 35px;
                    font-family: Tahoma, sans-serif;
                    color: #424242;
                    margin-top: 15px;
                    margin-bottom: 40px;
                }}
                h3 {{
                    font-size: 20px;
                    font-family: Tahoma, sans-serif;
                    color: #424242;
                    margin:0;
                }}

                table {{
                    border-collapse: collapse;
                    width: 98%;
                    margin-top: 5px;
                    text-align: left;
                }}
                table th {{
                    background-color: #f2f2f2;
                    text-align: left;
                }}
                table th, table td {{
                    padding: 10px 20px;
                    border-bottom: 1px solid #ddd;
                }}
                img {{
                    display: block;
                    margin-top: 0px;
                    width:100%;
                    height: auto;
                }}
                .tables-container {{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    margin-bottom: 2%;
                    margin-top: 1%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                    position:relative;
                }}
                .graphs-container{{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    margin-bottom: 2%;
                    margin-top: 2%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                }}
                p {{
                    font-size: 16px;
                    width: 97%;
                    color: #424242;
                    font-family: Verdana;
                    text-align: justify;
                    font-weight: 100;
                    margin-bottom: 45px;
                    line-height: 1.4;
                }}
                .real {{
                    color: #2C3E50;
                    font-weight: bold;
                }}
                .fake {{
                    color: #2E86C1;
                    font-weight: bold;
                }}

            </style>
        </head>
        <body>
            <div class='header'>
                <h1>Evaluation Report Simulated Users {self.number} </h1>
            </div>
            <div class='main-container'>
            <div class='tables-container'>
            <h2>Metrics<h2>
            <p>The following section includes an overview of metrics for assessing the quality of synthetic data compared to 
            the real data. Most metrics
            follow a range from 0-1 or 0-100%. With 1 being the best possible score (Data matches synthetic data really well) and
            0 being the lowest score. The Entropy quantifies the diversity of categories, compared to the real data
            a low number indicates that the fake data has a similar level of diversity and
            randomness as the real data. Finally the Kullback Leibler and Kolmogorov Smirnov are both very common measures for this application that essentially
            measure the distance between distributions. In which the lower the better.</p>
            """
        
        #Add Results Table
        if self.continuous_df is not None:
            html_content += "<h3>Continuous Features</h3>"
            html_content += results_table_continuous_html
            html_content += "<br><br>"
        
        if self.discrete_df is not None:
            html_content += "<h3>Discrete Features</h3>"
            html_content += results_table_discrete_html
            html_content += "<br><br>"
        
        html_content += """</div><div class='graphs-container'><h2>Graphs<h2>
        <p> The following section includes an overview of graphs, for each column. 
        From the first set of (KDE density) graphs, the distribution of the real data can be compared to the one that is generated. This gives a good indication along with the KL score, of how well the generated data,
        follows distributions in the real data. The second set of graphs, shows the coverage of categories. By displaying
        the <span class='fake'>synthetic</span> data next to the <span class='real'>real data</span>, we can see if the propotion of categories present in the real data is captured by the 
        generated data. Note, that for the discrete columns
        only columns are shown in the graph section with less then 7 categories.
        However, the category coverage and adherance score for all columns can still be found in the table
        above.
        
        """
        #Add Plots
        if len(self.graphs) > 0:
            for filename, title in self.graphs:
                location = os.path.join(self.contents_folder,filename)
                html_content += f"<h3>{title}</h3>"
                html_content += f'<img src="{location}" alt="{title}" width="70%"><br>'
        html_content +="""
        </div>
        </div>
        </body>
        </html>"""

        html_file = os.path.join(self.report_folder, 'EvaluationReport_Users_'+str(self.number)+'_.html')
        with open(html_file, 'w') as file:
            file.write(html_content)

class SimulationQuality:

    def __init__(self, report_folder='Reports', contents_folder='contents', avg_rate_real=0):
        self.contents_folder = contents_folder
        self.graphs = []
        self.avg_ar = avg_rate_real
        self.total_users_overal = 0
        self.total_unique_users = set()
        self.folder = self.path_to_report(report_folder)

    def path_to_report(self, folder):
        dir = os.getcwd()
        self.report_folder = os.path.join(dir, folder)

    def evaluate(self, logs, recs, number=1):
        self.current_time = time.time()
        self.number= number
        self.recs = recs
        self.logs = logs
        self.generate_table()
        self.cumulative_graph()
        self.queue_graph()
        self.recommendations_graph()
        self.generate_html()
        self.logs = None
        self.graphs = []
        self.recs = None
        
    def generate_table(self):
        self.total_users_overal += len(self.logs['user_id'])
        self.total_unique_users.update(self.logs['user_id'])

        total_users_arrived = str(self.total_users_overal) + ' users since begin'
        unique_users = str(len(self.total_unique_users)) + ' users since begin'
        new_items = str(self.logs['new_items_added']) + ' items'
        average_queue_length = str(np.round(np.mean(self.logs['queue_lengths']), 2)) + ' users waiting on server'
        average_session_length = str(np.round(np.mean(self.logs['session_lengths']), 0)) + ' interactions in session'

        user_type_counts = pd.Series(self.logs['user_types']).value_counts()
        most_common_user_type = user_type_counts.idxmax()
        most_common_user_type_count = user_type_counts.max()
        most_common_user_type_percentage = np.round((most_common_user_type_count / len(self.logs['user_types'])) * 100, 1)
        most_common_user_type_str = f"{most_common_user_type} ({most_common_user_type_percentage}%)"
        
        summary_data = {
            'Statistic': [
                'Number of Users Arrived', 
                'Number of Unique users',
                'Number of new items added this batch',
                'Average Queue Length', 
                'Average Session Length',
                'Most Common User Type'
            ],
            'Value': [
                total_users_arrived, 
                unique_users,
                new_items, 
                average_queue_length, 
                average_session_length, 
                most_common_user_type_str
            ]}
        self.summary_df = pd.DataFrame(summary_data)

    
    def cumulative_graph(self, interval=5):

        #Simulated
        arrival_times = self.logs['arrival_times']
        min_arrival_time = min(arrival_times)

        arrival_times = [time - min_arrival_time for time in arrival_times]
        n_intervals = int(max(arrival_times) / interval)

        intervals = {}
        current = 0

        for i in range(0, n_intervals+1):
            intervals[current] = 0
            current += interval

        for time in arrival_times:
            inter = (time // interval) * interval
            intervals[inter] += 1

        # Calculate cumulative users from simulation
        cumulative_users = {}
        total_users = 0
        for time_bin in sorted(intervals.keys()):
            total_users += intervals[time_bin]
            cumulative_users[time_bin] = total_users

        # Calculate cumulative expected users from avg arrival rate original data
        num_intervals = len(cumulative_users)
        average_users_per_interval = self.avg_ar * interval
        cumulative_average_users = np.cumsum(np.full(num_intervals, average_users_per_interval))
        
        #Graph
        intervals = sorted(cumulative_users.keys())
        users_cumulative = [cumulative_users[interval] for interval in intervals]

        sns.set_style('ticks')
        plt.figure(figsize=(16, 10))

        # Plot the average arrival rate
        plt.plot(intervals, cumulative_average_users, color='#2C3E50', label='Expected Cumulative Users (Based on Real Data)')

        plt.plot(intervals, users_cumulative, color='#2E86C1', label='Cumulative Users (Simulated)', marker='o')
        #plt.fill_between(intervals, users_cumulative, color='#2E86C1', alpha=0.7)

        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative Users Arrived')
        plt.title('Cumulative Users Arrival Over Time')
        plt.grid(True)
        plt.legend(loc='lower right')
        filename = os.path.join(self.report_folder, self.contents_folder, 'Cumulative'+str(self.current_time)+'.png')
        plt.savefig(filename)
        self.graphs.append(('Cumulative'+str(self.current_time)+'.png', 'Simulated Arrival Rates'))
        plt.close()


    def queue_graph(self):
        
        #data
        data = self.logs['queue_lengths']
        x_values = list(range(1, len(data) + 1))

        #Graph
        plt.figure(figsize=(16, 10))
        plt.plot(x_values, data, color='#2E86C1', marker='o')
        plt.grid(True)
        plt.xlabel('Arrivals')
        plt.ylabel('Number Waiting')
        plt.title('Waiting Customers')
        filename = os.path.join(self.report_folder, self.contents_folder, 'Queue'+str(self.current_time)+'.png')
        plt.savefig(filename)
        self.graphs.append(('Queue'+str(self.current_time)+'.png', 'Queue Length'))
        plt.close()
    
    def recommendations_graph(self):
        all_recommendations = []
        for log_entry in self.recs:
            all_recommendations.extend(log_entry['recommendations'])
    
        # Count the frequency of each recommended item
        recommendation_counts = Counter(all_recommendations)
        
        # Get the top 10 most frequent items
        top_10 = recommendation_counts.most_common(10)
        items, counts = zip(*top_10)
        items = list(map(str, items))

        #Graph
        plt.figure(figsize=(16, 10))
        plt.bar(items, counts, color='#2E86C1', alpha=0.7)
        plt.xlabel('Item ID')
        plt.ylabel('Count')
        plt.title('Top 10 Most Recommended Items')
        filename = os.path.join(self.report_folder, self.contents_folder, 'Recommendations'+str(self.current_time)+'.png')
        plt.savefig(filename)
        self.graphs.append(('Recommendations'+str(self.current_time)+'.png', 'Top Recommendations'))
        plt.close()
        
        
    def generate_html(self):
        assets_file = os.path.join(self.report_folder, self.contents_folder)

        summary_table_file = os.path.join(assets_file, 'summary_table'+str(self.current_time)+'.html')       
        self.summary_df.to_html(summary_table_file, index=False)
        with open(summary_table_file, 'r') as file:
            summary_html = file.read()

        #Generate HTML Content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{
                    margin:0;
                    padding:0;
                    background-color: #F2F4F4;
                    align-items:center;
                }}
                .header {{
                    width: 100%;
                    height: 100px;
                    background-color: #424242;
                    color: white;
                    padding: 10px 6%;
                    font-family: Tahoma, sans-serif;
                    font-size: 20px;
                }}
                .main-container {{
                    padding: 20px;
                    width:90%;
                    margin: auto;
                    height: 100%;
                    font-family: Tahoma, sans-serif;
                }}
                .graphs-container {{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    margin-bottom: 2%;
                    margin-top: 2%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                }}
                .table-container {{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    padding-top: 40px;
                    padding-bottom: 40px;
                    margin-bottom: 2%;
                    margin-top: 2%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                }}
                img {{
                    display: block;
                    margin-top: 0px;
                    width:100%;
                    height: auto;
                }}
                h3 {{
                    font-size: 24px;
                    font-family: Tahoma, sans-serif;
                    color: #424242;
                    margin:0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 90%;
                    margin-top: 15px;
                    text-align: left;
                }}
                table th {{
                    background-color: #f2f2f2;
                    text-align: left;
                }}
                table th, table td {{
                    padding: 10px 20px;
                    border-bottom: 1px solid #ddd;
                }}
                p {{
                    font-size: 16px;
                    width: 97%;
                    color: #424242;
                    font-family: Verdana;
                    text-align: justify;
                    font-weight: 100;
                    margin-bottom: 45px;
                    line-height: 1.4;
                }}

            </style>
        </head>
        <body>
            <div class='header'>
                <h1>Simulation Report {str(self.number)}</h1>
            </div>

            <div class='main-container'>
            """


        html_content += """<div class='table-container'>
        <h3>Summary Table</h3><br><p>This report contains the simulation statistics from the last simulation interval. The first section contains a summary table that includes some statistics of the previous simulation interval.
        The second section contains three graphs. The first graphs shows the cumulative count of the simulated arrivals. This is accompanied with a second line which resembles the expected amount of arrivals based on the average arrival rate
        calculated from original data. The second graph contain the average amount of users waiting in the queue, which could be an indicator of system limitations (waiting on response of recsys, a slow recsys and potential limitations can identified in this graph). 
        And finally the top 10 most frequently recommended items are shown to analyse, when checked over multiple report, if the type and amount of (popular) recommended items shifts over time.<p><br><br>"""
        html_content += summary_html
        html_content += """</div>"""



        #Add Plots
        if len(self.graphs) > 0:
            for filename, title in self.graphs:
                html_content += f"<div class='graphs-container'>"
                location = os.path.join(self.contents_folder,filename)
                html_content += f"<h3>{title}</h3>"
                html_content += f'<img src="{location}" alt="{title}" width="70%"><br>'
                html_content += f"</div>"
        

        #end HTML

        html_content +="""
        </div>
        </body>
        </html>"""

        html_file = os.path.join(self.report_folder, 'EvaluationReportSimulation_'+str(self.number)+'.html')
        with open(html_file, 'w') as file:
            file.write(html_content)
        
    
class DataQualityItems:

    def __init__(self, discrete_columns, continous_columns, folder='Reports', contents_folder='contents', id_field=None):
        self.continous_columns = continous_columns
        self.discrete_columns = discrete_columns
        self.contents_folder = contents_folder
        if id_field is not None:
            self.discrete_columns.remove(id_field)
        self.graphs = []
        self.metrics_discrete = {
            'Category_Entropy' : [],
            'Chi-Sqaure avg P-Value' : [],
        }
        self.metrics_continuous = {
            'Kolmogorov Smirnov' : [],
            'Mean Score' : [],
        }
        self.avg_scores_discrete = {
            'Category_Entropy' : [],
            'Chi-Sqaure avg P-Value' : [],
        }
        self.avg_scores_continuous = {
            'Kolmogorov Smirnov' : [],
            'Mean Score' : [],
        }
        self.count_reports = []
        self.path_to_report(folder)

    def path_to_report(self, folder):
        dir = os.getcwd()
        self.report_folder = os.path.join(dir, folder)

    def calc_scores(self):
        for key, values in self.metrics_discrete.items():
            if values:
                value = np.mean(values)
            else:
                value = 0
            self.avg_scores_discrete[key].append(value)
            self.metrics_discrete[key] = []
        
        for key, values in self.metrics_continuous.items():
            if values:
                value = np.mean(values)
            else:
                value = 0
            self.avg_scores_continuous[key].append(value)
            self.metrics_continuous[key] = []


    def evaluate(self, real_df, fake_df, number=1):
        '''Main Function that can be called during simulation, to evaluate data quality'''
        discrete, continuous = [], []
        self.real_df = real_df
        self.fake_df = fake_df
        self.number = number

        if self.discrete_columns:
            for col in self.discrete_columns:
                real, fake = real_df[col], fake_df[col]
                real = real.reset_index(drop=True)
                fake = fake.reset_index(drop=True)
                coverage = self.Category_Coverage(real, fake)
                adherence = self.Category_Adherence(real, fake)
                entropy = self.Category_Entropy(real, fake)
                dist_match = self.Category_Distribution_Similarity(real, fake)
                mv_similarity = self.Missing_Value_Similarity(real, fake)
                chi_square, p_value = self.chi_square(real, fake)
                discrete.append({'Column': col, 'Chi-Square (p-value)': str(chi_square)+' ('+str(p_value)+')', 'Category Coverage': coverage, 'Category Adherence': adherence,
                                'Category Entropy': entropy, 'Category Distribution Match': dist_match,
                                'Missing Value Similarity': mv_similarity})
        
        if self.continous_columns:
            for col in self.continous_columns:
                real, fake = real_df[col], fake_df[col]
                kl_div = self.KL_Divergence(real, fake)
                ks_stat, p_value = self.KS_Test(real, fake)
                stats = self.Stat_Similarity(real, fake)
                mv_similarity = self.Missing_Value_Similarity(real, fake)
                continuous.append({'Column': col, 'Kullback Leibler': kl_div, 'Kolmogorov Smirnov': (ks_stat, p_value),
                                   'Mean Score': stats[0], 'Median Score': stats[1], 'Std Score': stats[2],
                                   'Missing Value Similarity': mv_similarity})
        
        self.discrete_df = pd.DataFrame(discrete) if discrete else pd.DataFrame({'None': [0]})
        self.continuous_df = pd.DataFrame(continuous) if continuous else pd.DataFrame({'None': [0]})

        self.calc_scores()
        self.generate_contents()
        self.generate_html()
        self.graphs = [] #Reset graphs
        self.discrete_df = None
        self.continuous_df = None


    ### METRICS SECTION ###

    def KL_Divergence(self, real, fake, bins=10):
        hist_real, _ = np.histogram(real, bins=bins, density=True)
        hist_fake, _ = np.histogram(fake, bins=bins, density=True)
        hist_real[hist_real == 0], hist_fake[hist_fake == 0] = 1e-10, 1e-10
        kl_div = np.sum(hist_real * np.log(hist_real / hist_fake))
        kl_div = np.round(kl_div, 3)
        return kl_div

    def KS_Test(self, real, fake):
        ks_stat, p_value = ks_2samp(real, fake)
        self.metrics_continuous['Kolmogorov Smirnov'].append(np.round(ks_stat, 3))
        return np.round(ks_stat, 3), np.round(p_value, 3)

    def Stat_Similarity(self, real, fake):
        stats = ['mean', 'median', 'std']
        scores = []
        for stat in stats:
            real_val, fake_val = getattr(real, stat)(), getattr(fake, stat)()
            score = np.round(1 - (abs(real_val - fake_val) / (real.max() - real.min())), 2)
            scores.append(np.clip(score, 0, 1))
        self.metrics_continuous['Mean Score'].append(scores[0])
        return scores

    def Category_Coverage(self, real, fake):
        unique_real, unique_fake = set(real), set(fake)
        coverage = len(unique_fake.intersection(unique_real)) / len(unique_real)
        return str(np.round(coverage * 100, 1)) + '%'

    def Category_Distribution_Similarity(self, real, fake):
        real_counts = pd.Series(real).value_counts(normalize=True)
        fake_counts = pd.Series(fake).value_counts(normalize=True)
        all_cats = set(real_counts.index).union(set(fake_counts.index))
        diffs = [abs(real_counts.get(cat, 0) - fake_counts.get(cat, 0)) for cat in all_cats]
        diff_norm = np.round(1 - np.mean(diffs), 2)
        return diff_norm

    def Category_Adherence(self, real, fake):
        unique_real = set(real)
        valid_fake = [val for val in fake if val in unique_real]
        return str(np.round(len(valid_fake) / len(fake) * 100, 1)) + '%'

    def Category_Entropy(self, real, fake):
        real_counts = pd.Series(real).value_counts(normalize=True, sort=False)
        fake_counts = pd.Series(fake).value_counts(normalize=True, sort=False)
        all_cats = set(real_counts.index).union(set(fake_counts.index))
        real_probs = np.array([real_counts.get(cat, 0) for cat in all_cats])
        fake_probs = np.array([fake_counts.get(cat, 0) for cat in all_cats])
        ent = np.round(abs(entropy(real_probs) - entropy(fake_probs)), 4)
        if ent < 1:
            self.metrics_discrete['Category_Entropy'].append(ent)
        return ent

    def Missing_Value_Similarity(self, real, fake):
        real_prop, fake_prop = real.isna().sum() / len(real), fake.isna().sum() / len(fake)
        return 1 - (fake_prop - real_prop)
    
    def chi_square(self, real, fake):
        contingency_table = pd.crosstab(real, fake)
        chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)
        self.metrics_discrete['Chi-Sqaure avg P-Value'].append(p_value)
        return np.round(chi2_statistic,3), np.round(p_value,3)

    ### GRAPHS SECTION ###

    def KDE_graph(self, real, fake, title, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = None

        sns.kdeplot(real, color="#2C3E50", fill=True, alpha=0.9, ax=ax, label='Real Data')
        sns.kdeplot(real, color="#424949", fill=False, linewidth=2, ax=ax)
        sns.kdeplot(fake, color="skyblue", fill=True, alpha=0.5, ax=ax, label='Synthetic Data')
        sns.kdeplot(fake, color="#3498DB", fill=False, linewidth=2, alpha=0.5, ax=ax)

        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        return ax

    def Bar_plot(self, real, fake, title, ax=None):
        unique_categories_real = set(real)
        unique_categories_fake = set(fake)
        category_coverage = len(unique_categories_fake.intersection(unique_categories_real)) / len(unique_categories_real)
        category_counts = {}

        if len(unique_categories_real) < 7:
            category_counts_real = Counter(real)
            category_counts_synthetic = Counter(fake)

            for category in unique_categories_real:
                category_counts[category] = {
                    'real_count': category_counts_real[category],
                    'synthetic_count': category_counts_synthetic[category]
                }

            categories = list(category_counts.keys())
            real_counts = [category_counts[category]['real_count'] for category in categories]
            synthetic_counts = [category_counts[category]['synthetic_count'] for category in categories]

            width = 0.35
            ind = range(len(categories))
            ax.bar(ind, real_counts, width, label='Real Data', color='#2C3E50', alpha=0.9)
            ax.bar([i + width for i in ind], synthetic_counts, width, label='Synthetic Data', color='skyblue', alpha=0.5)

            ax.set_xlabel('Categories')
            ax.set_ylabel('Counts')
            ax.set_title(title)
            ax.set_xticks([i + width / 2 for i in ind])
            ax.set_xticklabels(categories)
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            return ax
        else:
            return None  # Return None if there are too many categories

    def generate_contents(self):
        # KDE Plots
        if self.continous_columns:
            num_kde_cols = 3
            num_kde_rows = math.ceil(len(self.continous_columns) / num_kde_cols)
            fig, axs = plt.subplots(num_kde_rows, num_kde_cols, figsize=(18, 5 * num_kde_rows))
            axs = axs.flatten()

            for i, cont_column in enumerate(self.continous_columns):
                real = self.real_df[cont_column]
                fake = self.fake_df[cont_column]
                self.KDE_graph(real, fake, title=cont_column, ax=axs[i])

            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            current_time = time.time()
            filename = os.path.join(self.report_folder, self.contents_folder, 'KDE_plot'+str(current_time)+'.png')
            plt.savefig(filename)
            self.graphs.append(('KDE_plot'+str(current_time)+'.png','Kullback-Leibler (KL)' ))
            plt.close(fig)

        # Bar Plots
        if self.discrete_columns:
            num_bar_cols = 3
            num_bar_rows = math.ceil(len(self.discrete_columns) / num_bar_cols)
            fig, axs = plt.subplots(num_bar_rows, num_bar_cols, figsize=(18, 5 * num_bar_rows))
            axs = axs.flatten()
            next_ax = 0  # Start index for the next plot

            for disc_column in self.discrete_columns:
                real = self.real_df[disc_column]
                fake = self.fake_df[disc_column]
                ax = self.Bar_plot(real, fake, title=disc_column, ax=axs[next_ax])
                if ax is not None:  # Only move to next position if plot is generated
                    next_ax += 1

            for j in range(next_ax, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout()
            current_time = time.time()
            filename = os.path.join(self.report_folder, self.contents_folder, 'Bar_plot'+str(current_time)+'.png')
            plt.savefig(filename)
            self.graphs.append(('Bar_plot'+str(current_time)+'.png', 'Category Coverage'))
            plt.close(fig)
        
        #History plot
        iteration = range(1, len(self.avg_scores_discrete['Category_Entropy']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Metrics over all iterations')

        # Define linestyles
        linestyles = ['-', '--', '-.', ':']

        # Combine discrete and continuous metrics into a single list for easier indexing
        all_metrics = list(self.avg_scores_discrete.items()) + list(self.avg_scores_continuous.items())

        # Plot each metric in a separate subplot
        for i, (key, values) in enumerate(all_metrics):
            row, col = divmod(i, 2)
            ax = axs[row, col]
            linestyle = linestyles[i % len(linestyles)]
            
            if key in self.avg_scores_discrete:
                ax.plot(iteration, values, label=f'Discrete: {key}', linestyle=linestyle, marker='o', color='#2C3E50', alpha=0.9, linewidth=2)
            else:
                ax.plot(iteration, values, label=f'Continuous: {key}', linestyle=linestyle, marker='s', color='skyblue', linewidth=2)
                
            ax.set_xlabel('Interval')
            ax.set_ylabel('Scores')
            ax.set_xticks(np.arange(0, len(iteration) + 2, step=1))
            ax.legend()
            ax.set_title(f'Metric: {key}')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to make room for the suptitle

        current_time = time.time()
        filename = os.path.join(self.report_folder, self.contents_folder, 'History' + str(current_time) + '.png')
        plt.savefig(filename)
        self.graphs.append(('History' + str(current_time) + '.png', 'Scores per iteration'))
        plt.close(fig)

        
    def generate_html(self):
        current_time = time.time()
        assets_file = os.path.join(self.report_folder, self.contents_folder)

        if self.discrete_df is not None:
            discrete_table_file = os.path.join(assets_file, 'results_table_discrete'+str(current_time)+'.html')       
            self.discrete_df.to_html(discrete_table_file, index=False)
            with open(discrete_table_file, 'r') as file:
                results_table_discrete_html = file.read()

        if self.continuous_df is not None:
            continuous_table_file = os.path.join(assets_file,'results_table_continuous'+str(current_time)+'.html')
            self.continuous_df.to_html(continuous_table_file, index=False)
            with open(continuous_table_file, 'r') as file:
                results_table_continuous_html = file.read()
        

        #Generate HTML Content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report Items {self.number}</title>
            <style>
                body {{
                    margin:0;
                    padding:0;
                    background-color: #F2F4F4;
                    align-items:center;
                }}
                .header {{
                    width: 100%;
                    height: 100px;
                    background-color: #424242;
                    color: white;
                    padding: 10px 6%;
                    font-family: Tahoma, sans-serif;
                    font-size: 20px;
                }}
                .main-container {{
                    padding: 20px;
                    width:90%;
                    margin: auto;
                    height: 100%;
                    font-family: Tahoma, sans-serif;
                }}
                h2 {{
                    font-size: 35px;
                    font-family: Tahoma, sans-serif;
                    color: #424242;
                    margin-top: 15px;
                    margin-bottom: 40px;
                }}
                h3 {{
                    font-size: 20px;
                    font-family: Tahoma, sans-serif;
                    color: #424242;
                    margin:0;
                }}

                table {{
                    border-collapse: collapse;
                    width: 98%;
                    margin-top: 5px;
                    text-align: left;
                }}
                table th {{
                    background-color: #f2f2f2;
                    text-align: left;
                }}
                table th, table td {{
                    padding: 10px 20px;
                    border-bottom: 1px solid #ddd;
                }}
                img {{
                    display: block;
                    margin-top: 0px;
                    width:100%;
                    height: auto;
                }}
                .tables-container {{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    margin-bottom: 2%;
                    margin-top: 1%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                    position:relative;
                }}
                .graphs-container{{
                    background-color: white;
                    width: 100%;
                    padding: 20px;
                    margin-bottom: 2%;
                    margin-top: 2%;
                    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
                }}
                p {{
                    font-size: 16px;
                    width: 97%;
                    color: #424242;
                    font-family: Verdana;
                    text-align: justify;
                    font-weight: 100;
                    margin-bottom: 45px;
                    line-height: 1.4;
                }}
                .real {{
                    color: #2C3E50;
                    font-weight: bold;
                }}
                .fake {{
                    color: #2E86C1;
                    font-weight: bold;
                }}

            </style>
        </head>
        <body>
            <div class='header'>
                <h1>Evaluation Report Simulated Items {self.number} </h1>
            </div>
            <div class='main-container'>
            <div class='tables-container'>
            <h2>Metrics<h2>
            <p>The following section includes an overview of metrics for assessing the quality of synthetic data compared to 
            the real data. Most metrics
            follow a range from 0-1 or 0-100%. With 1 being the best possible score (Data matches synthetic data really well) and
            0 being the lowest score. The Entropy quantifies the diversity of categories, compared to the real data
            a low number indicates that the fake data has a similar level of diversity and
            randomness as the real data. Finally the Kullback Leibler and Kolmogorov Smirnov are both very common measures for this application that essentially
            measure the distance between distributions. In which the lower the better.</p>
            """
        
        #Add Results Table
        if self.continuous_df is not None:
            html_content += "<h3>Continuous Features</h3>"
            html_content += results_table_continuous_html
            html_content += "<br><br>"
        
        if self.discrete_df is not None:
            html_content += "<h3>Discrete Features</h3>"
            html_content += results_table_discrete_html
            html_content += "<br><br>"
        
        html_content += """</div><div class='graphs-container'><h2>Graphs<h2>
        <p> The following section includes an overview of graphs, for each column. 
        From the first set of (KDE density) graphs, the distribution of the real data can be compared to the one that is generated. This gives a good indication along with the KL score, of how well the generated data,
        follows distributions in the real data. The second set of graphs, shows the coverage of categories. By displaying
        the <span class='fake'>synthetic</span> data next to the <span class='real'>real data</span>, we can see if the propotion of categories present in the real data is captured by the 
        generated data. Note, that for the discrete columns
        only columns are shown in the graph section with less then 7 categories.
        However, the category coverage and adherance score for all columns can still be found in the table
        above.
        
        """
        #Add Plots
        if len(self.graphs) > 0:
            for filename, title in self.graphs:
                location = os.path.join(self.contents_folder,filename)
                html_content += f"<h3>{title}</h3>"
                html_content += f'<img src="{location}" alt="{title}" width="70%"><br>'
        html_content +="""
        </div>
        </div>
        </body>
        </html>"""

        html_file = os.path.join(self.report_folder, 'EvaluationReport_Items_'+str(self.number)+'_.html')
        with open(html_file, 'w') as file:
            file.write(html_content)
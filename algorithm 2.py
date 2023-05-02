def load_data(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df


def state_combined_bar_plot(left, right):
    import matplotlib.pyplot as plt

    # Get the counts of records per state for both datasets
    left_counts = left['state'].value_counts()
    right_counts = right['state'].value_counts()
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # Plot the left dataset counts
    left_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Left Dataset')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Number of Records')
    # Plot the right dataset counts
    right_counts.plot(kind='bar', ax=ax2)
    ax2.set_title('Right Dataset')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Number of Records')
    # Show the plot
    plt.show()


def exploratory_data_analysis(left, right):
    print(f"left rows: {left.shape[0]}")
    print(f"left columns: {left.shape[0]}")
    print(f"right rows: {right.shape[0]}")
    print(f"right columns: {right.shape[0]}")
    print("left: {}".format(list(left.columns)))
    print("right: {}".format(list(right.columns)))
    print(f"There are {left.shape[0]} observations in the left dataset.")
    print(f"There are {right.shape[0]} observations in the right dataset.")


def city_bar_plot(df):
    """
    Generate a bar plot of the top 10 cities with the highest number of businesses using a colormap
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    city_counts = df['city'].value_counts().head(10)
    colors = plt.cm.tab20c(np.linspace(0, 1, len(city_counts)))
    city_counts.plot(kind='bar', color=colors)
    plt.title('Top 10 Cities with the Highest Number of Businesses')
    plt.xlabel('City')
    plt.ylabel('Count')
    plt.show()


def state_pie_plot(df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    df['state'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Businesses by State')
    plt.ylabel('')
    plt.show()


def clean_data(df):
    import pandas as pd
    
    # Replace "N/A" values in the "address" column with NaN
    df['address'] = df['address'].str.replace(r'\bN/A\b', '').replace('', pd.NA)
    # Remove trailing commas from the "address" column
    df['address'] = df['address'].str.rstrip(',')
    # Convert strings to lowercase/uppercase
    df['address'] = df['address'].str.lower()
    df['name'] = df['name'].str.lower()
    df['city'] = df['city'].str.lower()
    df['state'] = df['state'].str.upper()
    # Clean zip codes and change column names to make same name for all columns
    if "zip_code" in df.columns:
        df['zip_code'] = df['zip_code'].str.slice(0, 5)
    else:
        df['postal_code'] = df['postal_code'].fillna('')
        df['postal_code'] = df['postal_code'].astype(str)
        df['postal_code'] = df['postal_code'].str.replace(r'\.\d+', '')
        df.columns.values[0] = 'business_id'
        df.columns.values[5] = 'zip_code'

    return df


def matching_algo(left, right):
    def clustering(df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline

        vectorizer = TfidfVectorizer(stop_words='english')
        kmeans = KMeans(n_clusters=100, random_state=42)
        pipeline = Pipeline([('vectorizer', vectorizer), ('kmeans', kmeans)])

        df["combined"] = (df["address"].fillna('') + " " +
                          df["city"].fillna('') + " " +
                          df["state"].fillna(''))
    
        df["cluster"] = pipeline.fit_predict(df["combined"])

        return df
    
    left = clustering(left)
    right = clustering(right)

    from fuzzywuzzy import fuzz

    def match_entities(left_entity, right_entities):
        best_match = None
        best_score = 0
    
        for right_entity in right_entities.itertuples():
            score = fuzz.token_set_ratio(left_entity.combined, right_entity.combined)/100
            if score > best_score:
                best_score = score
                best_match = right_entity
            
        return best_match, best_score

    matches = []
    for cluster_label in set(left["cluster"]):
        left_cluster = left[left["cluster"] == cluster_label]
        right_cluster = right[right["cluster"] == cluster_label]
    
        for left_entity in left_cluster.itertuples():
            right_entities = right_cluster[right_cluster['name'] == left_entity.name]
            right_entities = right_cluster[right_cluster['zip_code'] == left_entity.zip_code]
            match, score = match_entities(left_entity, right_entities)
            if match and score > 0.8:
                matches.append((int(left_entity.business_id), int(match.business_id), score))

    return matches


def download_result(match):
    import pandas as pd
    match_df = pd.DataFrame(match, columns=['left_business_id', 'right_entity_id', 'confidence_score'])
    match_df.to_csv('matches.csv', index=False)
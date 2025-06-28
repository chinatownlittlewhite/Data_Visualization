import pandas as pd
import os
import re
from itertools import combinations
import gc
import numpy as np

# --- 配置 ---
DATA_DIR = './data'
OUTPUT_DIR = 'data_possessed'


# --- 主函数 ---
def main():
    print("--- 开始数据预处理与预计算流程 ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    try:
        # --- 1. 加载核心数据 ---
        print("正在加载核心数据 (basics, ratings, names)...")
        basics_df = pd.read_csv(os.path.join(DATA_DIR, 'title.basics.tsv'), sep='\t', low_memory=False)
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'title.ratings.tsv'), sep='\t', low_memory=False)
        names_df = pd.read_csv(os.path.join(DATA_DIR, 'name.basics.tsv'), sep='\t', low_memory=False)[
            ['nconst', 'primaryName']]

        # --- 2. 预处理并保存电影元数据 (用于Tab 1) ---
        print("正在预处理电影元数据...")
        movies_df = basics_df[basics_df['titleType'] == 'movie'].copy()
        metadata_df = pd.merge(movies_df, ratings_df, on='tconst')
        metadata_df['startYear'] = pd.to_numeric(metadata_df['startYear'], errors='coerce')
        metadata_df.dropna(subset=['startYear', 'genres'], inplace=True)
        metadata_df = metadata_df[metadata_df['genres'] != '\\N']
        metadata_df['startYear'] = metadata_df['startYear'].astype(int)
        metadata_df['genres_list'] = metadata_df['genres'].str.split(',')

        final_metadata = metadata_df[
            ['tconst', 'primaryTitle', 'startYear', 'genres_list', 'averageRating', 'numVotes']]
        final_metadata.to_parquet(os.path.join(OUTPUT_DIR, 'movies_metadata.parquet'))
        print("✔ 电影元数据已保存为 movies_metadata.parquet")
        del basics_df, ratings_df, movies_df;
        gc.collect()

        # --- 3. 预处理并保存全球化足迹数据 (用于Tab 2) ---
        print("正在预处理全球化足跡数据...")
        akas_df = pd.read_csv(os.path.join(DATA_DIR, 'title.akas.tsv'), sep='\t', low_memory=False, on_bad_lines='skip')
        akas_df = akas_df[akas_df['region'].str.match(r'^[A-Z]{2}$', na=False)]
        high_vote_movies = final_metadata[final_metadata['numVotes'] > 1000]['tconst']
        akas_df_filtered = akas_df[akas_df['titleId'].isin(high_vote_movies)]

        akas_agg = akas_df_filtered.groupby(['titleId', 'region']).agg(
            titles=('title', lambda x: ' / '.join(x)),
            title_count=('title', 'size')
        ).reset_index()

        akas_agg.rename(columns={'titleId': 'tconst', 'titles': 'title'}).to_parquet(
            os.path.join(OUTPUT_DIR, 'movie_akas.parquet'))
        print("✔ 电影全球化数据已保存为 movie_akas.parquet")
        del akas_df, akas_df_filtered, akas_agg;
        gc.collect()

        # --- 4. 预计算人物分析所需数据 (用于Tab 3) ---
        print("正在预计算人物分析数据 (此过程可能较慢)...")
        crew_df = pd.read_csv(os.path.join(DATA_DIR, 'title.crew.tsv'), sep='\t', low_memory=False)
        principals_df = pd.read_csv(os.path.join(DATA_DIR, 'title.principals.tsv'), sep='\t', low_memory=False)

        directors = crew_df[crew_df['directors'] != '\\N'][['tconst', 'directors']].assign(category='director')
        directors = directors.assign(nconst=directors['directors'].str.split(',')).explode('nconst')[
            ['tconst', 'nconst', 'category']]
        actors = principals_df[principals_df['category'].isin(['actor', 'actress'])][['tconst', 'nconst', 'category']]
        people_df = pd.concat([directors, actors])
        del crew_df, principals_df, directors, actors;
        gc.collect()

        career_data = pd.merge(people_df, final_metadata, on='tconst')
        person_movie_counts = career_data['nconst'].value_counts().reset_index()
        person_movie_counts.columns = ['nconst', 'movie_count']
        relevant_nconsts = career_data['nconst'].unique()
        names_df_filtered = names_df[names_df['nconst'].isin(relevant_nconsts)]
        names_with_counts_df = pd.merge(names_df_filtered, person_movie_counts, on='nconst', how='left')
        names_with_counts_df['movie_count'] = names_with_counts_df['movie_count'].fillna(0).astype(int)
        career_data = pd.merge(career_data, names_with_counts_df[['nconst', 'primaryName']], on='nconst', how='left')
        career_data.dropna(subset=['primaryName'], inplace=True)

        print("拆分并保存个人职业生涯数据为4个文件...")
        career_data_to_save = career_data[
            ['nconst', 'primaryName', 'tconst', 'primaryTitle', 'startYear', 'averageRating', 'numVotes']]
        career_chunks = np.array_split(career_data_to_save, 4)
        for i, chunk in enumerate(career_chunks):
            chunk.to_parquet(os.path.join(OUTPUT_DIR, f'career_data_part_{i + 1}.parquet'))
        print("✔ 个人职业生涯数据已拆分为4个文件并保存。")

        people_high_vote = people_df[people_df['tconst'].isin(high_vote_movies)]
        movie_to_people = people_high_vote.groupby('tconst')['nconst'].apply(list)
        collaboration_edges = []
        for people_list in movie_to_people:
            if len(people_list) > 20:
                people_list = pd.Series(people_list).sample(20, random_state=42).tolist()
            for p1, p2 in combinations(sorted(people_list), 2):
                collaboration_edges.append((p1, p2))
        collaboration_df = pd.DataFrame(collaboration_edges, columns=['person1', 'person2'])
        collaboration_counts = collaboration_df.groupby(['person1', 'person2']).size().reset_index(name='count')
        collaboration_counts.to_parquet(os.path.join(OUTPUT_DIR, 'collaboration_counts.parquet'))
        print("✔ 合作关系数据已保存为 collaboration_counts.parquet")

        names_with_counts_df.to_parquet(os.path.join(OUTPUT_DIR, 'names.parquet'))
        print("✔ 已筛选并带有作品数的影人姓名数据已保存为 names.parquet")
        del people_df, career_data, collaboration_df, collaboration_counts, movie_to_people, people_high_vote, names_df, names_with_counts_df;
        gc.collect()

        # --- 5. 预处理影评数据 (用于Tab 4) ---
        print("正在预处理影评数据...")
        reviews_df = pd.read_csv(os.path.join(DATA_DIR, 'IMDB Dataset.csv'))

        def clean_text(text):
            text = re.sub(r'<br\s*/?>', ' ', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        reviews_df['review_cleaned'] = reviews_df['review'].apply(clean_text)

        print("拆分并保存影评数据为2个文件...")
        reviews_to_save = reviews_df[['review_cleaned', 'sentiment']]
        review_chunks = np.array_split(reviews_to_save, 2)
        for i, chunk in enumerate(review_chunks):
            chunk.to_parquet(os.path.join(OUTPUT_DIR, f'reviews_cleaned_part_{i + 1}.parquet'))
        print("✔ 影评数据已拆分为2个文件并保存。")

        print("\n--- ✅ 所有数据预处理和预计算成功完成！ ---")

    except Exception as e:
        print(f"\n--- ❌ 发生错误: {e} ---")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import zhipuai
import requests
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from streamlit_searchbox import st_searchbox
import re
import pycountry
import warnings
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

# --- 页面基础设置 ---
st.set_page_config(
    page_title="IMDB电影数据分析",
    page_icon="✨",
    layout="wide"
)

import platform

system = platform.system()
if system == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
if system == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# --- 资源加载与缓存 ---
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "./distilbert-imdb-finetuned/distilbert-imdb-finetuned"
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=DEVICE)

# --- 新增: AI 模型选择UI ---
st.sidebar.header("🤖 AI 模型设置")
# 使用 session_state 来持久化模型选择
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Gemini (需代理)"

st.session_state.selected_model = st.sidebar.selectbox(
    "选择您想使用的AI模型:",
    ["Gemini (需代理)", "智谱AI (国内)", "本地Ollama"],
    index=["Gemini (需代理)", "智谱AI (国内)", "本地Ollama"].index(st.session_state.selected_model)
)


gemini_is_configured = False
zhipuai_is_configured = False

try:
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_is_configured = True
except (KeyError, AttributeError):
    pass # 允许在没有Gemini密钥的情况下运行

try:
    if "ZHIPU_API_KEY" in st.secrets and st.secrets["ZHIPU_API_KEY"]:
        # 仅检查密钥存在性，客户端在调用时实例化
        zhipuai_is_configured = True
except (KeyError, AttributeError):
    pass


if st.session_state.selected_model == "Gemini (需代理)" and not gemini_is_configured:
    st.sidebar.warning("Gemini API密钥未配置。请在 .streamlit/secrets.toml 文件中设置 `GEMINI_API_KEY`。")
elif st.session_state.selected_model == "智谱AI (国内)" and not zhipuai_is_configured:
    st.sidebar.warning("智谱AI API密钥未配置。请在 .streamlit/secrets.toml 文件中设置 `ZHIPU_API_KEY`。")
elif st.session_state.selected_model == "本地Ollama":
    st.sidebar.info("请确保您的本地Ollama服务正在运行。推荐模型: `llama3` 或 `qwen:7b`。")



@st.cache_data(show_spinner="🧠 AI 正在思考...")
def call_llm(prompt, model_selection):
    """根据选择调用不同的大语言模型并返回结果"""
    if model_selection == "Gemini (需代理)":
        if not gemini_is_configured:
            return "错误：Gemini API密钥未配置。"
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"调用Gemini时出错: {e}"

    elif model_selection == "智谱AI (国内)":
        if not zhipuai_is_configured:
            return "错误：智谱AI API密钥未配置。"
        try:
            client = zhipuai.ZhipuAI(api_key=st.secrets["ZHIPU_API_KEY"])
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"调用智谱AI时出错: {e}"

    elif model_selection == "本地Ollama":
        try:
            # 假设Ollama服务在标准端口运行
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=120) # 增加超时时间
            response.raise_for_status() # 如果请求失败则抛出异常
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            return "错误：无法连接到本地Ollama服务。请确保Ollama正在运行。"
        except Exception as e:
            return f"调用本地Ollama时出错: {e}"
    else:
        return "错误：无效的AI模型选择。"


# --- 资源加载与缓存 ---
@st.cache_data
def convert_alpha2_to_alpha3(alpha_2):
    if alpha_2 == 'UK': return 'GBR'
    if alpha_2 == 'SU': return 'RUS'
    try:
        return pycountry.countries.get(alpha_2=alpha_2).alpha_3
    except (AttributeError, LookupError):
        return None


@st.cache_data
def load_all_data():
    base_path = "data_possessed"
    try:
        print("Loading and concatenating chunked files for reviews and career data...")

        # 加载并合并 reviews_cleaned (2个部分)
        reviews_parts = [pd.read_parquet(f"{base_path}/reviews_cleaned_part_{i}.parquet") for i in range(1, 3)]
        reviews_df = pd.concat(reviews_parts, ignore_index=True)

        # 加载并合并 career_data (3个部分)
        career_parts = [pd.read_parquet(f"{base_path}/career_data_part_{i}.parquet") for i in range(1, 5)]
        career_df = pd.concat(career_parts, ignore_index=True)

        print("All data loaded successfully.")

        data = {
            'metadata': pd.read_parquet(f"{base_path}/movies_metadata.parquet"),
            'reviews': reviews_df,
            'names': pd.read_parquet(f"{base_path}/names.parquet").set_index('nconst'),
            'career': career_df,
            'akas': pd.read_parquet(f"{base_path}/movie_akas.parquet"),
            'collab': pd.read_parquet(f"{base_path}/collaboration_counts.parquet"),
        }
        data['names'] = data['names'].sort_values('primaryName')
        return data
    except FileNotFoundError as e:
        st.error(
            f"❌ 错误: 找不到数据文件。请确保您已成功运行 `prepare_data.py` 脚本来生成所有必需的文件。具体文件: {e.filename}")
        st.info(
            "如果您刚刚修改了脚本以拆分文件，请先删除旧的 `career_data.parquet` 和 `reviews_cleaned.parquet` 文件，然后重新运行 `prepare_data.py`。")
        return None


# --- 加载资源 ---
with st.spinner('正在加载海量电影数据...'):
    data = load_all_data()

if data is None: st.stop()

# --- 主界面 ---
st.title("✨ IMDB电影数据可视化平台")
st.markdown("欢迎来到IMDB数据探索平台！本平台融合**传统数据分析**与**生成式AI**，为您提供更深层次的电影行业洞察。")

with st.expander("ℹ️ 点击查看数据集总览"):
    st.markdown(f"""
    - **🎬 电影元数据**: 包含 **{len(data['metadata']):,}** 部电影信息。
    - **✍️ 影评数据**: 包含 **{len(data['reviews']):,}** 条用户评论。
    - **🧑‍🎨 影人数据**: 包含 **{len(data['names']):,}** 位核心影人信息。
    - **🌏 全球上映数据**: 记录高票电影的全球上映情况。
    """)
    st.markdown(f"**🤖 当前AI分析模型**: `{st.session_state.selected_model}` (可在左侧边栏切换)")

# --- Tab布局 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 电影元数据探索", "🌍 全球市场分析", "👥 行业网络与人物分析", "📝 影评文本分析",
     "🤖 影评智能剖析"])

# --- Tab 1: 电影元数据探索 ---
with tab1:
    st.header("📊 电影元数据探索")
    st.sidebar.header("元数据筛选器")
    min_year, max_year = int(data['metadata']['startYear'].min()), int(data['metadata']['startYear'].max())
    year_range = st.sidebar.slider("选择上映年份范围:", min_year, max_year, (min_year, max_year), key="tab1_year")
    all_genres = sorted(list(set(g for sublist in data['metadata']['genres_list'] for g in sublist)))
    selected_genres = st.sidebar.multiselect("选择电影类型:", options=all_genres, default=["Action", "Drama", "Comedy", "Family", "Crime", "Documentary", "Adventure", "Fantasy", "Thriller", "Horror"],
                                             key="tab1_genres")
    min_votes = st.sidebar.slider("最小投票数:", 0, 50000, 1000, step=100, key="tab1_votes")

    filtered_meta_df = data['metadata'][
        (data['metadata']['startYear'] >= year_range[0]) & (data['metadata']['startYear'] <= year_range[1]) &
        (data['metadata']['numVotes'] >= min_votes)
        ]
    if selected_genres:
        selected_genres_set = set(selected_genres)
        filtered_meta_df = filtered_meta_df[
            filtered_meta_df['genres_list'].apply(lambda g: not selected_genres_set.isdisjoint(g))]

    if filtered_meta_df.empty:
        st.warning("在当前筛选条件下，没有找到任何电影。")
    else:
        st.subheader("📈 核心指标一览")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("电影总数", f"{len(filtered_meta_df):,}")
        kpi2.metric("平均IMDB评分", f"{filtered_meta_df['averageRating'].mean():.2f}")
        kpi3.metric("总投票数", f"{filtered_meta_df['numVotes'].sum():,}")
        st.markdown("---")
        st.subheader("🎨 可视化分析")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🎭 各类型电影数量")
            genre_df = filtered_meta_df.explode('genres_list').rename(columns={'genres_list': 'genre'})
            genre_counts = genre_df[genre_df['genre'].isin(selected_genres)]['genre'].value_counts()
            fig_genre_count = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values,
                                     labels={'x': '电影类型', 'y': '数量'}, color=genre_counts.index,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_genre_count, use_container_width=True)
            st.markdown("##### ⭐ 评分 vs. 投票数 (对数尺度)")
            fig_scatter_votes = px.scatter(filtered_meta_df.sample(min(1000, len(filtered_meta_df))), x="averageRating",
                                           y="numVotes", hover_name="primaryTitle", log_y=True,
                                           labels={"averageRating": "IMDB评分", "numVotes": "投票数 (对数)"})
            st.plotly_chart(fig_scatter_votes, use_container_width=True)
        with col2:
            st.markdown("##### 📈 平均评分随年份变化")
            rating_over_time = filtered_meta_df.groupby('startYear')['averageRating'].mean().reset_index()
            fig_rating_time = px.line(rating_over_time, x='startYear', y='averageRating',
                                      labels={'startYear': '上映年份', 'averageRating': '平均IMDB评分'})
            st.plotly_chart(fig_rating_time, use_container_width=True)
            st.markdown("##### 📋 筛选结果详情 (Top 10)")
            st.dataframe(filtered_meta_df[['primaryTitle', 'startYear', 'averageRating', 'numVotes']].rename(
                columns={'primaryTitle': '标题', 'startYear': '年份', 'averageRating': '评分',
                         'numVotes': '投票数'}).head(10))

# --- Tab 2: 全球市场分析 ---
with tab2:
    st.header("🌍 电影全球化市场分析")
    st.markdown("从宏观视角分析**某一类电影**在全球的市场分布热度，并可下钻查看具体国家/地区的上映影片。")
    st.sidebar.header("全球化筛选器")
    min_year_t2, max_year_t2 = int(data['metadata']['startYear'].min()), int(data['metadata']['startYear'].max())
    year_range_t2 = st.sidebar.slider("选择上映年份范围:", min_year_t2, max_year_t2, (min_year_t2, max_year_t2),
                                      key="tab2_year")
    all_genres_t2 = sorted(list(set(g for sublist in data['metadata']['genres_list'] for g in sublist)))
    selected_genres_t2 = st.sidebar.multiselect("选择电影类型:", options=all_genres_t2, default=["Action", "Drama", "Comedy", "Family", "Crime", "Documentary", "Adventure", "Fantasy", "Thriller", "Horror"],
                                                key="tab2_genres")
    min_votes_t2 = st.sidebar.slider("最小投票数:", 1000, 100000, 25000, step=1000, key="tab2_votes")

    filtered_movies_map = data['metadata'][
        (data['metadata']['startYear'] >= year_range_t2[0]) & (data['metadata']['startYear'] <= year_range_t2[1]) &
        (data['metadata']['numVotes'] >= min_votes_t2)
        ]
    if selected_genres_t2:
        selected_genres_set_t2 = set(selected_genres_t2)
        filtered_movies_map = filtered_movies_map[
            filtered_movies_map['genres_list'].apply(lambda g: not selected_genres_set_t2.isdisjoint(g))
        ]

    if filtered_movies_map.empty:
        st.warning("在当前筛选条件下，没有找到任何电影。")
    else:
        st.info(f"根据您的筛选，共找到 **{len(filtered_movies_map):,}** 部电影进行全球市场分析。")
        target_tconsts = filtered_movies_map['tconst']
        map_akas_df = data['akas'][data['akas']['tconst'].isin(target_tconsts)]
        if map_akas_df.empty:
            st.warning("这些电影没有找到任何全球上映信息。")
        else:
            region_summary = map_akas_df.groupby('region')['tconst'].nunique().reset_index(
                name='movie_count').sort_values('movie_count', ascending=False)
            region_summary['iso_alpha'] = region_summary['region'].apply(convert_alpha2_to_alpha3)
            region_summary_mapped = region_summary.dropna(subset=['iso_alpha'])
            if region_summary_mapped.empty:
                st.warning("无法将筛选出的电影地区代码映射到标准地理编码，因此无法生成地图。")
            else:
                st.subheader("🗺️ 全球市场分布热力图")
                fig_map = go.Figure(data=go.Choropleth(
                    locations=region_summary_mapped['iso_alpha'],
                    z=region_summary_mapped['movie_count'],
                    text=region_summary_mapped['region'],
                    colorscale=px.colors.sequential.Plasma, reversescale=True,
                    marker_line_color='darkgray', marker_line_width=0.5,
                    colorbar_title='电影数量',
                    hovertemplate='<b>%{text}</b><br>电影数量: %{z}<extra></extra>'
                ))

                fig_map.update_layout(
                    title_text=f'筛选出的{len(filtered_movies_map):,}部电影的全球市场分布',
                    geo=dict(
                        showframe=False,
                        showcoastlines=False,
                        projection_type='natural earth',
                        lataxis_showgrid=True,
                        lonaxis_showgrid=True
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=600
                )

                st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("---")
            st.subheader("📍 下钻分析：查看具体地区的上映影片")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(region_summary[['region', 'movie_count']].rename(
                    columns={'region': '地区', 'movie_count': '电影数量'}))
            with col2:
                available_regions = region_summary['region'].unique()
                if len(available_regions) > 0:
                    selected_region = st.selectbox("选择一个地区查看详情:", options=available_regions)
                    if selected_region:
                        region_movie_tconsts = map_akas_df[map_akas_df['region'] == selected_region]['tconst'].unique()
                        region_movies_details = filtered_movies_map[
                            filtered_movies_map['tconst'].isin(region_movie_tconsts)]
                        region_akas = data['akas'][(data['akas']['tconst'].isin(region_movies_details['tconst'])) & (
                                    data['akas']['region'] == selected_region)]
                        final_df = pd.merge(
                            region_movies_details[['tconst', 'primaryTitle', 'startYear', 'averageRating']],
                            region_akas[['tconst', 'title']].drop_duplicates(subset=['tconst']), on='tconst',
                            how='left'
                        )

                        search_query = st.text_input(
                            f"在 **{selected_region}** 的结果中按标题搜索 🔎",
                            placeholder="例如：Dark Knight, Inception..."
                        )

                        display_df = final_df.rename(
                            columns={'primaryTitle': '原始标题', 'title': '当地译名', 'startYear': '年份',
                                     'averageRating': '评分'}
                        )

                        if search_query:
                            mask = display_df['原始标题'].str.contains(search_query, case=False, na=False) | \
                                   display_df['当地译名'].str.contains(search_query, case=False, na=False)
                            filtered_display_df = display_df[mask]
                        else:
                            filtered_display_df = display_df

                        st.write(f"在 **{selected_region}** 共找到 **{len(filtered_display_df)}** 部相关电影:")
                        st.dataframe(
                            filtered_display_df[['原始标题', '当地译名', '年份', '评分']],
                            use_container_width=True
                        )

# --- Tab 3: 行业网络与人物分析 ---
with tab3:
    st.header("👥 行业网络与人物分析")
    st.markdown("通过设置**最小作品数**筛选核心影人，再利用**模糊搜索**快速定位并分析其职业生涯与合作网络。")
    min_movie_count = 15

    person_list_df = data['names'].reset_index()
    person_list_df.dropna(subset=['primaryName', 'nconst', 'movie_count'], inplace=True)
    filtered_person_df = person_list_df[person_list_df['movie_count'] >= min_movie_count]

    def search_persons(search_term: str):
        if not search_term:
            recommendations = filtered_person_df.nlargest(5, 'movie_count')
            return (recommendations['primaryName'] + " (" + recommendations['nconst'] + ")").tolist()
        lower_search_term = search_term.lower()
        results = filtered_person_df[filtered_person_df['primaryName'].str.lower().str.contains(lower_search_term)]
        return (results['primaryName'] + " (" + results['nconst'] + ")").tolist()

    st.markdown(f"##### 🔍 搜索影人 (当前已筛选出 {len(filtered_person_df):,} 位作品数不低于 {min_movie_count} 的影人)")
    selected_display_name = st_searchbox(search_function=search_persons,
                                         placeholder="输入'spielberg' 试试",
                                         label="搜索一位演员或导演:", key="person_searchbox_filtered",
                                         default="Steven Spielberg (nm0000229)")

    if selected_display_name:
        match = re.search(r'\((\w+)\)$', selected_display_name)
        if not match:
            st.warning("请从搜索结果中选择一个有效的影人条目。")
        else:
            person_nconst, selected_person_name = match.group(1), selected_display_name.rsplit(' (', 1)[0]
            person_info = person_list_df[person_list_df['nconst'] == person_nconst].iloc[0]
            st.subheader(f"🎞️ {selected_person_name} 的分析报告")
            st.metric("收录作品总数", f"{int(person_info['movie_count'])} 部")

            st.markdown("#### 🚀 职业生涯轨迹")
            with st.spinner(f"正在查询 {selected_person_name} 的职业生涯数据..."):
                career_df = data['career'][data['career']['nconst'] == person_nconst]
                if not career_df.empty:
                    fig_career = px.scatter(career_df, x='startYear', y='averageRating', size='numVotes',
                                            color='numVotes', hover_name='primaryTitle',
                                            title="作品评分与影响力随年份变化",
                                            labels={'startYear': '年份', 'averageRating': 'IMDB评分',
                                                    'numVotes': '投票数'},
                                            color_continuous_scale=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_career, use_container_width=True)
                else:
                    st.info(f"未在数据集中找到 {selected_person_name} 的职业生涯数据。")

            st.markdown("#### 🤝 合作网络")
            with st.spinner(f"正在查询 {selected_person_name} 的合作数据..."):
                part1, part2 = data['collab'][data['collab']['person1'] == person_nconst], data['collab'][
                    data['collab']['person2'] == person_nconst]
                if not part1.empty: part1 = part1.rename(columns={'person2': 'collaborator_nconst'})
                if not part2.empty: part2 = part2.rename(columns={'person1': 'collaborator_nconst'})
                all_collabs = pd.concat([part1, part2])
                if not all_collabs.empty:
                    all_collabs['collaborator_name'] = all_collabs['collaborator_nconst'].map(
                        data['names']['primaryName'])
                    all_collabs.dropna(subset=['collaborator_name'], inplace=True)
                    top_collabs = all_collabs.groupby('collaborator_name')['count'].sum().nlargest(15)
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("##### 🏆 合作最频繁的影人")
                        fig_collab = px.bar(top_collabs, x=top_collabs.values, y=top_collabs.index, orientation='h',
                                            labels={'x': '合作次数', 'y': '影人姓名'},
                                            color=top_collabs.values,
                                            color_continuous_scale=px.colors.sequential.Teal)
                        fig_collab.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_collab, use_container_width=True)
                    with col2:
                        st.markdown("##### 🌐 交互式合作网络")
                        st.markdown(
                            "<small>网络图展示了与中心人物合作最频繁的影人。<b>节点大小</b>和<b>连线粗细</b>代表合作次数的多少。您可以使用图下方的控件调整布局。</small>",
                            unsafe_allow_html=True
                        )

                        net = Network(height="450px", width="100%", notebook=True, directed=False,
                                      cdn_resources='in_line')
                        net.barnes_hut(gravity=-3000, central_gravity=0.25, spring_length=200, spring_strength=0.05,
                                       damping=0.09)
                        central_node_title = f"""
                        <b>{selected_person_name}</b><br>
                        (中心人物)<br>
                        收录作品: {int(person_info['movie_count'])} 部
                        """
                        net.add_node(selected_person_name, label=selected_person_name, color="#FF4B4B", size=30,
                                     title=central_node_title, font={'size': 20})

                        for name, count in top_collabs.items():
                            node_size = 12 + count * 2.5
                            collaborator_title = f"""
                            <b>{name}</b><br>
                            与 {selected_person_name} 合作: {count} 次
                            """
                            net.add_node(name, label=name, size=node_size, title=collaborator_title,
                                         color="#66CDAA")
                            edge_title = f"合作 {count} 次"
                            net.add_edge(selected_person_name, name, value=count, title=edge_title)
                        # net.show_buttons(filter_=['physics'])

                        try:
                            net.save_graph("network.html")
                            with open("network.html", "r", encoding="utf-8") as f:
                                html_code = f.read()
                            components.html(html_code, height=500)
                        finally:
                            if os.path.exists("network.html"):
                                os.remove("network.html")
                else:
                    st.info(f"未在Top电影的合作关系数据库中找到 {selected_person_name} 的数据。")
    else:
        st.info("⬆️ 请在上面的搜索框中输入影人姓名进行搜索和选择。")


# --- Tab 4: 影评文本分析 ---
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CUSTOM_STOPWORDS = {
    "movie", "movie", "film", "films", "one", "show", "see", "watch", "time", "story",
    "character", "characters", "really", "make", "even", "get", "like",
    "acting", "plot", "director"
}
WORDCLOUD_STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)
SKLEARN_STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

@st.cache_data
def generate_wordcloud(text_series, title):
    sample_size = min(len(text_series.dropna()), 2000)
    text = ' '.join(review for review in text_series.dropna().sample(sample_size, random_state=42))
    if not text: return None
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        collocations=False,
        stopwords=WORDCLOUD_STOPWORDS
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontdict={'fontsize': 20})
    ax.axis('off')
    return fig


@st.cache_data
def get_top_ngrams(corpus, ngram_range=(2, 2), top_k=20):
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=list(SKLEARN_STOPWORDS)
    ).fit(corpus.dropna())
    bag_of_words = vec.transform(corpus.dropna())
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=['N-gram', '数量'])


with tab4:
    st.header("📝 影评文本分析")
    analysis_type = st.radio("选择分析类型:", ["词云图", "高频词组 (N-grams)", f"🤖 {st.session_state.selected_model.split(' ')[0]} AI 总结"], horizontal=True,
                             key="tab4_radio")

    positive_reviews = data['reviews'][data['reviews']['sentiment'] == 'positive']['review_cleaned']
    negative_reviews = data['reviews'][data['reviews']['sentiment'] == 'negative']['review_cleaned']

    if analysis_type == "词云图":
        st.markdown("通过词云图，直观对比正面与负面评论中的高频词汇。（已移除'movie', 'film'等通用词）")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 👍 正面评论高频词云")
            with st.spinner("正在生成正面词云..."):
                fig_pos = generate_wordcloud(positive_reviews, "正面评论高频词")
                if fig_pos: st.pyplot(fig_pos)
                else: st.warning("无正面评论数据。")
        with col2:
            st.markdown("##### 👎 负面评论高频词云")
            with st.spinner("正在生成负面词云..."):
                fig_neg = generate_wordcloud(negative_reviews, "负面评论高频词")
                if fig_neg: st.pyplot(fig_neg)
                else: st.warning("无负面评论数据。")

    elif analysis_type == "高频词组 (N-grams)":
        st.markdown("通过分析高频词组（二元组），发现更具意义的表达方式。（已移除'movie', 'film'等通用词）")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 👍 正面评论高频词组")
            with st.spinner("正在分析正面词组..."):
                pos_ngrams = get_top_ngrams(positive_reviews, ngram_range=(2, 2), top_k=15)
                fig = px.bar(pos_ngrams, x='数量', y='N-gram', orientation='h', title="Top 15 正面评论二元组", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("##### 👎 负面评论高频词组")
            with st.spinner("正在分析负面词组..."):
                neg_ngrams = get_top_ngrams(negative_reviews, ngram_range=(2, 2), top_k=15)
                fig = px.bar(neg_ngrams, x='数量', y='N-gram', orientation='h', title="Top 15 负面评论二元组", color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_type.endswith("AI 总结"):
        current_model_name = st.session_state.selected_model.split(' ')[0]
        st.markdown(f"利用 **{current_model_name}** 大模型，从海量评论中提炼核心观点，洞察观众的真实想法。")
        sample_size = 20
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 👍 正面评论主题总结")
            with st.spinner(f"🤖 {current_model_name} 正在阅读正面评论并提炼观点..."):
                pos_sample = "\n".join(positive_reviews.dropna().sample(min(len(positive_reviews.dropna()), sample_size), random_state=1))
                prompt = f"""
                作为一名专业的电影评论分析师，请根据以下{sample_size}条真实的正面影评，总结出观众们主要称赞的3到5个核心主题（例如：演员演技、导演风格、剧情转折、视觉效果、情感共鸣等）。
                对于每个主题，请用一句话概括，并引用1-2句最能代表该观点的原话作为例子。请使用Markdown格式化你的回答，重点加粗。

                待分析的正面评论：
                ---
                {pos_sample}
                ---
                """
                summary = call_llm(prompt, st.session_state.selected_model)
                st.markdown(summary)

        with col2:
            st.markdown("##### 👎 负面评论主题总结")
            with st.spinner(f"🤖 {current_model_name} 正在阅读负面评论并寻找槽点..."):
                neg_sample = "\n".join(negative_reviews.dropna().sample(min(len(negative_reviews.dropna()), sample_size), random_state=1))
                prompt = f"""
                作为一名专业的电影评论分析师，请根据以下{sample_size}条真实的负面评论，总结出观众们主要抱怨的3到5个核心问题（例如：剧情空洞、角色塑造失败、节奏拖沓、特效廉价等）。
                对于每个问题，请用一句话概括，并引用1-2句最能代表该观点的原话作为例子。请使用Markdown格式化你的回答，重点加粗。

                待分析的负面评论：
                ---
                {neg_sample}
                ---
                """
                summary = call_llm(prompt, st.session_state.selected_model)
                st.markdown(summary)

# --- Tab 5: 影评智能剖析 ---
with tab5:
    st.header("🤖 影评智能剖析")
    st.markdown("#### 1. 基于本地模型的简单情感分析")
    st.markdown("输入一段影评，在IMDB数据集上微调的distilbert模型将进行情感判断。")
    with st.spinner("加载情感分析模型..."):
        sentiment_pipeline = load_sentiment_pipeline()

    user_input_simple = st.text_area("在此输入英文影评文本(训练集为英文文本):",
                              "This movie was not just bad, it was a catastrophe. The acting felt wooden and the plot was full of holes.",
                              height=150, key="simple_analysis_input")

    if st.button("进行简单情感分析"):
        if user_input_simple:
            with st.spinner("本地模型分析中..."):
                result = sentiment_pipeline(user_input_simple)[0]
                if result['label'].startswith('POSITIVE') or result['label'] == 'LABEL_1':
                    st.success(f"分析结果：正面情感 (置信度: {result['score']:.2%})")
                else:
                    st.error(f"分析结果：负面情感 (置信度: {result['score']:.2%})")
        else:
            st.warning("请输入文本后再进行分析。")

    st.markdown("---")
    current_model_name_tab5 = st.session_state.selected_model.split(' ')[0]
    st.markdown(f"#### 2. 基于 {current_model_name_tab5} 的多维度深度分析")
    st.markdown(f"输入一段影评，**{current_model_name_tab5}** 将为您提供多维度的深度分析报告，而不仅仅是简单的情感判断。")

    user_input_deep = st.text_area("在此输入影评文本:",
                              "**这部电影简直太棒了，堪称一场视觉与心灵的盛宴！** 虽然剧情节奏稍慢，但演员的表演堪称完美，情感真挚动人，情节设计更是环环相扣、引人入胜。每一个镜头都充满艺术感，配乐更是点睛之笔。真庆幸没有错过这部杰作，两个小时的观影体验让人意犹未尽！",
                              height=150, key="deep_analysis_input")

    if st.button("🚀 开始深度分析"):
        if user_input_deep:
            prompt = f"""
            你是一位专业的影评人。请分析以下影评，并以 Markdown 格式提供结构化的分析报告。
            你的分析报告应包含以下部分：

            - **📝 总体情绪**：请清晰地说明该影评的评价是“正面”、“负面”还是“褒贬不一”。
            - **🎭 关键方面分析**：请识别影评中提到的电影的具体方面（例如，剧情、演技、摄影、节奏、配乐）。对于每个方面，请简要描述影评人的观点，并为其分配相应的情绪（“正面”、“负面”、“中性”）。
            - **💡 核心信息**：请用一句话概括影评人的主要观点。

            以下是需要分析的影评：
            ---
            "{user_input_deep}"
            ---
            """
            with st.spinner(f"🤖 {current_model_name_tab5} 正在进行深度剖析..."):
                analysis_result = call_llm(prompt, st.session_state.selected_model)
                st.markdown("---")
                st.subheader(f"🔍 {current_model_name_tab5} 深度分析报告")
                st.markdown(analysis_result)
        else:
            st.warning("请输入文本后再进行分析。")

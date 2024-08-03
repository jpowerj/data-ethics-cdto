# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: orig-wordcloud
#| fig-align: center
import spacy
nlp = {
  'en': spacy.load("en_core_web_sm"),
  'uk': spacy.load("uk_core_news_sm")
}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
portfolio_df = pd.read_csv("Student_portfolios.csv")
portfolio_df = portfolio_df[~pd.isna(portfolio_df['Bio'])].copy()
bio_list_en = list(portfolio_df['Bio'].values)
# Preprocess bios individually
def clean_bios(raw_bio_list, lang):
  cleaned_sents = []
  lang_nlp = nlp[lang]
  for cur_bio in raw_bio_list:
    bio_doc = lang_nlp(cur_bio)
    bio_tokens = [t.text for t in bio_doc if not t.is_stop]
    bio_sent = " ".join(bio_tokens)
    cleaned_sents.append(bio_sent)
  return cleaned_sents
cleaned_bio_list_en = clean_bios(bio_list_en, 'en')
def combine_bios(bio_list):
  bio_str = "\n".join(bio_list)
  return bio_str
bio_str_en = combine_bios(cleaned_bio_list_en)
wc_options = {
  'background_color': 'white',
  'font_path': 'assets/Roboto.ttf',
  'scale': 8,
  'random_state': 2024,
  'width': 500,
  'height': 400,
}
def display_wordcloud(text_str):
  cloud = WordCloud(**wc_options)
  cloud.generate(text_str)
  plt.imshow(cloud, interpolation='bilinear')
  plt.axis("off")
display_wordcloud(bio_str_en)
#
#
#
#
#
#
#| label: orig-wordcloud-uk
#| fig-align: center
bio_list_uk = list(portfolio_df['Bio_uk'].values)
cleaned_bio_list_uk = clean_bios(bio_list_uk, 'uk')
bio_str_uk = combine_bios(cleaned_bio_list_uk)
display_wordcloud(bio_str_uk)
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: jeff-bio-en
jj_sent_en = "Jeff teaches Data Ethics at Georgetown University."
jj_bio_en = " ".join([jj_sent_en] * 100)
print(jj_bio_en[:230] + "...")
#
#
#
#| label: wordcloud-jeff-bio-en
#| fig-align: center
bio_list_fake_en = bio_list_en.copy()
bio_list_fake_en.append(jj_bio_en)
bio_str_fake_en = combine_bios(bio_list_fake_en)
bio_cloud_fake_en = WordCloud(**wc_options)
bio_cloud_fake_en.generate(bio_str_fake_en)
plt.imshow(bio_cloud_fake_en, interpolation='bilinear')
plt.axis("off")
#
#
#
#
#
#
#| label: jeff-bio-uk
jj_sent_uk = "Джеф викладає етику даних у Джорджтаунському університеті."
jj_bio_uk = " ".join([jj_sent_uk] * 100)
print(jj_bio_uk[:200] + "...")
#
#
#
#| label: wordcloud-jeff-bio-ukr
#| fig-align: center
bio_list_fake_uk = bio_list_uk.copy()
bio_list_fake_uk.append(jj_bio_uk)
bio_str_fake_uk = combine_bios(bio_list_fake_uk)
bio_cloud_fake_uk = WordCloud(**wc_options)
bio_cloud_fake_uk.generate(bio_str_fake_uk)
plt.imshow(bio_cloud_fake_uk, interpolation='bilinear')
plt.axis("off")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: normalize-freqs-en
# Each bio's freqs should count 1/N in overall
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
cv_en = CountVectorizer(
  min_df=1,
  stop_words='english'
)
dtm_en_sparse = cv_en.fit_transform(bio_list_fake_en)
dtm_en = dtm_en_sparse.todense()
#print(dtm.shape)
row_sums_en = dtm_en.sum(axis=1)
#print(row_sums.shape)
dtm_r0_en = dtm_en[0,:]
r0_sum_en = dtm_r0_en.sum()
r0_norm_en = dtm_r0_en / r0_sum_en
# Now use `normalize()`
dtm_norm_en = normalize(np.asarray(dtm_en), axis=1, norm='l1')
# And compute col sums of normalized DTM
dtm_colsums_en = dtm_norm_en.sum(axis=0)
# And combine with the words in a df
freq_df_en = pd.DataFrame({
  'term': cv_en.get_feature_names_out(),
  'freq': dtm_colsums_en
})
#freq_df['term'].values
#
#
#
#| label: normalized-wordcloud-en
#| fig-align: center
freq_dict_en = freq_df_en.set_index('term').to_dict(orient='dict')['freq']
bio_cloud_eq_en = WordCloud(**wc_options)
bio_cloud_eq_en.generate_from_frequencies(freq_dict_en)
plt.imshow(bio_cloud_eq_en, interpolation='bilinear')
plt.axis("off")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

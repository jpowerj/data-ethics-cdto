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
from PIL import ImageFont
roboto_url = "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth,wght%5D.ttf"
svg_start = """
<svg xmlns="http://www.w3.org/2000/svg">
    <defs>
    <style type="text/css">
    @import url("https://fonts.googleapis.com/css2?family=Roboto");
    text {{ font-family: \'Roboto\', sans-serif;
    font-kerning:none;
    font-variant-ligatures:none
    }}
    </style>
    </defs>
"""
def to_svg_custom(cloud_obj):
  svg_str = svg_start
  for (word, count), font_size, position, orientation, color in cloud_obj.layout_:
    x = position[0]
    y = position[1]
    
    font = ImageFont.truetype(cloud_obj.font_path, font_size)
            
    ascent, descent = font.getmetrics()
    
    """
    from stackoverflow - doesn't seem to be according to PIL docs (should return height, width) but doesn't work otherwise...
    https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pil-imagefont
    """
    (getsize_width, baseline), (offset_x, offset_y) = font.font.getsize(word)
    
    """
    svg transform string - empty if no rotation (text horizontal), otherwise contains rotate and translate numbers
    """
    svgTransform = ""    
    
    svgFill = ' fill="{}"'.format(color)    
        
    """
    this is all it takes to transform x,y to svg space 
    it was arrived at using the methods of computer graphics programmers
    https://twitter.com/erkaman2/status/1104105232034861056
    """
    if orientation is None:
        svgX = y - offset_x
        svgY = x + ascent - offset_y      
        
    else:
        svgX = y + ascent - offset_y
        svgY = x + offset_x
        svgTransform = ' transform="rotate(-90, {}, {}) translate({}, 0)"'.format(svgX, svgY, -getsize_width)

    """
    print SVG to standard output 
    """
    svg_str += ('<text x="{}" y="{}" font-size="{}"{}{}>{}</text>'.format(svgX, svgY, font_size, svgTransform, svgFill, word))
  return svg_str + "</svg>"
#
#
#
#| label: orig-wordcloud
#| fig-align: center
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import SVG
portfolio_df = pd.read_excel("Student_portfolios.xlsx")
portfolio_df = portfolio_df[~pd.isna(portfolio_df['Bio'])].copy()
#portfolio_df.head()
def combine_bios(bio_list):
  bio_str = "\n".join(bio_list)
  return bio_str
main_bio_list = list(portfolio_df['Bio'].values)
bio_str = combine_bios(main_bio_list)
def display_wordcloud(text_str):
  cloud = WordCloud(
    background_color='white',
    font_path='assets/Roboto.ttf',
    scale=10
  )
  cloud.generate(bio_str)
  #svg_content = to_svg_custom(cloud)
  #with open('test.svg', 'w', encoding='utf-8') as outfile:
  #  outfile.write(svg_content)
  #plt.figure(figsize=(10,8), dpi=300)
  plt.imshow(cloud, interpolation='bilinear')
  plt.axis("off")
display_wordcloud(bio_str)
#SVG("test.svg")
#plt.imshow(cloud, interpolation='bilinear')
#plt.axis("off")
#
#
#
#
#
#
#
jj_sent = "Jeff teaches Data Ethics at Georgetown University."
jj_bio = " ".join([jj_sent] * 100)
print(jj_bio[:500] + "...")
#
#
#
#| label: wordcloud-jeff-bio
#| fig-align: center
bio_list_fake = main_bio_list.copy()
bio_list_fake.append(jj_bio)
bio_str_fake = combine_bios(bio_list_fake)
bio_cloud = WordCloud(background_color='white')
bio_cloud.generate(bio_str_fake)
plt.imshow(bio_cloud, interpolation='bilinear')
plt.axis("off")
#
#
#
#
#
#
#
#
# Each bio's freqs should count 1/N in overall
#print(bio_list_fake)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(
  min_df=1,
  stop_words='english'
)
result = cv.fit_transform(bio_list_fake)
dtm = result.todense()
#print(dtm.shape)
row_sums = dtm.sum(axis=1)
#print(row_sums.shape)
dtm_r0 = dtm[0,:]
r0_sum = dtm_r0.sum()
r0_norm = dtm_r0 / r0_sum
# Now with `normalize()`
from sklearn.preprocessing import normalize
dtm_norm = normalize(np.asarray(dtm), axis=1, norm='l1')
# And compute col sums of normalized DTM
dtm_colsums = dtm_norm.sum(axis=0)
# And combine with the words in a df
freq_df = pd.DataFrame({
  'term': cv.get_feature_names_out(),
  'freq': dtm_colsums
})
#freq_df['term'].values
#
#
#
#| label: fake-bio-wordcloud-normalized
#| fig-align: center
freq_dict = freq_df.set_index('term').to_dict(orient='dict')['freq']
bio_cloud_eq = WordCloud(background_color='white')
bio_cloud_eq.generate_from_frequencies(freq_dict)
plt.imshow(bio_cloud_eq, interpolation='bilinear')
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

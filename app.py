import streamlit as st
import numpy as np
import pandas as pd
import time

import os
import openai

from download_pd import download_link
from transformers import GPT2TokenizerFast
from PIL import Image

st.set_page_config(page_title="Classifier",page_icon="🧊",layout="wide")

col1, col4, col2, col3 = st.beta_columns([3, 2, 7, 4])

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


image = Image.open('./img/teleperformance_logo.png')
col1.image(image)
#image = Image.open('./img/TAP_logo.png')
#st.sidebar..image(image)

container = st.sidebar.beta_container()

OPENAI_API_KEY = container.text_input('OpenAI API Key', type='password')
if not OPENAI_API_KEY:
	container.warning('Please input an API KEY.')

st.sidebar.write("#")

openai.api_key = OPENAI_API_KEY

st.markdown("""
<style>
.big-font {
    font-size:15px !important;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown(
"""
<p class="big-font">
Using OpenAI GPT-3 we classify the verbatims that come from EDF surveys.
The verbatims are automatically classified into 5 categories: </p>

<ul>
<li style="font-size:12px"> Effort client </li>
<li style="font-size:12px"> CC </li>
<li style="font-size:12px"> EDF </li>
<li style="font-size:12px"> Enedis </li>
<li style="font-size:12px"> Problème technique </li>
</ul> 

<p class="big-font">
To get the prediction, you need to upload a <b>.xlsx</b> file containing the verbatims in a
column named <b>text</b>.
</p>
""", unsafe_allow_html=True)


colE, colP, colS = st.beta_columns([3, 9, 3])

colP.title('EDF verbatims classifier')


id_file = 'file-X5Ka6IToHB1IbpewH7MawH4u'

uploaded_file = colP.file_uploader("Verbatims", help="Upload the file containing the verbatims you want to classify", type=['.xlsx'])

if uploaded_file is not None:
	with st.spinner('Wait for it...'):
		df = pd.read_excel(uploaded_file)

	colP.success("Done! We found %d verbatims in the dataset."%len(df))

	if colP.checkbox('Show dataframe'):
		df

	dic = {"Solution apportée": "Solution",\
	       "Effort client": "Eff",\
	       "CC ": "Cons",\
	       "EDF": "Client",\
	       "Enedis": "Prest",\
	       "Problème technique": "Technique"}

	labels = ['Eff', 'Cons', 'Solution', 'Client', 'Technique', 'Prest']
	labels_tokens = {'Eff': [27848],'Cons': [3515],'Solution': [28186],'Client': [20985],'Technique': [41317],'Prest': [24158]}

	df_result = pd.DataFrame(df)
	df_result = df_result.reset_index()
	df_result = df_result.drop('index', axis=1)

	df_result['pred_1'] = " "
	df_result['pred_2'] = " "
	df_result['pred_3'] = " "
	df_result['score'] = 0


	if colP.button('Classify %d verbatims'%len(df)):
		my_bar = colP.progress(0)
		for i in range(len(df)):
		    verb = df['text'].iloc[i]
		    
		    result = openai.Classification.create(
		        file=id_file,
		        query=verb,
		        search_model="ada",
		        model="curie",
		        max_examples=10,
		        labels=labels,
		        logprobs=len(labels) + 1,  # Here we set it to be len(labels) + 1, but it can be larger.
		        expand=["completion"],
		    )
		    
		    first_token_to_label = {tokens[0]: label for label, tokens in labels_tokens.items()}

		    top_logprobs = result["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
		    token_probs = {
		        tokenizer.encode(token)[0]: np.exp(logp) 
		        for token, logp in top_logprobs.items()
		    }
		    label_probs = {
		        first_token_to_label[token]: prob 
		        for token, prob in token_probs.items()
		        if token in first_token_to_label
		    }

		    if sum(label_probs.values()) < 1.0:
		        label_probs["Unknown"] = 1.0 - sum(label_probs.values())
		    
		    sorted_label = list(dict( sorted(label_probs.items(), key=lambda item: item[1],reverse=True)).keys())
		    sorted_prob = list(dict( sorted(label_probs.items(), key=lambda item: item[1],reverse=True)).values())
		    
		    
		    df_result['pred_1'].iloc[i] = sorted_label[0]
		    df_result['pred_2'].iloc[i] = sorted_label[1]
		    df_result['pred_3'].iloc[i] = sorted_label[2]
		    df_result['score'].iloc[i] = sorted_prob[0]
		    my_bar.progress(((i+1)/len(df)))
	    
		colP.write(df_result)

		tmp_download_link = download_link(df_result, 'YOUR_DF.csv', 'Click here to download your data!')
		colP.markdown(tmp_download_link, unsafe_allow_html=True)


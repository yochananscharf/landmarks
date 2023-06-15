
import streamlit as st

import json
from PIL import Image


st.set_page_config(layout="wide")
st.title('Image Retrieval App') #:sunglasses:')

col1, col2 = st.columns(2)

#BASE_PATH = 'D:/dl_databases_new/global_ladybug/'#'D:/data/mapillary/map_data/tiles/'


if 'count_pair' not in st.session_state:

    
    #xml = json_file.read()
    #st.session_state.count_pair = 0
    st.session_state.vector_dict = None
    st.session_state.json_file = None
    st.session_state.path_anchor = ''
    st.session_state.path_positive = ''
    st.session_state.tagging_path = ''
    st.session_state.num_pairs = 0
    #f = open(json_file)
    # 



    

if  st.session_state.vector_dict != None:

    #st.session_state.image_query = st.file_uploader('upload image')
    image_path = st.file_uploader('Please Choose an Image', type=['jpeg', 'png', 'jpg'])
    #mage_matcher.query_path = image_path
    if image_path is not None:
        image = Image.open(image_path).convert('RGB') 
        st.image(image)
        vec = predict_query_image(image)
        get_top_n()

    with st.form('pair ?', clear_on_submit=True):
        st.session_state.count_pair = st.slider('progress', min_value=0, max_value = st.session_state.num_pairs, value=st.session_state.count_pair)
        with st.sidebar:
            #correct = False

            correct = st.checkbox('correct', value=True)
            sub = st.form_submit_button('next')

        pair = st.session_state.anchor_positive_pairs[st.session_state.count_pair]
        path_anchor = pair[0]
        path_positive = pair[1]
        image_anchor = Image.open(BASE_PATH+path_anchor)
        image_positive = Image.open(BASE_PATH+path_positive)
        col1.image(image_anchor, width=640)
        col2.image(image_positive, width=640)
        with st.sidebar:
            st.write(path_anchor)
            st.write(path_positive)
        st.session_state.path_anchor = path_anchor
        st.session_state.path_positive = path_positive


        
    st.session_state.count_pair += 1
else:
    #json_filePath = 'pairs_tagged_29_3_new.json'#'D:/Data/mapillary/map_data/rml_triplets_174.json' #pairs_tagged_29_3_new.json
    st.session_state.json_file = st.file_uploader('choose vector database (.json)')
    #st.session_state.json_file = open(json_filePath)

    if st.session_state.json_file is not None:
        #st.session_state.tagging_path = "tagged_{}.txt".format(st.session_state.json_file.name.split('.')[0])
        st.session_state.vector_dict = json.load(st.session_state.json_file)

        # get first key to update images-dir
        #res = next(iter(st.session_state.vector_dict))

        # st.session_state.anchor_positive_pairs = []
        # for k, v in st.session_state.vector_dict.items():
        #     if isinstance(v,str):
        #         st.session_state.anchor_positive_pairs.append((k, v))
        #     else:
        #         for pos in v[0]:
        #             if 'ladybug' in k:
        #                 st.session_state.anchor_positive_pairs.append((k, pos[0]))
        #             else:
        #                 st.session_state.anchor_positive_pairs.append((k.split('/')[-1], pos[0].split('/')[-1]))
        st.session_state.num_pairs = len(st.session_state.anchor_positive_pairs)
        st.experimental_rerun()
  
# if correct:
#     positive_tagging = (st.session_state.pos_path, st.session_state.pos_tile)
#     st.session_state.tagged_vals.append(positive_tagging)#v[st.session_state.count_pos]) 
# correct = False 
# st.session_state.pos_path = pos_path
# st.session_state.pos_tile = match_original
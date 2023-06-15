
import streamlit as st
from torch import hub
import torch
from torchvision import transforms
from PIL import Image
import json
from PIL import Image


st.set_page_config(layout="wide")
st.title('Image Retrieval App') #:sunglasses:')

col1, col2 = st.columns(2)

#BASE_PATH = 'D:/dl_databases_new/global_ladybug/'#'D:/data/mapillary/map_data/tiles/'
@st.experimental_singleton
def get_transform():
    resize = transforms.Resize((512,512))
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return transform

if 'count' not in st.session_state:
    model = hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model.load_state_dict(hub.load_state_dict_from_url(url=
            "https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))

    model = model.cuda()
    model.eval()
    transform = get_transform()
    #xml = json_file.read()
    st.session_state.count = 0
    st.session_state.vector_dict = None
    st.session_state.json_file = None
    st.session_state.vector_keys = ''
    st.session_state.vector_vals = ''
    #st.session_state.tagging_path = ''
    #st.session_state.num_pairs = 0
    #f = open(json_file)
    # 
#def get_top_n():



def predict_query_image(img):
    #img = Image.open(dir + img_path)
    img_tensor = transform(img)
    with torch.no_grad():
        features_vec = model(img_tensor)
    

if  st.session_state.vector_dict != None:

    #st.session_state.image_query = st.file_uploader('upload image')
    image_path = st.file_uploader('Please Choose an Image', type=['jpeg', 'png', 'jpg'])
    #mage_matcher.query_path = image_path
    if image_path is not None:
        image = Image.open(image_path).convert('RGB') 
        st.image(image)
        #vec = predict_query_image(image)
        #get_top_n()

    # with st.form('pair ?', clear_on_submit=True):
    #     st.session_state.count = st.slider('progress', min_value=0, max_value = st.session_state.num_pairs, value=st.session_state.count_pair)
    #     with st.sidebar:
    #         #correct = False

    #         correct = st.checkbox('correct', value=True)
    #         sub = st.form_submit_button('next')

        # pair = st.session_state.anchor_positive_pairs[st.session_state.count_pair]
        # path_anchor = pair[0]
        # path_positive = pair[1]


        # image_anchor = Image.open(BASE_PATH+path_anchor)
        # image_positive = Image.open(BASE_PATH+path_positive)
        # col1.image(image_anchor, width=640)
        # col2.image(image_positive, width=640)
        
        # with st.sidebar:
            # st.write(path_anchor)
            # st.write(path_positive)
        # st.session_state.path_anchor = path_anchor
        # st.session_state.path_positive = path_positive


        
    st.session_state.count += 1
else:
    #json_filePath = 'pairs_tagged_29_3_new.json'#'D:/Data/mapillary/map_data/rml_triplets_174.json' #pairs_tagged_29_3_new.json
    st.session_state.json_file = st.file_uploader('choose vector database (.json)')
    #st.session_state.json_file = open(json_filePath)

    if st.session_state.json_file is not None:
        #st.session_state.tagging_path = "tagged_{}.txt".format(st.session_state.json_file.name.split('.')[0])
        st.session_state.vector_dict = json.load(st.session_state.json_file)
        st.session_state.vector_vals = torch.stack([torch.tensor(v) for v in st.session_state.vector_dict.values()]).cuda()
        st.session_state.vector_keys = [k for k in st.session_state.vector_dict.keys()]
        num_keys = len(st.session_state.vector_keys)
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
  


import streamlit as st
from torch import hub
import torch
from torchvision import transforms
from PIL import Image
import json
from PIL import Image


st.set_page_config(layout="wide")
st.header('Image Retrieval App', anchor='center') #:sunglasses:')

col1, col2 = st.columns(2)

IMAGES_DIR = 'C:/Users/yocha/data-mining/deep-learning/project/PhotoTourism/Train/'#'D:/data/mapillary/map_data/tiles/'
@st.cache_resource
def get_transform():
    resize = transforms.Resize((512,512))
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return transform

transform = get_transform()
pdist =  torch.nn.PairwiseDistance(p=2)


#@st.cache_resource
def distance_to_vectors(vec, vecs):
  output = pdist(vec, vecs)
  return output




if 'count' not in st.session_state:
    st.write('Loading model ...')
    st.session_state.model = hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    st.session_state.model.load_state_dict(hub.load_state_dict_from_url(url=
            "https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))

    #model = model.cuda()
    st.session_state.model.eval()

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
def get_top_n(vec):
    distances = distance_to_vectors(vec, st.session_state.vector_vals)
    distance_dict = {st.session_state.vector_keys[i]:distances[i] for i in range(st.session_state.num_keys)}
    dict_sorted = dict(sorted(distance_dict.items(), key=lambda w:w[1], reverse=False))
    imgs_sorted_list = list(dict_sorted.keys())
    top_match = imgs_sorted_list[0]
    return top_match


def predict_query_image(img):
    #img = Image.open(dir + img_path)
    img_tensor = transform(img)
    with torch.no_grad():
        features_vec = st.session_state.model(img_tensor.unsqueeze(0))
    return features_vec

if  st.session_state.vector_dict != None:

    #st.session_state.image_query = st.file_uploader('upload image')
    image_path = st.file_uploader('Please Choose an Image', type=['jpeg', 'png', 'jpg'])
    #mage_matcher.query_path = image_path
    if image_path is not None:
        image = Image.open(image_path).convert('RGB') 
       
        st.write('predicting ...')
        vec = predict_query_image(image)
        img_pos_name = get_top_n(vec)
        landmark_dir = st.session_state.image_dict[img_pos_name][0]
        image_pos = Image.open(IMAGES_DIR+landmark_dir+'/'+img_pos_name).convert('RGB') 
      
        col1.write('query-image')
        col1.image(image.resize((512,512)), width=512)
        col2.write('positive-image')
        col2.image(image_pos.resize((512,512)), width=512)
        
        # with st.sidebar:
            # st.write(path_anchor)
            # st.write(path_positive)
        # st.session_state.path_anchor = path_anchor
        # st.session_state.path_positive = path_positive


        
    st.session_state.count += 1
else:
    #json_filePath = 'pairs_tagged_29_3_new.json'#'D:/Data/mapillary/map_data/rml_triplets_174.json' #pairs_tagged_29_3_new.json
    #st.session_state.json_file = st.file_uploader('choose vector database (.json)')
    st.session_state.json_file = open('features_dict_landmarks.json')
    st.session_state.json_file_1 = open('img_to_landmark.json')

    if st.session_state.json_file is not None:
        #st.session_state.tagging_path = "tagged_{}.txt".format(st.session_state.json_file.name.split('.')[0])
        st.write('Loading vectors ...')
        st.session_state.vector_dict = json.load(st.session_state.json_file)
        st.session_state.vector_vals = torch.stack([torch.tensor(v) for v in st.session_state.vector_dict.values()])#.cuda()
        st.session_state.vector_keys = [k for k in st.session_state.vector_dict.keys()]
        st.session_state.num_keys = len(st.session_state.vector_keys)
        st.write('vectors loaded')
        st.session_state.image_dict = json.load(st.session_state.json_file_1)
        st.experimental_rerun()
  
